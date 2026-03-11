"""
Autoresearch pretraining script — Apple Silicon (MPS) version.
Single-GPU, single-file. Adapted for VHDL corpus.
Usage: uv run train.py
"""

import os
import gc
import random
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, TIME_BUDGET as _TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# Override time budget: 3 hours for deep training on M4
TIME_BUDGET = 10800

# ---------------------------------------------------------------------------
# GPT Model (MPS-compatible, no Flash Attention, no bfloat16)
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 32768
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int = 4
    n_embd: int = 128
    window_pattern: str = "SL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        cos, sin = cos_sin
        # Apply rotary embeddings (expects B, T, H, D — transpose back and forth)
        q_rot = apply_rotary_emb(q.transpose(1, 2), cos, sin).transpose(1, 2)
        k_rot = apply_rotary_emb(k.transpose(1, 2), cos, sin).transpose(1, 2)
        q, k = norm(q_rot), norm(k_rot)

        # Expand kv heads if grouped query attention
        if self.n_kv_head < self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # Standard scaled dot-product attention (MPS compatible)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = int(8 * config.n_embd / 3)
        # Round to multiple of 64 for efficiency
        hidden = ((hidden + 63) // 64) * 64
        self.c_gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_up = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.silu(self.c_gate(x)) * self.c_up(x))


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Rotary embeddings
        head_dim = config.n_embd // config.n_head
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_gate.weight, -s, s)
            torch.nn.init.uniform_(block.mlp.c_up.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device if hasattr(self.transformer.wte.weight, 'device') else 'cpu'
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.float(), sin.float()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        nparams_exclude = self.transformer.wte.weight.numel()
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = self.config.n_layer * 12 * h * q * t
        return 6 * (nparams - nparams_exclude) + attn_flops

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin)
        x = norm(x)

        logits = self.lm_head(x)
        logits = logits.float()

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 48       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 64           # target head dimension for attention

# Optimization
TOTAL_BATCH_SIZE = 2**14  # ~16K tokens per step — more optimizer steps in budget
LEARNING_RATE = 6e-4      # higher LR may work with 1-hour budget (more steps)
WEIGHT_DECAY = 0.0        # no decay for short training
ADAM_BETAS = (0.9, 0.95)
WARMUP_RATIO = 0.05
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0

# Model size — tuned for 16GB M4, ~20M params
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 16   # per-device batch size (halved for 16GB RAM)

# ---------------------------------------------------------------------------
# Helpers (importable by generate.py)
# ---------------------------------------------------------------------------

def build_model_config(depth, vocab_size=None):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    vs = vocab_size if vocab_size is not None else 32768
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vs,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
    )


VHDL_PROMPTS = [
    "library ieee;\nuse ieee.std_logic_1164.all;\n\nentity",
    "library ieee;\nuse ieee.std_logic_1164.all;\nuse ieee.numeric_std.all;\n\nentity alu is",
    "library ieee;\nuse ieee.std_logic_1164.all;\n\nentity counter is\n  port (\n    clk : in std_logic;\n    reset : in std_logic;\n    count : out std_logic_vector(7 downto 0)\n  );\nend entity counter;\n\narchitecture rtl of counter is",
    "library ieee;\nuse ieee.std_logic_1164.all;\n\nentity fifo is",
    "library ieee;\nuse ieee.std_logic_1164.all;\nuse ieee.numeric_std.all;\n\nentity state_machine is",
]


@torch.no_grad()
def generate_vhdl(model, tokenizer, prompt, device, max_new_tokens=512, temperature=0.8, top_k=50):
    """Autoregressive VHDL generation. prompt can be a string or list of token IDs."""
    if isinstance(prompt, list):
        ids = prompt
    else:
        ids = tokenizer.encode(prompt)
    if len(ids) >= MAX_SEQ_LEN:
        ids = ids[:MAX_SEQ_LEN - 1]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    bos_id = tokenizer.get_bos_token_id()

    for _ in range(max_new_tokens):
        x_cond = x if x.size(1) <= MAX_SEQ_LEN else x[:, -MAX_SEQ_LEN:]
        logits = model(x_cond)
        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        if next_token.item() == bos_id:
            break
        x = torch.cat([x, next_token], dim=1)

    code = tokenizer.decode(x[0].tolist())
    # Extract first complete VHDL unit (up to end of architecture/entity)
    import re
    m = re.search(r'(?i)(end\s+architecture\s*\w*\s*;|end\s+\w+\s*;)', code)
    if m:
        code = code[:m.end()]
    return code


def compile_vhdl(code, mode="analyze", timeout=10):
    """
    Try to compile VHDL with GHDL.
    mode="syntax"  -> ghdl -s (syntax check only, easier to pass)
    mode="analyze" -> ghdl -a (full analysis with type checking)
    Returns (success: bool, error_msg: str or None, error_count: int)
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.vhd', delete=False) as f:
        f.write(code)
        tmp_path = f.name
    try:
        flag = "-s" if mode == "syntax" else "-a"
        result = subprocess.run(
            ["ghdl", flag, "--std=08", tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        success = result.returncode == 0
        error = result.stderr.strip() if not success else None
        error_count = error.count(":error:") if error else 0
        return success, error, error_count
    except (subprocess.TimeoutExpired, FileNotFoundError, UnicodeDecodeError):
        return False, "ghdl error", 99
    finally:
        os.unlink(tmp_path)
        work_cf = os.path.join(os.getcwd(), "work-obj08.cf")
        if os.path.exists(work_cf):
            os.unlink(work_cf)

# ---------------------------------------------------------------------------
# Training (only runs when executed directly)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Compiler feedback: rejection sampling
# ---------------------------------------------------------------------------

# Instruction tuning
PRETRAIN_RATIO = 1.0            # all time for pretraining (instruction data mixed in)
INSTRUCTION_MIX_RATIO = 0.05   # 5% of micro-steps use instruction data
INST_TOKEN = "<|reserved_1|>"
RESP_TOKEN = "<|reserved_2|>"

GENERATE_BATCH = 8
GENERATE_MAX_TOKENS = 256
GENERATE_TEMPERATURE = 0.9

# (description_variants, vhdl_code) pairs for instruction tuning
# Each entry: (list_of_descriptions, vhdl_code_string)
INSTRUCTION_DATA = [
    (["an inverter", "a NOT gate", "an inverter gate", "a logic inverter"],
     """library ieee;
use ieee.std_logic_1164.all;

entity inverter is
  port (
    a : in std_logic;
    y : out std_logic
  );
end entity inverter;

architecture rtl of inverter is
begin
  y <= not a;
end architecture rtl;
"""),
    (["an AND gate", "a 2-input AND gate", "a logical AND gate"],
     """library ieee;
use ieee.std_logic_1164.all;

entity and_gate is
  port (
    a : in std_logic;
    b : in std_logic;
    y : out std_logic
  );
end entity and_gate;

architecture rtl of and_gate is
begin
  y <= a and b;
end architecture rtl;
"""),
    (["an 8-bit counter", "a counter with synchronous reset", "an 8-bit binary counter with reset"],
     """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity counter is
  port (
    clk   : in  std_logic;
    reset : in  std_logic;
    count : out std_logic_vector(7 downto 0)
  );
end entity counter;

architecture rtl of counter is
  signal cnt : unsigned(7 downto 0);
begin
  process(clk)
  begin
    if rising_edge(clk) then
      if reset = '1' then
        cnt <= (others => '0');
      else
        cnt <= cnt + 1;
      end if;
    end if;
  end process;
  count <= std_logic_vector(cnt);
end architecture rtl;
"""),
    (["a 2-to-1 multiplexer", "a 2-to-1 mux", "a mux with two inputs"],
     """library ieee;
use ieee.std_logic_1164.all;

entity mux2 is
  port (
    a   : in  std_logic;
    b   : in  std_logic;
    sel : in  std_logic;
    y   : out std_logic
  );
end entity mux2;

architecture rtl of mux2 is
begin
  y <= a when sel = '0' else b;
end architecture rtl;
"""),
    (["a D flip-flop", "a DFF", "a D-type flip-flop", "a single-bit register"],
     """library ieee;
use ieee.std_logic_1164.all;

entity dff is
  port (
    clk : in  std_logic;
    d   : in  std_logic;
    q   : out std_logic
  );
end entity dff;

architecture rtl of dff is
begin
  process(clk)
  begin
    if rising_edge(clk) then
      q <= d;
    end if;
  end process;
end architecture rtl;
"""),
    (["a register file", "a 4-entry register file", "a register file with 8-bit data"],
     """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity register_file is
  port (
    clk   : in  std_logic;
    we    : in  std_logic;
    addr  : in  std_logic_vector(1 downto 0);
    din   : in  std_logic_vector(7 downto 0);
    dout  : out std_logic_vector(7 downto 0)
  );
end entity register_file;

architecture rtl of register_file is
  type reg_array is array (0 to 3) of std_logic_vector(7 downto 0);
  signal regs : reg_array;
begin
  process(clk)
  begin
    if rising_edge(clk) then
      if we = '1' then
        regs(to_integer(unsigned(addr))) <= din;
      end if;
    end if;
  end process;
  dout <= regs(to_integer(unsigned(addr)));
end architecture rtl;
"""),
    (["an SR latch", "a set-reset latch", "an SR flip-flop"],
     """library ieee;
use ieee.std_logic_1164.all;

entity sr_latch is
  port (
    s : in  std_logic;
    r : in  std_logic;
    q : out std_logic
  );
end entity sr_latch;

architecture rtl of sr_latch is
  signal q_int : std_logic;
begin
  q_int <= '1' when s = '1' else
           '0' when r = '1' else
           q_int;
  q <= q_int;
end architecture rtl;
"""),
    (["a tristate buffer", "a tri-state buffer", "a tristate output driver"],
     """library ieee;
use ieee.std_logic_1164.all;

entity tristate is
  port (
    din : in  std_logic;
    en  : in  std_logic;
    dout : out std_logic
  );
end entity tristate;

architecture rtl of tristate is
begin
  dout <= din when en = '1' else 'Z';
end architecture rtl;
"""),
    (["a half adder", "a half-adder circuit", "a 1-bit half adder"],
     """library ieee;
use ieee.std_logic_1164.all;

entity half_adder is
  port (a, b : in std_logic; sum_out, carry : out std_logic);
end entity half_adder;

architecture rtl of half_adder is
begin
  sum_out <= a xor b;
  carry <= a and b;
end architecture rtl;
"""),
    (["a full adder", "a full-adder circuit", "a 1-bit full adder with carry"],
     """library ieee;
use ieee.std_logic_1164.all;

entity full_adder is
  port (a, b, cin : in std_logic; sum_out, cout : out std_logic);
end entity full_adder;

architecture rtl of full_adder is
begin
  sum_out <= a xor b xor cin;
  cout <= (a and b) or (cin and (a xor b));
end architecture rtl;
"""),
    (["an OR gate", "a 2-input OR gate", "a logical OR gate"],
     """library ieee;
use ieee.std_logic_1164.all;

entity or_gate is
  port (a, b : in std_logic; y : out std_logic);
end entity or_gate;

architecture rtl of or_gate is
begin
  y <= a or b;
end architecture rtl;
"""),
    (["an XOR gate", "a 2-input XOR gate", "an exclusive-OR gate"],
     """library ieee;
use ieee.std_logic_1164.all;

entity xor_gate is
  port (a, b : in std_logic; y : out std_logic);
end entity xor_gate;

architecture rtl of xor_gate is
begin
  y <= a xor b;
end architecture rtl;
"""),
    (["a NAND gate", "a 2-input NAND gate"],
     """library ieee;
use ieee.std_logic_1164.all;

entity nand_gate is
  port (a, b : in std_logic; y : out std_logic);
end entity nand_gate;

architecture rtl of nand_gate is
begin
  y <= not (a and b);
end architecture rtl;
"""),
    (["a NOR gate", "a 2-input NOR gate"],
     """library ieee;
use ieee.std_logic_1164.all;

entity nor_gate is
  port (a, b : in std_logic; y : out std_logic);
end entity nor_gate;

architecture rtl of nor_gate is
begin
  y <= not (a or b);
end architecture rtl;
"""),
    (["a 4-to-1 multiplexer", "a 4-to-1 mux", "a 4-input mux with 2-bit select"],
     """library ieee;
use ieee.std_logic_1164.all;

entity mux4 is
  port (
    a, b, c, d : in std_logic;
    sel : in std_logic_vector(1 downto 0);
    y : out std_logic
  );
end entity mux4;

architecture rtl of mux4 is
begin
  with sel select
    y <= a when "00",
         b when "01",
         c when "10",
         d when others;
end architecture rtl;
"""),
    (["an 8-bit comparator", "a comparator with equal, greater, less outputs"],
     """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity comparator is
  port (
    a : in std_logic_vector(7 downto 0);
    b : in std_logic_vector(7 downto 0);
    eq : out std_logic;
    gt : out std_logic;
    lt : out std_logic
  );
end entity comparator;

architecture rtl of comparator is
begin
  eq <= '1' when a = b else '0';
  gt <= '1' when unsigned(a) > unsigned(b) else '0';
  lt <= '1' when unsigned(a) < unsigned(b) else '0';
end architecture rtl;
"""),
    (["a clock divider", "a frequency divider", "a clock divider by 256"],
     """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity clock_divider is
  port (
    clk_in : in std_logic;
    reset : in std_logic;
    clk_out : out std_logic
  );
end entity clock_divider;

architecture rtl of clock_divider is
  signal counter : unsigned(7 downto 0);
  signal clk_reg : std_logic;
begin
  process(clk_in)
  begin
    if rising_edge(clk_in) then
      if reset = '1' then
        counter <= (others => '0');
        clk_reg <= '0';
      else
        if counter = 127 then
          counter <= (others => '0');
          clk_reg <= not clk_reg;
        else
          counter <= counter + 1;
        end if;
      end if;
    end if;
  end process;
  clk_out <= clk_reg;
end architecture rtl;
"""),
    (["an 8-bit shift register", "a serial-in parallel-out shift register", "a shift register with reset"],
     """library ieee;
use ieee.std_logic_1164.all;

entity shift_register is
  port (
    clk : in std_logic;
    reset : in std_logic;
    din : in std_logic;
    dout : out std_logic_vector(7 downto 0)
  );
end entity shift_register;

architecture rtl of shift_register is
  signal reg : std_logic_vector(7 downto 0);
begin
  process(clk)
  begin
    if rising_edge(clk) then
      if reset = '1' then
        reg <= (others => '0');
      else
        reg <= reg(6 downto 0) & din;
      end if;
    end if;
  end process;
  dout <= reg;
end architecture rtl;
"""),
    (["a 3-to-8 decoder", "a decoder with enable", "a 3-to-8 line decoder"],
     """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity decoder_3to8 is
  port (
    sel : in std_logic_vector(2 downto 0);
    en : in std_logic;
    y : out std_logic_vector(7 downto 0)
  );
end entity decoder_3to8;

architecture rtl of decoder_3to8 is
begin
  process(sel, en)
  begin
    y <= (others => '0');
    if en = '1' then
      y(to_integer(unsigned(sel))) <= '1';
    end if;
  end process;
end architecture rtl;
"""),
    (["a parity generator", "an 8-bit parity checker", "an even parity generator"],
     """library ieee;
use ieee.std_logic_1164.all;

entity parity_generator is
  port (
    din : in std_logic_vector(7 downto 0);
    parity : out std_logic
  );
end entity parity_generator;

architecture rtl of parity_generator is
begin
  parity <= din(0) xor din(1) xor din(2) xor din(3) xor din(4) xor din(5) xor din(6) xor din(7);
end architecture rtl;
"""),
    (["a buffer gate", "a unity-gain buffer", "a simple buffer"],
     """library ieee;
use ieee.std_logic_1164.all;

entity buffer_gate is
  port (
    a : in std_logic;
    y : out std_logic
  );
end entity buffer_gate;

architecture rtl of buffer_gate is
begin
  y <= a;
end architecture rtl;
"""),
    (["a JK flip-flop", "a JK flipflop", "a JK-type flip-flop"],
     """library ieee;
use ieee.std_logic_1164.all;

entity jk_flipflop is
  port (
    clk : in std_logic;
    j   : in std_logic;
    k   : in std_logic;
    q   : out std_logic
  );
end entity jk_flipflop;

architecture rtl of jk_flipflop is
  signal q_int : std_logic := '0';
begin
  process(clk)
  begin
    if rising_edge(clk) then
      if j = '1' and k = '1' then
        q_int <= not q_int;
      elsif j = '1' then
        q_int <= '1';
      elsif k = '1' then
        q_int <= '0';
      end if;
    end if;
  end process;
  q <= q_int;
end architecture rtl;
"""),
    (["a T flip-flop", "a toggle flip-flop", "a T-type flip-flop"],
     """library ieee;
use ieee.std_logic_1164.all;

entity t_flipflop is
  port (
    clk : in std_logic;
    t   : in std_logic;
    q   : out std_logic
  );
end entity t_flipflop;

architecture rtl of t_flipflop is
  signal q_int : std_logic := '0';
begin
  process(clk)
  begin
    if rising_edge(clk) then
      if t = '1' then
        q_int <= not q_int;
      end if;
    end if;
  end process;
  q <= q_int;
end architecture rtl;
"""),
]

# Instruction prompt templates for variety
INSTRUCTION_FORMATS = [
    "Design {desc}",
    "Write VHDL code for {desc}",
    "Create {desc} in VHDL",
    "Implement {desc}",
    "Write a VHDL module for {desc}",
    "Generate VHDL for {desc}",
    "Code {desc} in VHDL",
    "Build {desc} using VHDL",
    "Write the VHDL for {desc}",
    "{desc}",
]

# Keep VHDL_TEMPLATES for backward compatibility (generate.py uses it)
VHDL_TEMPLATES = [code for _, code in INSTRUCTION_DATA]


def build_instruction_pairs():
    """Build all (instruction_string, vhdl_code) pairs from templates + textbook data."""
    pairs = []
    # Built-in template pairs (with format variants)
    for desc_variants, code in INSTRUCTION_DATA:
        for desc in desc_variants:
            for fmt in INSTRUCTION_FORMATS:
                pairs.append((fmt.format(desc=desc), code))
    # Load textbook instruction pairs if available
    import json
    textbook_path = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "instruction_pairs.json")
    if os.path.exists(textbook_path):
        with open(textbook_path) as f:
            textbook_pairs = json.load(f)
        pairs.extend(textbook_pairs)
        print(f"Loaded {len(textbook_pairs)} textbook instruction pairs")
    return pairs


def get_instruction_batch(pairs, tokenizer, device, batch_size, seq_len):
    """Create a training batch from instruction pairs."""
    inst_id = tokenizer.enc.encode_single_token(INST_TOKEN)
    resp_id = tokenizer.enc.encode_single_token(RESP_TOKEN)
    bos_id = tokenizer.get_bos_token_id()

    row_len = seq_len + 1
    all_tokens = []
    while len(all_tokens) < batch_size * row_len * 2:
        desc, code = random.choice(pairs)
        tokens = [bos_id, inst_id] + tokenizer.encode(desc) + [resp_id] + tokenizer.encode(code)
        all_tokens.extend(tokens)

    num_rows = min(batch_size, len(all_tokens) // row_len)
    buf = torch.tensor(all_tokens[:num_rows * row_len], dtype=torch.long)
    buf = buf.view(num_rows, row_len)
    x = buf[:, :-1].to(device)
    y = buf[:, 1:].to(device)
    return x, y


@torch.no_grad()
def generate_vhdl_batch(model, tokenizer, device, n=GENERATE_BATCH,
                        max_new_tokens=GENERATE_MAX_TOKENS, temperature=GENERATE_TEMPERATURE):
    """Generate multiple VHDL samples from different prompts."""
    samples = []
    for i in range(n):
        prompt = VHDL_PROMPTS[i % len(VHDL_PROMPTS)]
        code = generate_vhdl(model, tokenizer, prompt, device,
                           max_new_tokens=max_new_tokens, temperature=temperature)
        samples.append(code)
    return samples


def score_vhdl(code):
    """
    Score generated VHDL on a 0-1 scale using progressive compiler checks.
    0.0 = total garbage
    0.3 = has VHDL structure but fails syntax
    0.6 = passes syntax check (ghdl -s)
    1.0 = passes full analysis (ghdl -a)
    """
    text_lower = code.lower()

    # Basic structural check: does it look like VHDL at all?
    has_library = "library" in text_lower
    has_entity = "entity" in text_lower
    has_arch = "architecture" in text_lower
    has_end = "end" in text_lower
    structural_score = sum([has_library, has_entity, has_arch, has_end]) / 4.0

    if structural_score < 0.5:
        return 0.0  # doesn't even look like VHDL

    # Syntax check (lenient)
    syntax_ok, _, syntax_errors = compile_vhdl(code, mode="syntax")
    if not syntax_ok:
        # Partial credit: fewer errors = better
        return 0.3 * structural_score * max(0, 1.0 - syntax_errors / 20.0)

    # Full analysis (strict)
    analyze_ok, _, analyze_errors = compile_vhdl(code, mode="analyze")
    if analyze_ok:
        return 1.0  # perfect score
    else:
        # Passed syntax but failed analysis — still pretty good
        return 0.6 + 0.4 * max(0, 1.0 - analyze_errors / 10.0)


def make_feedback_batch(good_samples, tokenizer, device, batch_size, seq_len):
    """
    Turn a list of VHDL strings into a training batch.
    Tokenize, pack into fixed-length sequences, return (x, y).
    """
    bos = tokenizer.get_bos_token_id()
    all_tokens = []
    for code in good_samples:
        ids = [bos] + tokenizer.encode(code)
        all_tokens.extend(ids)

    if len(all_tokens) < seq_len + 1:
        return None, None  # not enough tokens

    # Pack into batch
    row_len = seq_len + 1
    num_rows = min(batch_size, len(all_tokens) // row_len)
    if num_rows == 0:
        return None, None

    buf = torch.tensor(all_tokens[:num_rows * row_len], dtype=torch.long)
    buf = buf.view(num_rows, row_len)
    x = buf[:, :-1].to(device)
    y = buf[:, 1:].to(device)
    return x, y


# ---------------------------------------------------------------------------
# Training (only runs when executed directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    # Device selection: MPS > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    autocast_dtype = torch.float16
    autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=autocast_dtype)

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size:,}")

    config = build_model_config(DEPTH, vocab_size)
    print(f"Model config: {asdict(config)}")

    model = GPT(config)
    model.to(device)
    model.init_weights()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,} ({num_params / 1e6:.1f}M)")
    num_flops_per_token = model.estimate_flops()
    print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

    tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
    assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

    # AdamW optimizer (Muon's SVD ops may not work on MPS)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        weight_decay=WEIGHT_DECAY,
    )

    # torch.compile disabled: causes 3min compilation overhead on MPS
    # and recompilation during autoregressive generation (variable seq lengths)
    print("torch.compile: disabled (MPS overhead too high)")

    # Curriculum learning: short sequences first, ramp to full context
    # Phase 1: T=256 (learn VHDL keywords, types, basic syntax)
    # Phase 2: T=512 (learn entity/architecture structure)
    # Phase 3: T=1024 (learn full document structure)
    CURRICULUM = [(0.0, 256), (0.15, 512), (0.35, MAX_SEQ_LEN)]
    curr_T = CURRICULUM[0][1]
    train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, curr_T, "train")
    x, y, epoch = next(train_loader)

    pretrain_budget = TIME_BUDGET
    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Gradient accumulation steps: {grad_accum_steps}")
    print(f"Curriculum: {[(f'{p:.0%}', t) for p, t in CURRICULUM]}")
    print(f"Instruction mix ratio: {INSTRUCTION_MIX_RATIO:.0%}")

    # Build instruction pairs for chat training
    instruction_pairs = build_instruction_pairs()
    print(f"Instruction pairs: {len(instruction_pairs)}")

    # Schedules (cosine with warmup)
    import math as _math
    def get_lr_multiplier(progress):
        if progress < WARMUP_RATIO:
            return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
        else:
            decay_progress = (progress - WARMUP_RATIO) / (1.0 - WARMUP_RATIO)
            return FINAL_LR_FRAC + (1.0 - FINAL_LR_FRAC) * 0.5 * (1.0 + _math.cos(_math.pi * decay_progress))

    # ===================================================================
    # PHASE 1: Standard pretraining on VHDL corpus
    # ===================================================================

    print(f"\n{'='*60}")
    print(f"PHASE 1: Pretraining ({pretrain_budget}s)")
    print(f"{'='*60}")

    t_start_training = time.time()
    smooth_train_loss = 0
    total_training_time = 0
    step = 0
    progress = 0.0

    while True:
        t0 = time.time()

        # Curriculum: check if we need to switch to longer sequences
        new_T = CURRICULUM[0][1]
        for threshold, t_len in CURRICULUM:
            if progress >= threshold:
                new_T = t_len
        if new_T != curr_T:
            curr_T = new_T
            train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, curr_T, "train")
            x, y, epoch = next(train_loader)
            print(f"\n[Curriculum] Switching to T={curr_T}")

        for micro_step in range(grad_accum_steps):
            # Mix instruction data at INSTRUCTION_MIX_RATIO
            if random.random() < INSTRUCTION_MIX_RATIO:
                inst_x, inst_y = get_instruction_batch(
                    instruction_pairs, tokenizer, device,
                    batch_size=DEVICE_BATCH_SIZE, seq_len=curr_T)
                with autocast_ctx:
                    loss = model(inst_x, inst_y)
            else:
                with autocast_ctx:
                    loss = model(x, y)
                x, y, epoch = next(train_loader)
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()

        # Progress and schedules (relative to pretrain budget)
        progress = min(total_training_time / pretrain_budget, 1.0)
        lrm = get_lr_multiplier(progress)
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE * lrm
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.zero_grad(set_to_none=True)

        train_loss_f = train_loss.item()

        # Fast fail: abort if loss is exploding
        if train_loss_f > 100:
            print("FAIL")
            exit(1)

        t1 = time.time()
        dt = t1 - t0

        if step > 10:
            total_training_time += dt

        # Logging
        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
        pct_done = 100 * progress
        actual_tokens = DEVICE_BATCH_SIZE * curr_T * grad_accum_steps
        tok_per_sec = int(actual_tokens / dt)
        remaining = max(0, pretrain_budget - total_training_time)

        print(f"\r[P1] step {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | T={curr_T} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

        # GC management
        if step == 0:
            gc.collect()

        step += 1

        # Time's up for phase 1
        if step > 10 and total_training_time >= pretrain_budget:
            break

    print()
    pretrain_steps = step
    pretrain_tokens = step * TOTAL_BATCH_SIZE
    print(f"Phase 1 done: {pretrain_steps} steps, {pretrain_tokens/1e6:.1f}M tokens")

    total_tokens = pretrain_tokens
    feedback_round = 0
    total_generated = 0
    total_accepted = 0
    feedback_time = 0
    template_steps = 0

    # ===================================================================
    # Final evaluation
    # ===================================================================

    model.eval()
    with autocast_ctx:
        val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

    # Save checkpoint
    torch.save(model.state_dict(), "model.pt")

    # Compile-rate evaluation
    NUM_COMPILE_SAMPLES = 10
    compiled_ok = 0
    syntax_ok_count = 0
    print(f"\nCompile evaluation ({NUM_COMPILE_SAMPLES} samples)...")
    for i in range(NUM_COMPILE_SAMPLES):
        prompt = VHDL_PROMPTS[i % len(VHDL_PROMPTS)]
        with autocast_ctx:
            code = generate_vhdl(model, tokenizer, prompt, device,
                             max_new_tokens=256, temperature=0.4, top_k=30)
        s_ok, _, _ = compile_vhdl(code, mode="syntax")
        a_ok, _, _ = compile_vhdl(code, mode="analyze")
        if a_ok:
            compiled_ok += 1
        if s_ok:
            syntax_ok_count += 1
        status = "COMPILE" if a_ok else ("SYNTAX" if s_ok else "FAIL")
        print(f"  [{i+1}/{NUM_COMPILE_SAMPLES}] {status}")
    compile_rate = compiled_ok / NUM_COMPILE_SAMPLES
    syntax_rate = syntax_ok_count / NUM_COMPILE_SAMPLES

    # Final summary
    t_end = time.time()

    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"compile_rate:     {compile_rate:.4f}")
    print(f"syntax_rate:      {syntax_rate:.4f}")
    print(f"compiled:         {compiled_ok}/{NUM_COMPILE_SAMPLES}")
    print(f"syntax_ok:        {syntax_ok_count}/{NUM_COMPILE_SAMPLES}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"pretrain_seconds: {pretrain_budget:.0f}")
    print(f"feedback_seconds: {feedback_time:.1f}")
    print(f"feedback_rounds:  {feedback_round}")
    print(f"feedback_accept:  {total_accepted}/{total_generated}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {pretrain_steps + template_steps}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"depth:            {DEPTH}")
