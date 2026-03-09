# autoresearch — VHDL on Apple Silicon (Scaled)

This is an experiment to have the LLM do its own research, optimizing a GPT model for VHDL code generation on Apple Silicon (MPS). The model is ~10M params, trained for 30 minutes per experiment.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar9`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `collect_vhdl.py` — VHDL corpus collection from GitHub (multiple sources). Run once to build data.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains VHDL parquet shards and a tokenizer. If not, tell the human to run:
   ```bash
   python collect_vhdl.py    # build VHDL corpus (~1000 repos)
   python prepare.py         # train BPE tokenizer on VHDL
   ```
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Hardware context (M1 Max, 32GB unified memory)

- **No Flash Attention**: Use `F.scaled_dot_product_attention` (PyTorch native, MPS-compatible).
- **No bfloat16**: MPS bfloat16 support is spotty. Use `float16` for autocast.
- **No torch.cuda anything**: No CUDA synchronize, no CUDA memory tracking, no CUDA streams.
- **No Muon optimizer**: Muon's SVD/polar decomposition may not work on MPS. Use AdamW.
- **torch.compile**: May or may not work on MPS. The code tries it and falls back gracefully.
- **Model size**: ~10M params (DEPTH=8, ASPECT_RATIO=48, HEAD_DIM=64). Can push to ~20M if memory allows.
- **Batch size**: TOTAL_BATCH_SIZE=2^17, DEVICE_BATCH_SIZE=32. Can increase if no OOM.
- **32GB unified memory**: Shared between CPU and GPU. Monitor memory — don't OOM the system.

## Two-Phase Training (built into train.py)

Each training run automatically executes two phases:

### Phase 1: Pretraining (75% of time budget = ~22.5 min)
Standard language modeling on the VHDL corpus. The model learns VHDL syntax, structure, and patterns.

### Phase 2: Compiler Feedback (25% of time budget = ~7.5 min)
Rejection sampling fine-tuning loop:
1. Generate 32 VHDL samples from the model
2. Score each with GHDL compiler (progressive: structural check → syntax → full analysis)
3. Keep samples that score well, weighted by quality (compilable code gets 3x weight)
4. Fine-tune the model on accepted samples
5. Also train on corpus data to prevent catastrophic forgetting
6. Repeat until time runs out

This teaches the model to produce VHDL that actually compiles, not just text that looks like VHDL.

### What the agent can experiment with:
- PRETRAIN_RATIO (how much time for pretraining vs feedback)
- FEEDBACK_LR (learning rate for fine-tuning phase)
- GENERATE_BATCH (samples per feedback round)
- GENERATE_TEMPERATURE (diversity of generated samples)
- Scoring weights (how much to reward syntax vs full compilation)
- Model architecture, optimizer, hyperparameters (everything in Phase 1)
- MLP ratio (4x vs 3x vs 8/3x for SwiGLU)
- Activation functions, attention patterns, etc.

## Experimentation

Each experiment runs on MPS (Apple Silicon GPU). The training script runs for a **fixed time budget of 30 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is two metrics: lowest val_bpb AND highest compile_rate.** After training, the script generates 20 VHDL samples and compiles each with GHDL. The compile_rate (0.0-1.0) measures how often the model produces valid VHDL. Prioritize val_bpb as the primary metric, but compile_rate is the secondary metric that shows real-world usefulness. A model with great val_bpb but 0% compile rate is useless. Since the time budget is fixed, you don't need to worry about training time — it's always 30 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**Memory** is a soft constraint. With 32GB unified memory, you can push the model larger than defaults, but don't OOM the system. If you OOM, back off model size.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## VHDL-specific experiment ideas

Beyond standard LLM tricks, consider VHDL-specific optimizations:
- **Keyword-aware attention**: VHDL has a small set of keywords (entity, architecture, process, signal, port, etc.). Consider attention patterns that leverage this structure.
- **Hierarchical structure**: VHDL code has a natural hierarchy (library → entity → architecture → process). Models that capture this may compress better.
- **Tokenizer efficiency**: The BPE tokenizer is trained on VHDL. Check if common VHDL patterns (`:=`, `<=`, `std_logic_vector`, `rising_edge`) are single tokens.
- **Positional patterns**: VHDL indentation is semantic. Consider whether positional encodings interact well with VHDL structure.
- **Repetitive structure**: VHDL port maps and signal declarations are highly repetitive. Consider if relative position encodings help.
- **Long-range dependencies**: VHDL architectures reference entities declared far away. Consider if increasing context or adding memory helps.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          1.234567
compile_rate:     0.3500
syntax_rate:      0.6000
compiled:         7/20
syntax_ok:        12/20
training_seconds: 1800.1
pretrain_seconds: 1350
feedback_seconds: 450.1
feedback_rounds:  8
feedback_accept:  45/256
total_seconds:    1825.9
total_tokens_M:   50.3
num_steps:        750
num_params_M:     10.5
depth:            8
```

You can extract the key metrics from the log file:

```
grep "^val_bpb:\|^compile_rate:\|^syntax_rate:\|^feedback_accept:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	compile_rate	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. compile_rate (0.00-1.00) — fraction of generated VHDL that compiles with GHDL
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	compile_rate	status	description
a1b2c3d	1.234567	0.05	keep	baseline (~10M params depth=8)
b2c3d4e	1.180000	0.15	keep	increase depth to 10
c3d4e5f	1.200000	0.10	discard	switch to GeLU activation
d4e5f6g	0.000000	0.00	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar9`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~30 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 40 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.
