"""
Generate VHDL code from a trained model and optionally compile with GHDL.

Usage:
    uv run generate.py                                    # generate with default prompt
    uv run generate.py --prompt "library ieee;"           # custom prompt
    uv run generate.py --chat "Design a counter"          # chat mode (English -> VHDL)
    uv run generate.py --n 10 --compile                   # generate 10 samples, compile each
    uv run generate.py --compile --save-dir ./generated   # save compilable VHDL files
"""

import argparse
import os

import torch

from prepare import Tokenizer
from train import GPT, build_model_config, generate_vhdl, compile_vhdl, DEPTH, VHDL_PROMPTS
from train import INST_TOKEN, RESP_TOKEN


def load_model(checkpoint_path="model.pt"):
    """Load trained model from checkpoint."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    config = build_model_config(DEPTH, vocab_size)
    model = GPT(config)

    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        model.init_weights()
        print("No checkpoint found — using random weights (train first!)")

    model.to(device)
    model.eval()
    return model, tokenizer, device


def main():
    parser = argparse.ArgumentParser(description="Generate VHDL from trained model")
    parser.add_argument("--prompt", type=str, default=None, help="VHDL prompt to complete")
    parser.add_argument("--chat", type=str, default=None, help="Chat mode: English description -> VHDL")
    parser.add_argument("--checkpoint", type=str, default="model.pt", help="Model checkpoint path")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--n", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--compile", action="store_true", help="Compile generated VHDL with GHDL")
    parser.add_argument("--save-dir", type=str, default=None, help="Save compilable VHDL files here")
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.checkpoint)

    if args.chat:
        # Chat mode: manually construct token IDs with proper special tokens
        inst_id = tokenizer.enc.encode_single_token(INST_TOKEN)
        resp_id = tokenizer.enc.encode_single_token(RESP_TOKEN)
        desc_ids = tokenizer.encode(args.chat)
        prompt_ids = [inst_id] + desc_ids + [resp_id]

        print(f"\nChat: {args.chat}")
        print('='*60)

        for i in range(args.n):
            code = generate_vhdl(model, tokenizer, prompt_ids, device,
                               max_new_tokens=args.max_tokens,
                               temperature=args.temperature, top_k=args.top_k)
            # Strip instruction tokens from output
            for tok in [INST_TOKEN, RESP_TOKEN]:
                code = code.replace(tok, "")
            print(code)

            if args.compile:
                success, error, _ = compile_vhdl(code)
                print(f"\n--- GHDL: {'PASS' if success else 'FAIL'} ---")
                if error:
                    print(error)
            if i < args.n - 1:
                print(f"\n{'='*60}")
    elif args.compile and args.n > 1:
        # Batch compile evaluation mode
        print(f"\nGenerating {args.n} samples and compiling...")
        compiled_ok = 0
        results = []
        for i in range(args.n):
            prompt = VHDL_PROMPTS[i % len(VHDL_PROMPTS)]
            code = generate_vhdl(model, tokenizer, prompt, device,
                               max_new_tokens=args.max_tokens, temperature=args.temperature)
            success, error, _ = compile_vhdl(code)
            if success:
                compiled_ok += 1
            results.append((prompt, code, success, error))
            status = "OK" if success else f"FAIL: {error[:80] if error else '?'}"
            print(f"  [{i+1}/{args.n}] {status}")

        rate = compiled_ok / args.n
        print(f"\ncompile_rate: {rate:.2%} ({compiled_ok}/{args.n})")

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            saved = 0
            for i, (prompt, code, success, _) in enumerate(results):
                if success:
                    path = os.path.join(args.save_dir, f"generated_{i:03d}.vhd")
                    with open(path, "w") as f:
                        f.write(code)
                    saved += 1
            print(f"Saved {saved} compilable files to {args.save_dir}")
    else:
        # Single generation mode
        for i in range(args.n):
            prompt = args.prompt or VHDL_PROMPTS[i % len(VHDL_PROMPTS)]
            print(f"\n{'='*60}")
            print(f"Prompt: {prompt[:80]}...")
            print('='*60)

            code = generate_vhdl(model, tokenizer, prompt, device,
                               max_new_tokens=args.max_tokens,
                               temperature=args.temperature, top_k=args.top_k)
            print(code)

            if args.compile:
                success, error, _ = compile_vhdl(code)
                print(f"\n--- GHDL: {'PASS' if success else 'FAIL'} ---")
                if error:
                    print(error)


if __name__ == "__main__":
    main()
