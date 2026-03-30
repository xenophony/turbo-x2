#!/usr/bin/env python3
"""CLI: Run inference on a TurboQuant-quantized model.

Usage:
    python inference.py --model ./quantized/qwen-0.5b-2bit --prompt "The capital of France is"
"""

import argparse
import torch
from transformers import AutoTokenizer

from turboquant.model import load_quantized_model, decompress_model


def main():
    parser = argparse.ArgumentParser(description="Run inference with a TurboQuant model")
    parser.add_argument("--model", type=str, required=True, help="Path to quantized model directory")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--device", type=str, default=None, help="Device (default: auto)")
    parser.add_argument("--decompress", action="store_true", default=False,
                        help="Decompress to bf16 at load time for native inference speed")
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading quantized model from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_quantized_model(args.model, device=device)
    if args.decompress:
        decompress_model(model)
    model.eval()

    print(f"Generating (device={device})...")
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.temperature > 0,
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n{'='*60}")
    print(response)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
