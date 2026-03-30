#!/usr/bin/env python3
"""CLI: Quantize a HuggingFace model to 2-bit (or n-bit) TurboQuant format.

Usage:
    python quantize.py --model Qwen/Qwen2.5-0.5B-Instruct --bits 2 --output ./quantized/qwen-0.5b-2bit
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from turboquant.model import replace_linear_layers, save_quantized_model


def main():
    parser = argparse.ArgumentParser(description="Quantize a model with TurboQuant")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID or path")
    parser.add_argument("--bits", type=int, default=2, choices=[2, 4], help="Bit width (default: 2)")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--group-size", type=int, default=None, help="Group size (default: full row)")
    parser.add_argument("--seed", type=int, default=42, help="Rotation seed")
    parser.add_argument("--rotation", type=str, default="hadamard", choices=["hadamard", "qr"])
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--quantize-embeddings", action="store_true", default=False,
                        help="Also quantize embedding layers (saves memory, may reduce quality)")
    parser.add_argument("--embedding-bits", type=int, default=None, choices=[2, 4],
                        help="Bit width for embeddings (default: same as --bits)")
    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    load_dtype = dtype_map[args.dtype]

    print(f"Loading {args.model} in {args.dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=load_dtype, device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Quantizing to {args.bits}-bit with {args.rotation} rotation...")
    replace_linear_layers(
        model,
        bit_width=args.bits,
        group_size=args.group_size,
        seed=args.seed,
        rotation=args.rotation,
        quantize_embeddings=args.quantize_embeddings,
        embedding_bit_width=args.embedding_bits,
    )

    print(f"Saving to {args.output}...")
    save_quantized_model(model, args.output, tokenizer=tokenizer)
    print("Done!")


if __name__ == "__main__":
    main()
