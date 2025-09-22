from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
from datasets import load_dataset
import os
import time
import math
import pandas as pd
from model_compression import compress_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument(
        "--lora_rank", type=float, help="Rank for LoRA compression", default=0.1
    )
    parser.add_argument("--quantize_lora", action="store_true", help="Quantize LoRA")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Parsed arguments: {args}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    )

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    dense_memory = torch.cuda.memory_allocated()

    model = compress_model(
        model,
        skip_layers=[model.lm_head],
        lora_rank=args.lora_rank,
        quantize_lora=args.quantize_lora,
    )

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    compressed_memory = torch.cuda.memory_allocated()

    print(
        f"Memory results for model: {args.model} with LoRA rank {args.lora_rank} and quantize_lora={args.quantize_lora}"
    )
    print(f"Dense model memory: {dense_memory / (1024 ** 3):.2f} GB")
    print(f"Compressed model memory: {compressed_memory / (1024 ** 3):.2f} GB")
    print(f"Memory reduction: {(compressed_memory / dense_memory):.2f}x")
