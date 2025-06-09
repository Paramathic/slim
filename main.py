import argparse
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from slim.prune import  prune_and_quantize
from slim.eval import eval_ppl
from slim.utils import report_gpu_memory, check_sparsity
from slim.lora import quantize_lora
from slim.quantization.quantization import attach_input_quantization_hooks
from utils.model import get_llm, distribute_model
from slim.fine_tune import fine_tune
import lm_eval


CSV_COLUMNS = ["model", "prune_method", "sparsity_ratio", "sparsity_type", "lora_rank",
               "slim_lora", "shift_zero_metrics", "prune_lora", "quantize_lora", "lora_tile_size", "eval_dataset",
               "quantize_weight", "bitwidth", "tiled_weight_quantization", "weight_tile_size", "quantize_input",
               "input_bitwidth", "input_group_size", "fine_tune", "optimizer", "slim_quant", "perplexity",
               "mmlu", "piqa", "arc_easy", "arc_challenge", "winogrande", "openbookqa", "average"]


def add_result_to_csv(args, ppl, lmharness_results):
    # Load CSV if it exists, otherwise create a new DataFrame with given columns
    directory = os.path.dirname(args.output_csv_path)
    if not os.path.exists(directory):
        os.mkdir(directory)
    if os.path.exists(args.output_csv_path):
        df = pd.read_csv(args.output_csv_path)
    else:
        df = pd.DataFrame(columns=CSV_COLUMNS)

    num_tasks = 8

    # Check if the row combination exists and update perplexity
    new_row_data = {column: getattr(args, column) for column in CSV_COLUMNS[:-num_tasks]}
    row_exists = df.index[(df[CSV_COLUMNS[:-num_tasks]] == pd.Series(new_row_data)).all(axis=1)]

    # Now we don't mind adding perplexity
    new_row_data['perplexity'] = ppl
    for task in lmharness_results:
        new_row_data[task] = lmharness_results[task]

    if row_exists.empty:
        # Row combination does not exist, add a new row
        new_row_df = pd.DataFrame([new_row_data], columns=CSV_COLUMNS)
        df = pd.concat([df, new_row_df], ignore_index=True)
    else:
        # Row combination exists, modify perplexity
        index_to_update = row_exists.values[0]
        df.at[index_to_update, 'perplexity'] = new_row_data['perplexity']
        for task in lmharness_results:
            df.at[index_to_update, task] = lmharness_results[task]

    # Save to CSV
    df.to_csv(args.output_csv_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str)
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt",
                                                             "ablate_wanda_seq",  "joint_pq", "maskllm"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str)

    parser.add_argument("--slim_lora", action="store_true")
    parser.add_argument("--lora_rank", type=float, default=0.0)
    parser.add_argument("--separate_lora", action="store_true")
    parser.add_argument("--prune_lora", action="store_true")
    parser.add_argument("--quantize_lora", action="store_true")
    parser.add_argument("--lora_tile_size", type=int, default=256)
    parser.add_argument("--pad_lora", action="store_true", help="Whether to pad LoRA to "
                        "lora_tile_size (without quantization)")

    parser.add_argument("--bitwidth", type=int, default=8)
    parser.add_argument("--quantize_weight", action="store_true")
    parser.add_argument("--tiled_weight_quantization", action="store_true")
    parser.add_argument("--weight_tile_size", type=int, default=256)
    parser.add_argument("--calibration_dataset", type=str, default="c4",
                        choices=["c4", "slimpajama"])
    parser.add_argument("--eval_dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "c4", "openwebtext", "slimpajama"])
    parser.add_argument("--shift_zero_metrics", action="store_true")
    parser.add_argument("--slim_quant", action="store_true")
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--output_csv_path", type=str, default=None,
                        help='Output CSV to accumulate experiment result')
    parser.add_argument('--test_lmharness', action="store_true", help="Whether to test LMEHarness tasks")
    parser.add_argument('--fine_tune', action="store_true",
                        help="Whether to fine-tune the model after pruning")
    parser.add_argument('--evaluate_perplexity', action="store_true",
                        help="Whether to evaluate the model perplexity")
    parser.add_argument('--local_files_only', action="store_true",
                        help="Whether to use local files only")
    parser.add_argument('--quantize_input', action="store_true", help="Whether to quantize input")
    parser.add_argument("--input_bitwidth", type=int, default=8, help="Input quantization bitwidth")
    parser.add_argument("--input_group_size", type=int, default=-1, help="Input quantization group size")
    parser.add_argument("--optimizer", type=str, default="adamw_torch",
                        help="Optimizer for fien-tuning models")
    parser.add_argument("--hf_token", type=str, default="")
    parser.add_argument("--joint_pq_mixing_factor", type=float, default=2.1)
    parser.add_argument("--scale_important_weights", action="store_true",)
    parser.add_argument("--maskllm_checkpoint", type=str, default=None,
                        help="Checkpoint for MaskLLM mask")


    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    model_name = args.model.split("/")[-1]
    print(f"Loading model {model_name}")
    model, lm_eval_model, orig_device_map = get_llm(
        model_name=args.model,
        local_files_only=args.local_files_only,
        hf_token=args.hf_token,
    )

    model = model.to(torch.bfloat16)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=False,
        token=args.hf_token,
    )

    report_gpu_memory("Before Pruning")

    prune_and_quantize(
        model,
        tokenizer,
        bitwidth=args.bitwidth,
        slim_quant=args.slim_quant,
        weight_tiled_quantization=args.tiled_weight_quantization,
        weight_tile_size=args.weight_tile_size,
        prune_method=args.prune_method,
        sparsity_ratio=args.sparsity_ratio,
        sparsity_type=args.sparsity_type,
        quantize_weight=args.quantize_weight,
        nsamples=args.nsamples,
        shift_zero_metrics=args.shift_zero_metrics,
        lora_rank=args.lora_rank,
        slim_lora=args.slim_lora,
        prune_lora=args.prune_lora,
        quantize_lora=args.quantize_lora,
        lora_tile_size=args.lora_tile_size,
        separate_lora=args.separate_lora,
        seed=args.seed,
        joint_pq_mixing_factor=args.joint_pq_mixing_factor,
        calibration_dataset=args.calibration_dataset,
        pad_lora=args.pad_lora,
        scale_important_weights=args.scale_important_weights,
        mask_checkpoint=args.maskllm_checkpoint,
    )
    report_gpu_memory("After pruning")

    model = distribute_model(model, device_map=orig_device_map)

    
    print("*" * 30)
    ################################################################
    if args.quantize_weight and args.quantize_lora and args.lora_rank > 0.:
        quantize_lora(
            model,
            args.bitwidth,
            args.lora_tile_size,
        )
    ################################################################
    if args.fine_tune:
        report_gpu_memory("Before Fine-tuning")
        fine_tune(model, tokenizer, optimizer=args.optimizer)
        report_gpu_memory("After Fine-tuning")
        print("*" * 30)
    ################################################################
    if args.quantize_input:
        print("Enabling input quantization:")
        attach_input_quantization_hooks(model,
                                        args.input_bitwidth,
                                        args.input_group_size,
                                        )
    ################################################################
    ppl_test = 0.
    if args.evaluate_perplexity:
        ppl_test = eval_ppl(
            model,
            tokenizer,
            args.eval_dataset,
            args.eval_batch_size,
        )
        print(f"Perplexity: {ppl_test:.2f}")
        print("*" * 30)
    ################################################################
    sparsity_ratio = check_sparsity(model)
    print(f"Model Sparsity Ratio: {sparsity_ratio:.2f}")
    print("*" * 30)
    ################################################################
    
    lmharness_results = {}
    if args.test_lmharness:
        results = lm_eval.simple_evaluate(
            model=lm_eval_model,
            tasks=["mmlu", "piqa", "arc_easy", "arc_challenge", "winogrande", "openbookqa"],
            verbosity="ERROR"
        )
        lmharness_results["mmlu"] = results['results']["mmlu"]["acc,none"]
        lmharness_results["piqa"] = results['results']["piqa"]["acc,none"]
        lmharness_results["arc_easy"] = results['results']["arc_easy"]["acc,none"]
        lmharness_results["arc_challenge"] = results['results']["arc_challenge"]["acc,none"]
        lmharness_results["winogrande"] = results['results']["winogrande"]["acc,none"]
        lmharness_results["openbookqa"] = results['results']["openbookqa"]["acc,none"]
        average = []
        for task in lmharness_results:
            average.append(lmharness_results[task])
        average = np.mean(average)
        lmharness_results["average"] = average
        print("LM Harness Results: ", lmharness_results)


    if args.output_csv_path:
        add_result_to_csv(args, ppl_test, lmharness_results)


if __name__ == '__main__':
    main()
