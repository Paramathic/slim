import argparse
import os
import numpy as np
import pandas as pd
import torch
import wandb
from transformers import AutoTokenizer
from slim.prune import prune_and_quantize
from slim.eval import eval_ppl
from slim.utils import report_gpu_memory, check_sparsity
from slim.lora import quantize_lora
from slim.quantization.quantization import attach_input_quantization_hooks
from utils.model import get_llm, distribute_model
from slim.fine_tune import fine_tune
from slim.save_model import save_model
import lm_eval
import torch.distributed as dist
from slim.load_model_with_acceleration import load_compressed_model


CSV_COLUMNS = [
    "model",
    "prune_method",
    "sparsity_ratio",
    "sparsity_type",
    "lora_rank",
    "slim_lora",
    "shift_zero_metrics",
    "prune_lora",
    "quantize_lora",
    "lora_tile_size",
    "eval_dataset",
    "quantize_weight",
    "bitwidth",
    "tiled_weight_quantization",
    "weight_tile_size",
    "quantize_input",
    "input_bitwidth",
    "input_group_size",
    "fine_tune",
    "learning_rate",
    "finetune_token_count",
    "fine_tuning_global_batch_size",
    "weight_decay",
    "optimizer",
    "slim_quant",
    "perplexity",
    "average",
]


def add_result_to_csv(args, ppl, lmharness_results):
    # Load CSV if it exists, otherwise create a new DataFrame with given columns
    directory = os.path.dirname(args.output_csv_path)
    if not os.path.exists(directory):
        os.mkdir(directory)

    all_columns = CSV_COLUMNS[:-1] + args.lm_harness_tasks + CSV_COLUMNS[-1:]

    if os.path.exists(args.output_csv_path):
        df = pd.read_csv(args.output_csv_path)
    else:
        df = pd.DataFrame(columns=all_columns)

    # Check if the row combination exists and update perplexity
    new_row_data = {column: getattr(args, column) for column in CSV_COLUMNS[:-2]}
    row_exists = df.index[(df[CSV_COLUMNS[:-2]] == pd.Series(new_row_data)).all(axis=1)]

    # Now we don't mind adding perplexity
    new_row_data["perplexity"] = ppl
    for task in lmharness_results:
        new_row_data[task] = lmharness_results[task]

    if row_exists.empty:
        # Row combination does not exist, add a new row
        new_row_df = pd.DataFrame([new_row_data], columns=all_columns)
        df = pd.concat([df, new_row_df], ignore_index=True)
    else:
        # Row combination exists, modify perplexity
        index_to_update = row_exists.values[0]
        df.at[index_to_update, "perplexity"] = new_row_data["perplexity"]
        for task in lmharness_results:
            df.at[index_to_update, task] = lmharness_results[task]

    # Save to CSV
    df.to_csv(args.output_csv_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="LLaMA model")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration samples."
    )
    parser.add_argument(
        "--sparsity_ratio", type=float, default=0, help="Sparsity level"
    )
    parser.add_argument("--sparsity_type", type=str)
    parser.add_argument(
        "--prune_method",
        type=str,
        choices=[
            "magnitude",
            "wanda",
            "sparsegpt",
            "ablate_wanda_seq",
            "joint_pq",
            "maskllm",
        ],
    )
    parser.add_argument("--cache_dir", default="llm_weights", type=str)

    parser.add_argument("--slim_lora", action="store_true")
    parser.add_argument("--lora_rank", type=float, default=0.0)
    parser.add_argument("--separate_lora", action="store_true")
    parser.add_argument("--prune_lora", action="store_true")
    parser.add_argument("--quantize_lora", action="store_true")
    parser.add_argument("--lora_tile_size", type=int, default=256)
    parser.add_argument(
        "--pad_lora",
        action="store_true",
        help="Whether to pad LoRA to " "lora_tile_size (without quantization)",
    )

    parser.add_argument("--bitwidth", type=int, default=8)
    parser.add_argument("--quantize_weight", action="store_true")
    parser.add_argument("--tiled_weight_quantization", action="store_true")
    parser.add_argument("--weight_tile_size", type=int, default=128)
    parser.add_argument(
        "--calibration_dataset", type=str, default="c4", choices=["c4", "slimpajama"]
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "openwebtext", "slimpajama"],
    )
    parser.add_argument("--shift_zero_metrics", action="store_true")
    parser.add_argument("--slim_quant", action="store_true")
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument(
        "--output_csv_path",
        type=str,
        default=None,
        help="Output CSV to accumulate experiment result",
    )
    parser.add_argument(
        "--test_lmharness", action="store_true", help="Whether to test LMEHarness tasks"
    )
    parser.add_argument(
        "--lm_harness_tasks",
        type=str,
        nargs="+",
        default=[
            "mmlu",
            "piqa",
            "arc_easy",
            "arc_challenge",
            "winogrande",
            "openbookqa",
        ],
        help="LM Harness tasks to evaluate",
    )
    parser.add_argument(
        "--fine_tune",
        action="store_true",
        help="Whether to fine-tune the model after pruning",
    )
    parser.add_argument(
        "--evaluate_perplexity",
        action="store_true",
        help="Whether to evaluate the model perplexity",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Whether to use local files only",
    )
    parser.add_argument(
        "--quantize_input", action="store_true", help="Whether to quantize input"
    )
    parser.add_argument(
        "--input_bitwidth", type=int, default=8, help="Input quantization bitwidth"
    )
    parser.add_argument(
        "--input_group_size", type=int, default=-1, help="Input quantization group size"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw_torch",
        help="Optimizer for fien-tuning models",
    )
    parser.add_argument("--hf_token", type=str, default="")

    parser.add_argument("--joint_pq_mixing_factor", type=float, default=2.1)
    parser.add_argument(
        "--scale_important_weights",
        action="store_true",
    )
    parser.add_argument(
        "--maskllm_checkpoint",
        type=str,
        default=None,
        help="Checkpoint for MaskLLM mask",
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="SLiM", help="W&B project name"
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None, help="W&B run name (optional)"
    )
    parser.add_argument(
        "--save_checkpoint_path",
        type=str,
        default=None,
        help="Directory to save the model checkpoint",
    )
    parser.add_argument(
        "--column_wise_grouping",
        action="store_true",
        default=False,
        help="Whether to use column-wise grouping for quantization",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for fine-tuning",
    )
    parser.add_argument(
        "--finetune_token_count",
        type=int,
        default=300_000,
        help="Number of tokens to use for fine-tuning",
    )
    parser.add_argument(
        "--fine_tuning_global_batch_size",
        type=int,
        default=128,
        help="Global batch size for fine-tuning",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-2, help="Weight decay for fine-tuning"
    )
    parser.add_argument(
        "--fine_tuning_seqlen",
        type=int,
        default=4096,
        help="Sequence length for fine-tuning",
    )

    args = parser.parse_args()

    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # Trainer uses this
    if is_distributed:
        import datetime

        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))

    # Initialize wandb if enabled
    if args.use_wandb and rank == 0:
        # Create run name if not provided
        run_name = args.wandb_run_name
        if run_name is None:
            model_name = args.model.split("/")[-1]
            run_name = args.save_checkpoint_path.split("/")[-1]
            # run_name = f"{model_name}_{args.prune_method}_{args.sparsity_ratio}_lora{args.lora_rank}_slimlora{args.slim_lora}_quantlora{args.quantize_lora}_quantweight{args.quantize_weight}_slimquant{args.slim_quant}_finetune{args.fine_tune}"

        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    model_name = args.model.split("/")[-1]
    if rank == 0:
        print(f"Loading model {model_name}")
        model, lm_eval_model = get_llm(
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
            column_wise_grouping=args.column_wise_grouping,
        )
        report_gpu_memory("After pruning")
        save_model(model, args.save_checkpoint_path, args)

    if is_distributed:
        dist.barrier()  # Ensure all processes wait until pruning is done

        model, tokenizer, cfg, lora_hooks = load_compressed_model(
            args.save_checkpoint_path, device_map={"": local_rank}
        )
        model.config.max_position_embeddings = 2048
        model.eval()
        model.cuda(local_rank)
    else:
        model = distribute_model(model)

    print("*" * 30)
    ################################################################
    if args.quantize_weight and args.quantize_lora and args.lora_rank > 0.0:
        quantize_lora(
            model,
            args.bitwidth,
            args.lora_tile_size,
            column_wise_grouping=args.column_wise_grouping,
        )
    ################################################################
    if args.fine_tune:
        if rank == 0:
            report_gpu_memory("Before Fine-tuning")
        fine_tune(
            model,
            tokenizer,
            optimizer=args.optimizer,
            use_wandb=args.use_wandb,
            learning_rate=args.learning_rate,
            max_train_samples=args.finetune_token_count,
            global_batch_size=args.fine_tuning_global_batch_size,
            weight_decay=args.weight_decay,
            block_size=args.fine_tuning_seqlen,
        )
        if rank == 0:
            report_gpu_memory("After Fine-tuning")
            print("*" * 30)
    ################################################################

    if rank == 0:
        if args.quantize_input:
            print("Enabling input quantization:")
            attach_input_quantization_hooks(
                model,
                args.input_bitwidth,
                args.input_group_size,
            )
        ################################################################

        report_gpu_memory("After Fine-tuning")
        ppl_test = 0.0
        if args.evaluate_perplexity:
            ppl_test = eval_ppl(
                model,
                tokenizer,
                args.eval_dataset,
                args.eval_batch_size,
            )
            print(f"Perplexity: {ppl_test:.2f}")

            # Log perplexity to wandb
            if args.use_wandb:
                wandb.log({"perplexity": ppl_test})

            print("*" * 30)
        ################################################################
        sparsity_ratio = check_sparsity(model)
        print(f"Model Sparsity Ratio: {sparsity_ratio:.2f}")

        # Log sparsity ratio to wandb
        if args.use_wandb:
            wandb.log({"sparsity_ratio": sparsity_ratio})

        print("*" * 30)
        ################################################################

    lmharness_results = {}
    model = model.cpu()
    torch.cuda.empty_cache()
    lm_eval_model._model.load_state_dict(model.state_dict(), strict=True)
    if is_distributed:
        lm_eval_model._model = lm_eval_model._model.to(local_rank)
    else:
        lm_eval_model._model = distribute_model(lm_eval_model._model)

    if args.test_lmharness:
        results = lm_eval.simple_evaluate(
            model=lm_eval_model,
            tasks=args.lm_harness_tasks,
            verbosity="ERROR",
        )
        if local_rank == 0:
            for task in args.lm_harness_tasks:
                if task in results["results"]:
                    lmharness_results[task] = results["results"][task]["acc,none"]
            average = []
            for task in lmharness_results:
                average.append(lmharness_results[task])
            if average:
                average = np.mean(average)
                lmharness_results["average"] = average
            print("LM Harness Results: ", lmharness_results)
    if rank == 0:
        # Log LM Harness results to wandb
        if args.use_wandb:
            wandb.log(lmharness_results)

        if args.output_csv_path:
            add_result_to_csv(args, ppl_test, lmharness_results)

        if args.save_checkpoint_path is not None:
            save_model(model, args.save_checkpoint_path, args)


if __name__ == "__main__":
    main()
