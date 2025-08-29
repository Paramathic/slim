import os
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import torch
from datasets import load_dataset, load_from_disk
import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from transformers.testing_utils import CaptureLogger
from transformers.utils.versions import require_version


def disable_linear_layer_grads(model):
    known_modules = {"Linear", "Conv1d"}
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if module_type in known_modules:
            if hasattr(module, "lora_left"):
                module.weight.requires_grad = False
                module.weight.data = module.weight.half()
                if module.bias is not None:
                    module.bias.data = module.bias.half()
                    module.bias.requires_grad = False
                if hasattr(module, "lora_left_mask"):
                    def mask_lora(self, inputs):
                        self.lora_left.data[self.lora_left_mask] = 0

                    module.register_forward_pre_hook(mask_lora)
            else:
                def mask_weight(self, inputs):
                    self.weight.data[self.weight_mask] = 0
                module.weight.requires_grad = True
                module.weight_mask = (module.weight.data.clone().detach().cpu() == 0)
                module.register_forward_pre_hook(mask_weight)
                if hasattr(module, "scaling_factor"):
                    if module.scaling_factor is None:
                        raise NotImplementedError("Group quantization is not supported for fine-tuning")


def requantize(model, q=4):
    known_modules = {"Linear", "Conv1d"}
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if module_type in known_modules:
            if hasattr(module, "lora_left"):
                pass
            else:
                if not hasattr(module, 'scaling_factor'):
                    pass
                else:
                    if module.scaling_factor is None:
                        raise NotImplementedError("Group quantization is not supported for fine-tuning")
                    scaling_factor = module.scaling_factor
                    weight = module.weight
                    quantized_weight = torch.round((weight * scaling_factor).float())
                    module.weight.data = (quantized_weight / scaling_factor).to(module.weight.dtype)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def fine_tune(
        model,
        tokenizer,
        dataset_name="c4",
        dataset_config_name=None,
        validation_split_percentage=5,
        streaming=False,
        preprocessing_num_workers=None,
        overwrite_cache=False,
        block_size=None,
        max_train_samples=30000,
        max_eval_samples=128,
        cache_dir="data",
        optimizer="adamw_torch",
        global_batch_size=64,
        local_batch_size=1,
        use_wandb=False,
):
    """
    Fine-tune a model on a dataset using HuggingFace Trainer. In case of existence of LoRA, the original weights are
    frozen ant the LoRA is only updated. Otherwise, the weights are updated and requantized in the end if needed.

    Args:
        model: torch.nn.Module - The model to fine-tune
        tokenizer: transformers.PreTrainedTokenizer - The tokenizer to use
        dataset_name: str - The name of the dataset to use (via the datasets library).
        dataset_config_name: str - The configuration name of the dataset to use (via the datasets library).
        validation_split_percentage: int - The percentage of the train set used as validation set in case there's no
            validation split
        streaming: bool - Enable streaming mode
        preprocessing_num_workers: int - The number of processes to use for the preprocessing.
        overwrite_cache: bool - Overwrite the cached training and evaluation sets
        block_size: int - Optional input sequence length after tokenization.
        max_train_samples: int - For debugging purposes or quicker training, truncate the number of training examples to
            this value if set.
        max_eval_samples: int - For debugging purposes or quicker training, truncate the number of evaluation examples to
            this value if set.
        cache_dir: str - The directory to store the cache
        optimizer: str - The optimizer to use
        global_batch_size: int - The global batch size for training
        local_batch_size: int - The local batch size to run on the device
        use_wandb: bool - Whether to enable Weights & Biases logging

    Returns:
        None
    """
    bf16 = transformers.utils.import_utils.is_torch_bf16_gpu_available()
    if not bf16:
        model = model.float()
    else:
        model = model.to(torch.bfloat16)
    
    # Set report_to based on wandb usage
    report_to = "wandb" if use_wandb else "none"
    
    training_args = TrainingArguments(
        output_dir="output",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=local_batch_size,
        per_device_eval_batch_size=local_batch_size,
        num_train_epochs=1,
        logging_dir="logs",
        logging_steps=100,
        eval_steps=100,
        save_steps=5000,
        save_total_limit=1,
        bf16=bf16,
        fp16=not bf16,
        group_by_length=False,
        gradient_accumulation_steps=global_batch_size // local_batch_size,
        warmup_steps=5,
        optim=optimizer,
        save_strategy="steps",
        report_to=report_to,
        gradient_checkpointing=True,
        learning_rate=1e-4,
    )
    ################################################################################################################
    if os.path.exists(f"{cache_dir}/c4-raw.pt"):
        raw_datasets = load_from_disk(f"{cache_dir}/c4-raw.pt")
    else:
        try:
            raw_datasets = load_dataset('allenai/c4',
                                        'allenai--c4',
                                        data_files={'train': 'en/c4-train.00000-of-01024.json.gz',
                                                    'validation': 'en/c4-validation.00000-of-00008.json.gz'},
                                        cache_dir=cache_dir)
        except:
            raw_datasets = load_dataset('allenai/c4',
                                        data_files={'train': 'en/c4-train.00000-of-01024.json.gz',
                                                    'validation': 'en/c4-validation.00000-of-00008.json.gz'},
                                        cache_dir=cache_dir)

        raw_datasets.save_to_disk(f"{cache_dir}/c4-raw.pt")

    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            dataset_name,
            dataset_config_name,
            split=f"train[:{validation_split_percentage}%]",
            streaming=streaming,
        )
        raw_datasets["train"] = load_dataset(
            dataset_name,
            dataset_config_name,
            split=f"train[{validation_split_percentage}%:]",
            streaming=streaming,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    # Preprocessing the datasets.
    # First we tokenize all the texts.


    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")


    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output


    with training_args.main_process_first(desc="Dataset map tokenization"):
        if not streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )

    if block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            print(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if block_size > tokenizer.model_max_length:
            print(
                f"The block_size passed ({block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that global_batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with training_args.main_process_first(desc="Grouping texts together"):
        if not streaming:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=preprocessing_num_workers,
                load_from_cache_file=not overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if max_train_samples is not None:
            max_train_samples = min(len(train_dataset), max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = None

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    disable_linear_layer_grads(model)

    ################################################################################################################
    model.config.use_cache = False
    if training_args.do_train:
        model.train()
        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it.
            data_collator=default_data_collator,
            compute_metrics=compute_metrics if training_args.do_eval else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
            if training_args.do_eval
            else None,
        )
        train_result = trainer.train()
        metrics = train_result.metrics

        max_train_samples = (
            max_train_samples if max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        
        # Log fine-tuning metrics to wandb if enabled
        if use_wandb:
            import wandb
            # Log training metrics with "finetune_" prefix to distinguish from main metrics
            wandb_metrics = {}
            for key, value in metrics.items():
                wandb_metrics[f"finetune_{key}"] = value
            wandb.log(wandb_metrics)

    requantize(model)