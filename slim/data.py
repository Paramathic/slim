# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset, load_from_disk
import os
import tqdm.auto as tqdm


# Set seed for reproducibility
def set_seed(seed):
    """
    Set seed for reproducibility

    Args:
        seed: int, The seed to set

    Returns:
        None
    """
    np.random.seed(seed)
    torch.random.manual_seed(seed)


# Wrapper for tokenized input IDs
class TokenizerWrapper:
    """
    Wrapper for tokenized input IDs
    """

    def __init__(self, input_ids):
        self.input_ids = input_ids


# Load and process wikitext2 dataset
def get_wikitext2(seed, tokenizer):
    """
    Load and process WikiText2 Test dataset

    Args:
        seed: int, The seed to set
        tokenizer: PreTrainedTokenizer, The tokenizer to use
        cache_dir: str, The directory to cache the dataset

    Returns:
        trainloader: list, The list of training samples
        testenc: TokenizerWrapper, The tokenized test dataset
    """
    print("Loading WikiText2 dataset.")
    # Load train and test datasets
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # Encode datasets
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    return trainloader, testenc


# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    """
    Load and process C4 dataset

    Args:
        nsamples: int, The number of samples to generate
        seed: int, The seed to set
        seqlen: int, The sequence length
        tokenizer: PreTrainedTokenizer, The tokenizer to use
        cache_dir: str, The directory to cache the dataset

    Returns:
        trainloader: list, The list of training samples
        valenc: TokenizerWrapper, The tokenized validation dataset
    """
    print("Loading C4 dataset.")
    # Load train and validation datasets
    if os.path.exists(f"data/c4-train.pt"):
        traindata = load_from_disk(f"data/c4-train.pt")
        valdata = load_from_disk(f"data/c4-val.pt")
    else:
        try:
            traindata = load_dataset(
                "allenai/c4",
                "allenai--c4",
                data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
                split="train",
            )
            valdata = load_dataset(
                "allenai/c4",
                "allenai--c4",
                data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
                split="validation",
            )
        except:
            traindata = load_dataset(
                "allenai/c4",
                data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
                split="train",
            )
            valdata = load_dataset(
                "allenai/c4",
                data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
                split="validation",
            )

        traindata.save_to_disk(f"data/c4-train.pt")
        valdata.save_to_disk(f"data/c4-val.pt")

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    progress_bar = tqdm.tqdm(range(nsamples))
    for _ in progress_bar:
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        progress_bar.set_description("Generating Samples")

    # Prepare validation dataset
    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc


def get_openwebtext(seed, seqlen, tokenizer):
    """
    Load and process OpenWebText dataset

    Args:
        seed: int, The seed to set
        seqlen: int, The sequence length
        tokenizer: PreTrainedTokenizer, The tokenizer to use
        cache_dir: str, The directory to cache the dataset

    Returns:
        trainloader: None
        valenc: TokenizerWrapper, The tokenized validation dataset
    """
    # Load train and validation datasets
    print("Loading OpenWebText dataset.")
    raw_datasets = load_dataset("openwebtext")
    raw_datasets = raw_datasets["train"].train_test_split(
        test_size=0.05,
        seed=seed,
        shuffle=True,  # Otherwise test will be at the end of the dataset
    )
    trainloader = None
    valdata = raw_datasets["test"]
    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc


def get_slimpajama(nsamples, seed, seqlen, tokenizer):
    """
    Load and process SlimPajama dataset

    Args:
        seed: int, The seed to set
        tokenizer: PreTrainedTokenizer, The tokenizer to use
        cache_dir: str, The directory to cache the dataset

    Returns:
        trainloader: list, The list of training samples
        testenc: TokenizerWrapper, The tokenized test dataset
    """
    print("Loading SlimPajama dataset.")
    # Load train and test datasets
    traindata = load_dataset("DKYoon/SlimPajama-6B", split="train")
    testdata = load_dataset("DKYoon/SlimPajama-6B", split="test")

    # Encode datasets
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    progress_bar = tqdm.tqdm(range(nsamples))
    for _ in progress_bar:
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        progress_bar.set_description("Generating Samples")

    return trainloader, testenc


# Function to select the appropriate loader based on dataset name
def get_loaders(
    name,
    nsamples=128,
    seed=0,
    seqlen=2048,
    tokenizer=None,
):
    """
    Get loaders for the specified dataset

    Args:
        name: str, The name of the dataset
        nsamples: int, The number of samples to generate
        seed: int, The seed to set
        seqlen: int, The sequence length
        tokenizer: PreTrainedTokenizer, The tokenizer to use
        cache_dir: str, The directory to cache the dataset

    Returns:
        trainloader: list, The list of training samples
        testenc: TokenizerWrapper, The tokenized test dataset
    """
    if "wikitext2" in name.lower():
        return get_wikitext2(seed, tokenizer)
    elif "c4" in name.lower():
        return get_c4(nsamples, seed, seqlen, tokenizer)
    elif "openwebtext" in name.lower():
        return get_openwebtext(seed, seqlen, tokenizer)
    elif "slimpajama" in name.lower():
        return get_slimpajama(nsamples, seed, seqlen, tokenizer)
    else:
        raise ValueError(f"Unknown dataset {name}")


if __name__ == "__main__":
    from transformers import AutoTokenizer
    import lm_eval

    try:
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained=facebook/opt-125m,dtype=half,device=cpu",
            tasks=[
                "mmlu",
                "piqa",
                "arc_easy",
                "arc_challenge",
                "winogrande",
                "openbookqa",
            ],
            verbosity="ERROR",
        )
    except:
        pass

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    for name in ["wikitext2", "c4", "openwebtext", "slimpajama"]:
        try:
            trainloader, testenc = get_loaders(
                name, nsamples=128, seqlen=1024, tokenizer=tokenizer
            )
        except:
            pass
