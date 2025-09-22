# Import necessary modules
import torch
import torch.nn as nn

# Import get_loaders function from data module within the same directory
from .data import get_loaders

import tqdm.auto as tqdm


# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(
    model,
    tokenizer,
    eval_dataset,
    eval_batch_size,
):
    """
    Evaluate the perplexity of a model on a dataset.

    Args:
        model: nn.Module, The model to evaluate
        tokenizer: PreTrainedTokenizer, The tokenizer to use
        eval_dataset: str, The dataset to evaluate on
        eval_batch_size: int, The batch size to use for evaluation

    Returns:
        float, The perplexity of the model on the dataset
    """
    # Set dataset
    dataset = eval_dataset

    # Print status
    print(f"Evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset,
        seed=0,
        seqlen=model.config.max_position_embeddings,
        tokenizer=tokenizer,
    )

    # Evaluate perplexity in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(
            model,
            testloader,
            eval_batch_size,
            model.device,
        )
    return ppl_test


@torch.no_grad()
def eval_ppl_wikitext(
    model,
    testenc,
    bs=1,
    device=None,
):
    """
    Evaluate the perplexity of a model on WikiText2.

    Args:
        model: nn.Module, The model to evaluate
        testenc: TokenizerWrapper, The tokenized test dataset
        bs: int, The batch size to use for evaluation
        device: str, The device to use for evaluation

    Returns:
        float, The perplexity of the model on the dataset
    """
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.config.max_position_embeddings

    # List to store negative log likelihoods
    nlls = []

    model.eval()
    with torch.no_grad():
        # Loop through each batch
        progress_bar = tqdm.tqdm(range(0, nsamples, bs))
        for i in progress_bar:

            # Calculate end index
            j = min(i + bs, nsamples)

            # Prepare inputs and move to device
            inputs = testenc[
                :,
                (i * model.config.max_position_embeddings) : (
                    j * model.config.max_position_embeddings
                ),
            ].to(device)
            inputs = inputs.reshape(j - i, model.config.max_position_embeddings)

            # Forward pass through the model
            lm_logits = model(inputs).logits

            # Shift logits and labels for next token prediction
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]

            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )

            # Calculate negative log likelihood
            neg_log_likelihood = (
                loss.float() * model.config.max_position_embeddings * (j - i)
            )

            # Append to list of negative log likelihoods
            nlls.append(neg_log_likelihood)

            progress_bar.set_description(
                f"Perplexity: {(torch.exp(torch.stack(nlls).sum() / (i * model.config.max_position_embeddings)).item()):.2f}"
            )

        # Compute perplexity
        ppl = torch.exp(
            torch.stack(nlls).sum() / (nsamples * model.config.max_position_embeddings)
        )

    return ppl.item()
