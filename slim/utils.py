import torch
import numpy as np

def check_sparsity(model):
    """
    Check the end-to-end sparsity ratio of a model.
    model: torch.nn.Module - The model to check
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = get_layers_list(model)
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()

        print(f"Layer {i} Sparsity Ratio: {float(sub_count) / sub_params:.2f}")

    model.config.use_cache = use_cache
    return float(count) / total_params


def report_gpu_memory(message=""):
    """
    Report the allocated memory on the GPU in GB.
    message: str, a message to print before the memory report
    """
    torch.cuda.empty_cache()
    print(message, f" - Allocated Memory: {(torch.cuda.memory_allocated() / 1024 / 1024 / 1024):.2f}GB")


def get_layers_list(model):
    """
    Get the decoder layers of the model.
    model: nn.Module, the model to extract the layers from
    """
    if hasattr(model, "model"):
        if hasattr(model.model, "layers"):
            layers = model.model.layers
        else:
            if hasattr(model.model, "decoder"):
                layers = model.model.decoder.layers
            else:
                raise NotImplementedError
    elif hasattr(model, "transformer"):
        layers = model.transformer.h
    else:
        raise NotImplementedError
    return layers


def shift_zeros(x):
    """
    Shift zeros to the smallest positive value in the tensor.
    x: torch.Tensor, the input tensor
    """
    min_positive = x.clone().detach()
    min_positive[min_positive == 0] = 1
    min_positive = min_positive.min()
    return x + min_positive


def prune_nm(mat, n, m):
    """
    Prune the matrix using N:M sparsity.
    mat: torch.Tensor, the input matrix
    n: int, N in N:M sparsity
    m: int, M in N:M sparsity

    """
    mask = (torch.zeros_like(mat) == 1)
    for ii in range(mat.shape[1]):
        if ii % m == 0:
            tmp = mat[:, ii:(ii + m)]
            topk = torch.topk(tmp, n, dim=1, largest=False)[1]
            mask.scatter_(1, ii + topk, True)
    return mask


def remove_outlier(x, std_factor=2):
    """
    Remove outliers from a list.
    x: list, the input list
    std_factor: float, the standard deviation factor to consider an element as an outlier
    """
    mean = np.mean(x)
    std = np.std(x)
    return [e for e in x if (mean - std_factor * std < e < mean + std_factor * std)]


def find_layers(module, layers=[torch.nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res