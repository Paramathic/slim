import torch
import torch.nn as nn
from .sparsegpt import SparseGPT
from .sparsegpt import Quantizer as SparseGPTQuantizer
from .layerwrapper import WrappedGPT
from .data import get_loaders
from .utils import get_layers_list, shift_zeros, find_layers, prune_nm
from .lora import add_lora
from slim.quantization.quantization import Quantizer as AutoQuantizer, QuantizedMatmul
import tqdm.auto as tqdm
from .jsq_utils import clip_matrix, generate_ss
from .smooth import smooth_layer
from huggingface_hub import hf_hub_download
import numpy as np


def prepare_calibration_input(model, dataloader, nsamples=128):
    """
    Prepare inputs for calibration.

    Args:
        model: torch.nn.Module - The model to calibrate
        dataloader: torch.utils.data.DataLoader - The dataloader to use for calibration

    Returns:
        inps: torch.Tensor - The input tensor for calibration
        outs: torch.Tensor - The output tensor for calibration
        attention_mask: torch.Tensor - The attention mask for calibration
    """
    use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False
    layers = get_layers_list(model)

    dtype = next(iter(model.parameters())).dtype
    torch.cuda.empty_cache()
    inps = torch.zeros(
        (nsamples, model.config.max_position_embeddings, model.config.hidden_size),
        dtype=dtype,
        device="cpu",
    )
    input_device = "cpu"
    inps.requires_grad = False
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp.to(input_device)
            cache["i"] += 1
            for key in kwargs:
                cache[key] = kwargs[key]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0])
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    model.config.use_cache = use_cache
    del cache["i"]
    return inps, outs, cache


def prune_magnitude(
    model,
    sparsity_ratio,
    prune_n=0,
    prune_m=0,
    quantize_weight=False,
    bitwidth=4,
    slim_quant=False,
    tiled_weight_quantization=False,
    weight_tile_size=256,
    column_wise_grouping=False,
):
    """
    Prune a model using magnitude pruning and quantize weights using SLiM-Quant or AbsMax.

    Args:
        model: torch.nn.Module - The model to prune
        sparsity_ratio: float - The ratio of weights to prune
        prune_n: int - The number N in N:M pruning
        prune_m: int - The number M in N:M pruning
        quantize_weight: bool - Whether to quantize weights
        bitwidth: int - The bitwidth to use for quantization
        slim_quant: bool - Whether to use SLiM-Quant
        tiled_weight_quantization: bool - Whether to use block quantization
        weight_tile_size: int - The size of the blocks for block quantization

    Returns:
        None
    """
    layers = get_layers_list(model)
    progress_bar = tqdm.tqdm(range(len(layers)))

    if quantize_weight:
        quantizer = AutoQuantizer(
            "weight",
            num_bits=bitwidth,
            slim_quant=slim_quant,
            block_quantization=tiled_weight_quantization,
            block_dim=weight_tile_size,
            column_wise_grouping=column_wise_grouping,
        )
    else:
        quantizer = None

    for i in progress_bar:
        progress_bar.set_description(f"Layer {i}")
        layer = layers[i].cuda()
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = prune_nm(W_metric, prune_n, prune_m)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][
                    int(W.numel() * sparsity_ratio)
                ].cpu()
                W_mask = W_metric <= thresh

            W[W_mask] = 0
            subset[name].weight.data = W
            if quantizer is not None:
                quantized_weight = quantizer.quantize_weight(subset[name].weight.data)
                subset[name].weight.data = quantizer.dequantize_absmax(
                    quantized_weight
                ).to(torch.bfloat16)
                subset[name].register_buffer(
                    "quantization_scaling_factor", quantizer.scaling_factor
                )
        layer = layer.cpu()


def prune_wanda(
    model,
    tokenizer,
    sparsity_ratio=0.5,
    prune_n=0,
    prune_m=0,
    quantize_weight=False,
    bitwidth=4,
    slim_quant=False,
    tiled_weight_quantization=False,
    weight_tile_size=256,
    shift_zero_metrics=True,
    lora_rank=0.0,
    slim_lora=True,
    prune_lora=False,
    quantize_lora=False,
    lora_tile_size=256,
    separate_lora=True,
    nsamples=128,
    seed=0,
    calibration_dataset="c4",
    pad_lora=False,
    quantize_first=True,
    scale_important_weights=False,
    column_wise_grouping=False,
):
    """
    Prune a model using WANDA and quantize weights using SLiM-Quant or AbsMax and add low-rank adapter using SLiM or SVD.

    Args:
        model: torch.nn.Module - The model to prune
        tokenizer: transformers.Tokenizer - The tokenizer for the model
        sparsity_ratio: float - The ratio of weights to prune
        prune_n: int - The number N in N:M pruning
        prune_m: int - The number M in N:M pruning
        quantize_weight: bool - Whether to quantize weights
        bitwidth: int - The bitwidth to use for quantization
        slim_quant: bool - Whether to use SLiM-Quant
        tiled_weight_quantization: bool -
        weight_tile_size: int - The size of the blocks for block quantization
        shift_zero_metrics: bool - Whether to shift zero metrics
        lora_rank: float - The rank ratio for low-rank adapter
        slim_lora: bool - Whether to use SLiM for low-rank adapter
        prune_lora: bool - Whether to prune the low-rank adapter
        quantize_lora: bool - Whether to quantize the low-rank adapter
        lora_tile_size: int - The size of the blocks for block quantization of the low-rank adapter
        separate_lora: bool - Whether to separate the low-rank adapter
        nsamples: int - The number of samples to use for calibration
        seed: int - The seed to use for calibration
        calibration_dataset: str - The dataset to use for calibration

    Returns:
        None
    """
    use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False

    dataloader, _ = get_loaders(
        calibration_dataset,
        nsamples=nsamples,
        seed=seed,
        seqlen=model.config.max_position_embeddings,
        tokenizer=tokenizer,
    )
    with torch.no_grad():
        inps, outs, kwargs = prepare_calibration_input(model, dataloader, nsamples)

    if quantize_weight:
        quantizer = AutoQuantizer(
            "weight",
            num_bits=bitwidth,
            slim_quant=slim_quant,
            block_quantization=tiled_weight_quantization,
            block_dim=weight_tile_size,
            column_wise_grouping=column_wise_grouping,
        )
    else:
        quantizer = None

    layers = get_layers_list(model)

    progress_bar = tqdm.tqdm(range(len(layers)))

    for i in progress_bar:
        progress_bar.set_description(f"Layer {i} - Gathering data")
        layer = layers[i].cuda()

        subset = find_layers(layer)
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(nsamples):
            with torch.no_grad():
                for key in kwargs:
                    if isinstance(kwargs[key], torch.Tensor):
                        kwargs[key] = kwargs[key].cuda()
                    if isinstance(kwargs[key], tuple):
                        kwargs[key] = tuple([k.cuda() for k in kwargs[key]])
                outs[j] = layer(inps[j].unsqueeze(0).cuda(), **kwargs)[0].to(
                    outs[j].device
                )

        for h in handles:
            h.remove()

        for name in subset:
            progress_bar.set_description(f"Layer {i} - Pruning and Quantizing {name}")
            if shift_zero_metrics:
                wrapped_layers[name].scaler_row = shift_zeros(
                    wrapped_layers[name].scaler_row
                )

            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )

            W_mask = (
                torch.zeros_like(W_metric) == 1
            )  ## initialize a mask to be all False
            if prune_n != 0:
                if hasattr(subset[name].weight, "mask"):
                    W_mask = subset[name].weight.mask
                    del subset[name].weight.mask
                else:
                    W_mask = prune_nm(W_metric, prune_n, prune_m)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                # unstructured pruning
                indices = sort_res[1][:, : int(W_metric.shape[1] * sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

            if lora_rank > 0.0:
                lora_tile_size = lora_tile_size if (quantize_lora or pad_lora) else None
                add_lora(
                    subset[name],
                    W_mask=W_mask,
                    rank_ratio=lora_rank,
                    slim_lora=slim_lora,
                    activations=wrapped_layers[name],
                    quantizer=quantizer,
                    prune_lora=prune_lora,
                    separate_lora=separate_lora,
                    lora_tile_size=lora_tile_size,
                    quantize_first=quantize_first,
                    scale_important_weights=scale_important_weights,
                )

                if quantizer is not None:
                    subset[name].register_buffer(
                        "quantization_scaling_factor", quantizer.scaling_factor
                    )

                if separate_lora:

                    def add_lora_hook(module, input, output):
                        if hasattr(module, "lora_quantizer"):
                            xl = QuantizedMatmul.apply(
                                input[0].to(module.lora_left.dtype)
                                / torch.sqrt(module.lora_rank),
                                module.lora_left,
                                module.lora_quantizer,
                            )
                            xlr = QuantizedMatmul.apply(
                                xl / torch.sqrt(module.lora_rank),
                                module.lora_right,
                                module.lora_quantizer,
                            )
                            output += xlr

                        else:
                            output += torch.matmul(
                                torch.matmul(
                                    input[0].to(module.lora_left.dtype),
                                    module.lora_left / torch.sqrt(module.lora_rank),
                                ),
                                module.lora_right / torch.sqrt(module.lora_rank),
                            )

                    subset[name].lora_rank = torch.tensor(
                        subset[name].lora_left.shape[1]
                    )
                    subset[name].lora_left = torch.nn.Parameter(
                        subset[name].lora_left * torch.sqrt(subset[name].lora_rank)
                    )
                    subset[name].lora_right = torch.nn.Parameter(
                        subset[name].lora_right * torch.sqrt(subset[name].lora_rank)
                    )
                    subset[name].register_forward_hook(add_lora_hook)

            else:
                if scale_important_weights:
                    # Get 1% of largest activations
                    metric = subset[name].scaler_row * subset[
                        name
                    ].weight.data.abs().sum(dim=0)
                    important_weights = metric.topk(
                        int(0.01 * metric.numel()), largest=True, sorted=False
                    )[1]
                else:
                    important_weights = None
                if quantize_first:
                    if quantizer is not None:
                        quantized_weight = quantizer.quantize_weight(
                            subset[name].weight.data, important_weights
                        )
                        subset[name].weight.data = quantizer.dequantize_absmax(
                            quantized_weight
                        ).to(torch.bfloat16)
                        if quantizer is not None:
                            subset[name].register_buffer(
                                "quantization_scaling_factor", quantizer.scaling_factor
                            )
                    subset[name].weight.data[W_mask] = 0  ## set weights to zero
                else:
                    subset[name].weight.data[W_mask] = 0  ## set weights to zero
                    if quantizer is not None:
                        quantized_weight = quantizer.quantize_weight(
                            subset[name].weight.data, important_weights
                        )
                        subset[name].weight.data = quantizer.dequantize_absmax(
                            quantized_weight
                        ).to(torch.bfloat16)
                        if quantizer is not None:
                            subset[name].register_buffer(
                                "quantization_scaling_factor", quantizer.scaling_factor
                            )

        progress_bar.set_description(f"Layer {i} - Evaluating Output")

        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0).cuda(), **kwargs)[0].to(
                    outs[j].device
                )
        inps, outs = outs, inps

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(
    model,
    tokenizer,
    sparsity_ratio=0.5,
    prune_n=0,
    prune_m=0,
    nsamples=128,
    seed=0,
    quantize_weight=False,
    bitwidth=4,
    tiled_weight_quantization=False,
    weight_tile_size=256,
    calibration_dataset="c4",
):
    """
    Prune a model using SparseGPT and quantize weights using OPTQ (GPTQ).
    SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa

    Args:
        model: torch.nn.Module - The model to prune
        tokenizer: transformers.Tokenizer - The tokenizer for the model
        device: torch.device - The device to use for pruning
        sparsity_ratio: float - The ratio of weights to prune
        prune_n: int - The number N in N:M pruning
        prune_m: int - The number M in N:M pruning
        nsamples: int - The number of samples to use for calibration
        seed: int - The seed to use for calibration
        quantize_weight: bool - Whether to quantize weights
        bitwidth: int - The bitwidth to use for quantization
        tiled_weight_quantization: bool - Whether to use block quantization
        weight_tile_size: int - The size of the blocks for block
        calibration_dataset: str - The dataset to use for calibration

    Returns:
        None
    """
    use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False

    dataloader, _ = get_loaders(
        calibration_dataset,
        nsamples=nsamples,
        seed=seed,
        seqlen=model.config.max_position_embeddings,
        tokenizer=tokenizer,
    )

    with torch.no_grad():
        inps, outs, kwargs = prepare_calibration_input(model, dataloader, nsamples)

    layers = get_layers_list(model)

    progress_bar = tqdm.tqdm(range(len(layers)))

    for i in progress_bar:
        progress_bar.set_description(f"Layer {i} - Gathering data")
        layer = layers[i].cuda()

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])
            if quantize_weight:
                gpts[name].quantizer = SparseGPTQuantizer()
                gpts[name].quantizer.configure(
                    bitwidth,
                    perchannel=tiled_weight_quantization,
                    sym=True,
                    mse=False,
                )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(nsamples):
            for key in kwargs:
                if isinstance(kwargs[key], torch.Tensor):
                    kwargs[key] = kwargs[key].cuda()
                if isinstance(kwargs[key], tuple):
                    kwargs[key] = tuple([k.cuda() for k in kwargs[key]])

            outs[j] = layer(inps[j].unsqueeze(0).cuda(), **kwargs)[0].to(outs.device)

        for h in handles:
            h.remove()

        for name in gpts:
            progress_bar.set_description(f"Layer {i} - Pruning and Quantizing {name}")
            gpts[name].fasterprune(
                sparsity_ratio,
                prune_n=prune_n,
                prune_m=prune_m,
                percdamp=0.01,
                blocksize=weight_tile_size,
            )
            if quantize_weight:
                subset[name].register_buffer(
                    "quantization_scaling_factor",
                    1.0 / gpts[name].quantizer.scaling_factor,
                )
            gpts[name].free()

        progress_bar.set_description(f"Layer {i} - Evaluating Output")
        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0).cuda(), **kwargs)[0].to(
                    outs[j].device
                )

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def quantize_model(
    model,
    bitwidth=4,
    slim_quant=False,
    weight_tiled_quantization=False,
    weight_tile_size=256,
    column_wise_grouping=False,
):
    """
    Quantize the model using the AutoQuantizer class.

    Args:
        model: torch.nn.Module - The model to quantize
        bitwidth: int - The bitwidth to quantize the model to
        slim_quant: bool - Use SLiM-Quant
        weight_tiled_quantization: bool - Use block quantization
        weight_tile_size: int - The size of the block for block quantization

    Returns:
        None
    """
    quantizer = AutoQuantizer(
        "weight",
        num_bits=bitwidth,
        slim_quant=slim_quant,
        block_quantization=weight_tiled_quantization,
        block_dim=weight_tile_size,
        column_wise_grouping=column_wise_grouping,
    )
    layers = get_layers_list(model)

    progress_bar = tqdm.tqdm(range(len(layers)))

    for i in progress_bar:
        layer = layers[i]
        layer = layer.cuda()

        subset = find_layers(layer)

        for name in subset:
            progress_bar.set_description(f"Layer {i} - Quantizing {name}")

            quantized_weight = quantizer.dequantize_absmax(
                quantizer.quantize_weight(subset[name].weight.data)
            )

            subset[name].weight.data = quantized_weight.to(subset[name].weight.dtype)
            subset[name].register_buffer(
                "quantization_scaling_factor", quantizer.scaling_factor
            )
        layer = layer.cpu()


def joint_pq(
    model,
    tokenizer,
    prune_n=0,
    prune_m=0,
    nsamples=128,
    bitwidth=4,
    sparsity_ratio=0.5,
    weight_tile_size=256,
    mixing_factor=2.1,
    seed=0,
    calibration_dataset="c4",
    lora_rank=0.0,
    slim_lora=True,
    prune_lora=False,
    quantize_lora=False,
    lora_tile_size=256,
    separate_lora=True,
    quantize_first=True,
    pad_lora=False,
    scale_important_weights=False,
    column_wise_grouping=False,
):
    """
    Prune and quantize a model using joint pruning and quantization.

    Args:
        model: torch.nn.Module - The model to prune and quantize
        tokenizer: transformers.Tokenizer - The tokenizer for the model
        prune_n: int - The number N in N:M pruning
        prune_m: int - The number M in N:M pruning
        nsamples: int - The number of samples to use for calibration
        bitwidth: int - The bitwidth to quantize the model to
        sparsity_ratio: float - The ratio of weights to prune
        weight_tile_size: int - The size of the blocks for block quantization
        mixing_factor: float - The mixing factor for WANDA
        seed: int - The seed to use for calibration
        calibration_dataset: str - The dataset to use for calibration
        lora_rank: float - The rank ratio for low-rank adapter
        slim_lora: bool - Whether to use SLiM for low-rank adapter
        prune_lora: bool - Whether to prune the low-rank adapter
        quantize_lora: bool - Whether to quantize the low-rank adapter
        lora_tile_size: int - The size of the blocks for block quantization of the low-rank adapter
        separate_lora: bool - Whether to separate the low-rank adapter
        quantize_first: bool - Whether to quantize the weights before or after pruning
        pad_lora: bool - Whether to pad the LoRA weights
        scale_important_weights: bool - Whether to scale the important weights

    Returns:
        None
    """

    dataloader, _ = get_loaders(
        calibration_dataset,
        nsamples=nsamples,
        seed=seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )
    with torch.no_grad():
        inps, outs, kwargs = prepare_calibration_input(model, dataloader, nsamples)

    layers = get_layers_list(model)

    quantizer = AutoQuantizer(
        "weight",
        num_bits=bitwidth,
        slim_quant=False,
        block_quantization=True,
        block_dim=weight_tile_size,
        column_wise_grouping=column_wise_grouping,
    )

    progress_bar = tqdm.tqdm(range(len(layers)))

    for i in progress_bar:
        layer = layers[i].cuda()
        layer_name = f"model.layers.{i}"

        subset = find_layers(layer)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        progress_bar.set_description(f"Layer {i} - Gathering data")
        act_scales = {}

        def stat_tensor(name, tensor):
            hidden_dim = tensor.shape[-1]
            tensor = tensor.view(-1, hidden_dim).abs().detach()
            comming_max = torch.max(tensor, dim=0)[0].float().cpu()

            if name in act_scales:
                act_scales[layer_name + "." + name] = torch.max(
                    act_scales[name], comming_max
                )
            else:
                act_scales[layer_name + "." + name] = comming_max

        def add_batch(name):
            def tmp(_, inp, out):
                inp = clip_matrix(inp[0].data, True, 0, 1e-2)
                stat_tensor(name, inp)
                wrapped_layers[name].add_batch(inp, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(nsamples):
            with torch.no_grad():
                for key in kwargs:
                    if isinstance(kwargs[key], torch.Tensor):
                        kwargs[key] = kwargs[key].cuda()
                    if isinstance(kwargs[key], tuple):
                        kwargs[key] = tuple([k.cuda() for k in kwargs[key]])
                outs[j] = layer(inps[j].unsqueeze(0).cuda(), **kwargs)[0].to(
                    outs.device
                )
        for h in handles:
            h.remove()

        for name in subset:
            progress_bar.set_description(f"Layer {i} - Pruning {name}")
            weight = torch.abs(subset[name].weight.data)
            activation = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))

            ss = generate_ss(
                wrapped_layers[name].inp_sum / wrapped_layers[name].inp_num,
                subset[name].weight.data,
            )
            W_metric = weight * activation
            W_metric = W_metric + mixing_factor * ss

            W_mask = torch.zeros_like(W_metric) == 1
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                # unstructured pruning
                indices = sort_res[1][:, : int(W_metric.shape[1] * sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(nsamples):
            with torch.no_grad():
                for key in kwargs:
                    if isinstance(kwargs[key], torch.Tensor):
                        kwargs[key] = kwargs[key].cuda()
                    if isinstance(kwargs[key], tuple):
                        kwargs[key] = tuple([k.cuda() for k in kwargs[key]])
                outs[j] = layer(inps[j].unsqueeze(0).cuda(), **kwargs)[0].to(
                    outs.device
                )

        progress_bar.set_description(f"Layer {i} - Smoothing {name}")
        smooth_layer(layer_name, layer, act_scales, 0.5)

        for name in subset:
            if lora_rank > 0.0:
                progress_bar.set_description(
                    f"Layer {i} - Quantizing and Adding LoRA to {name}"
                )
                lora_tile_size = lora_tile_size if (quantize_lora or pad_lora) else None
                add_lora(
                    subset[name],
                    W_mask=subset[name].weight.data == 0,
                    rank_ratio=lora_rank,
                    slim_lora=slim_lora,
                    activations=wrapped_layers[name],
                    quantizer=quantizer,
                    prune_lora=prune_lora,
                    separate_lora=separate_lora,
                    lora_tile_size=lora_tile_size,
                    quantize_first=quantize_first,
                    scale_important_weights=scale_important_weights,
                )

                if quantizer is not None:
                    subset[name].register_buffer(
                        "quantization_scaling_factor", quantizer.scaling_factor
                    )

                if separate_lora:

                    def add_lora_hook(module, input, output):
                        if hasattr(module, "lora_quantizer"):
                            xl = QuantizedMatmul.apply(
                                input[0].to(module.lora_left.dtype)
                                / torch.sqrt(module.lora_rank),
                                module.lora_left,
                                module.lora_quantizer,
                            )
                            xlr = QuantizedMatmul.apply(
                                xl / torch.sqrt(module.lora_rank),
                                module.lora_right,
                                module.lora_quantizer,
                            )
                            output += xlr

                        else:
                            output += torch.matmul(
                                torch.matmul(
                                    input[0].to(module.lora_left.dtype),
                                    module.lora_left / torch.sqrt(module.lora_rank),
                                ),
                                module.lora_right / torch.sqrt(module.lora_rank),
                            )

                    subset[name].lora_rank = torch.tensor(
                        subset[name].lora_left.shape[1]
                    )
                    subset[name].lora_left = torch.nn.Parameter(
                        subset[name].lora_left * torch.sqrt(subset[name].lora_rank)
                    )
                    subset[name].lora_right = torch.nn.Parameter(
                        subset[name].lora_right * torch.sqrt(subset[name].lora_rank)
                    )
                    subset[name].register_forward_hook(add_lora_hook)
            else:
                if scale_important_weights:
                    # Get 1% of largest activations
                    metric = subset[name].scaler_row * subset[
                        name
                    ].weight.data.abs().sum(dim=0)
                    important_weights = metric.topk(
                        int(0.01 * metric.numel()), largest=True, sorted=False
                    )[1]
                else:
                    important_weights = None
                progress_bar.set_description(f"Layer {i} - Quantizing {name}")
                quantized_weight = quantizer.quantize_weight(
                    subset[name].weight.data, important_weights
                )
                subset[name].weight.data = quantizer.dequantize_absmax(
                    quantized_weight
                ).to(torch.bfloat16)
                subset[name].register_buffer(
                    "quantization_scaling_factor", quantizer.scaling_factor
                )

        inps, outs = outs, inps
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()


def prune_and_quantize(
    model,
    tokenizer,
    bitwidth=4,
    slim_quant=True,
    weight_tiled_quantization=False,
    weight_tile_size=256,
    prune_method="wanda",
    sparsity_ratio=0.5,
    sparsity_type="2:4",
    quantize_weight=False,
    nsamples=128,
    shift_zero_metrics=True,
    lora_rank=0.0,
    slim_lora=True,
    prune_lora=False,
    quantize_lora=False,
    lora_tile_size=256,
    separate_lora=True,
    seed=0,
    joint_pq_mixing_factor=2.1,
    calibration_dataset="c4",
    pad_lora=False,
    scale_important_weights=False,
    mask_checkpoint=None,
    column_wise_grouping=False,
):
    """
    Prune and quantize a model and add low-rank adapter to it.

    Args:
        model: torch.nn.Module - The model to prune and quantize
        tokenizer: transformers.Tokenizer - The tokenizer for the model
        bitwidth: int - The bitwidth to quantize the model to
        slim_quant: bool - Use SLiM-Quant
        weight_tiled_quantization: bool - Use block quantization
        weight_tile_size: int - The size of the block for block quantization
        prune_method: str - The pruning method to use, one of "wanda", "magnitude", "sparsegpt", "joint_pq"
        sparsity_ratio: float - The ratio of weights to prune
        sparsity_type: str - The type of sparsity to use (unstructured, dense, N:M)
        quantize_weight: bool - Whether to quantize weights
        nsamples: int - The number of samples to use for calibration
        shift_zero_metrics: bool - Whether to shift zero metrics in Wanda
        lora_rank: float - The rank ratio for low-rank adapter
        slim_lora: bool - Whether to use SLiM for low-rank adapter
        prune_lora: bool - Whether to 2:4 prune the L low-rank adapter
        quantize_lora: bool - Whether to quantize the low-rank adapter
        lora_tile_size: int - The size of the block for block quantization of the low-rank adapter
        separate_lora: bool - Whether to separate the low-rank adapter
        seed: int - The seed to use for calibration
        joint_pq_mixing_factor: float - The mixing factor for joint pruning and quantization
        calibration_dataset: str - The dataset to use for calibration
        pad_lora: bool - Whether to pad the low-rank adapter to the quantization tile size (whithout quantizing)
        scale_important_weights: bool - Whether to scale the important weights before quantization,
        mask_checkpoint: str - The checkpoint to use for MaskLLM pruning
        column_wise_grouping: bool - Whether to use column-wise grouping for quantization

    Returns:
        None
    """
    if sparsity_ratio == 0.0 or sparsity_type == "dense":
        if quantize_weight:
            print("Quantizing the dense model:")
            if lora_rank > 0:
                raise NotImplementedError(
                    "LoRA approximation not implemented for quantization only - "
                    "Please use pruning with low sparsity ratio for quantization only."
                )
            quantize_model(
                model,
                bitwidth,
                slim_quant,
                weight_tiled_quantization,
                weight_tile_size,
                column_wise_grouping=column_wise_grouping,
            )
        else:
            print("Using original dense model.")
    else:
        print("Sparsity Ratio: ", sparsity_ratio)
        print("Pruning Structure: ", sparsity_type)
        # Handling n:m sparsity
        prune_n, prune_m = 0, 0
        if sparsity_type != "unstructured":
            prune_n, prune_m = map(int, sparsity_type.split(":"))
            prune_n = prune_m - prune_n
            assert (
                sparsity_ratio == prune_n / prune_m
            ), f"Sparsity ratio must be {prune_n / prune_m} for structured N:M sparsity"
        if prune_method in ["wanda", "maskllm"]:
            if prune_method == "wanda":
                pruning_name = "Wanda"
            else:
                pruning_name = "MaskLLM"
            if quantize_weight:
                if slim_quant:
                    quantization_method = "SLiM-Quant"
                else:
                    if weight_tiled_quantization:
                        quantization_method = "Tiled Group AbsMax"
                    else:
                        quantization_method = "AbsMax"
                print(
                    f"Pruning the model with {pruning_name} and quantizing the weights using {quantization_method}."
                )
            else:
                print(f"Pruning the model with {pruning_name}.")
            if lora_rank > 0:
                if slim_lora:
                    print(
                        f"Adding SLiM-LoRA approximation with rank ratio {lora_rank}."
                    )
                else:
                    print(
                        f"Adding Naive-LoRA approximation with rank ratio {lora_rank}."
                    )
            if prune_lora and not (prune_n == 2 and prune_m == 4):
                raise NotImplementedError(
                    "Pruning LoRA is only supported for 2:4 sparsity ratio"
                )
            if prune_method == "maskllm":
                assert (
                    mask_checkpoint is not None
                ), "Mask checkpoint must be provided for MaskLLM pruning"
                assert (
                    prune_n == 2 and prune_m == 4
                ), "MaskLLM pruning only supports 2:4 sparsity ratio"
                try:
                    if mask_checkpoint.endswith(".pt"):
                        with torch.no_grad():
                            checkpoint = torch.load(mask_checkpoint, map_location="cpu")
                            for k, v in checkpoint.items():
                                if v.dim() == 2:
                                    mask = v == 0
                                    param = dict(model.named_parameters()).get(k, None)
                                    if param is None:
                                        print(f"Parameter {k} not found in the model.")
                                    else:
                                        param.mask = mask
                    else:
                        downloaded_mask = hf_hub_download(
                            repo_id=mask_checkpoint, filename="mask_compressed.npz"
                        )
                        mask_ckpt = np.load(downloaded_mask)
                        for k, v in mask_ckpt.items():
                            k_original = k.replace(".mask", "")
                            v = np.unpackbits(v)  # to bits
                            mask = torch.from_numpy(v).float()
                            param = dict(model.named_parameters()).get(k_original, None)
                            mask = mask.view(*param.shape)
                            param.mask = (mask == 0).bool()
                except FileNotFoundError:
                    raise FileNotFoundError(
                        "Mask checkpoint not found. Please provide a valid checkpoint."
                    )
            prune_wanda(
                model,
                tokenizer,
                sparsity_ratio,
                prune_n,
                prune_m,
                quantize_weight,
                bitwidth,
                slim_quant,
                weight_tiled_quantization,
                weight_tile_size,
                shift_zero_metrics,
                lora_rank,
                slim_lora,
                prune_lora,
                quantize_lora,
                lora_tile_size,
                separate_lora,
                nsamples,
                seed,
                calibration_dataset,
                pad_lora,
                scale_important_weights=scale_important_weights,
                column_wise_grouping=column_wise_grouping,
            )
        elif prune_method == "magnitude":
            if scale_important_weights and quantize_weight:
                raise NotImplementedError(
                    "Scaling important weights not implemented for magnitude pruning and "
                    "quantization"
                )
            if lora_rank > 0:
                raise NotImplementedError(
                    "LoRA approximation not implemented for magnitude pruning"
                )
            if quantize_weight:
                if slim_quant:
                    quantization_method = "SLiM-Quant"
                else:
                    if weight_tiled_quantization:
                        quantization_method = "Tiled Group AbsMax"
                    else:
                        quantization_method = "AbsMax"
                print(
                    f"Pruning the model with Magnitude Pruning "
                    f"and quantizing the weights using {quantization_method}."
                )
            else:
                print("Pruning the model with Magnitude Pruning.")
            prune_magnitude(
                model,
                sparsity_ratio,
                prune_n,
                prune_m,
                quantize_weight,
                bitwidth,
                slim_quant,
                weight_tiled_quantization,
                weight_tile_size,
                column_wise_grouping=column_wise_grouping,
            )
        elif prune_method == "sparsegpt":
            if scale_important_weights and quantize_weight:
                raise NotImplementedError(
                    "Scaling important weights not implemented for magnitude pruning and "
                    "quantization"
                )
            if lora_rank > 0:
                raise NotImplementedError(
                    "LoRA approximation not implemented for SparseGPT"
                )
            if slim_quant:
                raise NotImplementedError(
                    "SparseGPT can only support OPTQ (GPTQ) quantization"
                )
            if quantize_weight:
                if weight_tiled_quantization:
                    quantization_method = "Group OPTQ (GPTQ)"
                else:
                    quantization_method = "OPTQ (GPTQ)"
                print(
                    f"Pruning the model with SparseGPT and quantizing the weights using {quantization_method}."
                )
            else:
                print("Pruning the model with SparseGPT.")
            if column_wise_grouping:
                raise NotImplementedError(
                    "Column-wise grouping not implemented for OPTQ"
                )
            prune_sparsegpt(
                model,
                tokenizer,
                sparsity_ratio,
                prune_n,
                prune_m,
                nsamples,
                seed,
                quantize_weight,
                bitwidth,
                weight_tiled_quantization,
                weight_tile_size,
                calibration_dataset,
            )
        elif prune_method == "joint_pq":
            if weight_tiled_quantization is False:
                raise NotImplementedError(
                    "Joint pruning and quantization only supports block quantization"
                )
            if slim_quant:
                raise NotImplementedError(
                    "Joint pruning and quantization only supports AbsMax"
                )
            if quantize_weight is False:
                raise NotImplementedError(
                    "Joint pruning and quantization requires quantizing weights"
                )
            joint_pq(
                model,
                tokenizer,
                prune_n,
                prune_m,
                nsamples,
                bitwidth,
                sparsity_ratio,
                weight_tile_size,
                joint_pq_mixing_factor,
                seed,
                calibration_dataset,
                lora_rank,
                slim_lora,
                prune_lora,
                quantize_lora,
                lora_tile_size,
                separate_lora,
                pad_lora,
                scale_important_weights,
                column_wise_grouping=column_wise_grouping,
            )
        else:
            raise NotImplementedError(f"Pruning method {prune_method} not implemented")
