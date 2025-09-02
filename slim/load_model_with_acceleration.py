import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import safetensors.torch
import torch
from slim.quantization.quantization import Quantizer as AutoQuantizer
from utils.model import add_empty_lora, distribute_model
import argparse
import time

# Import speedup components
from vllm.scalar_type import scalar_types

from vllm.model_executor.layers.quantization.utils.marlin_utils_test_24 import (
    compress_quantized_24_weight,
    get_weight_perm_24,
    marlin_weights,
    marlin_permute_scales_24,
)

from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    MarlinWorkspace, marlin_quantize)

from vllm.model_executor.layers.quantization.gptq_marlin_24 import (
    GPTQ_MARLIN_24_MAX_PARALLEL, GPTQ_MARLIN_24_MIN_THREAD_N)

from vllm import _custom_ops as ops
from types import MethodType

from tqdm import tqdm

from slim.eval import eval_ppl_wikitext
from slim.data import get_wikitext2


def load_compressed_model(model_path):
    """
    Load a compressed model from the specified path.

    Args:
        model_path (str): The path to the model directory.

    Returns:
        tuple: A tuple containing the model, tokenizer, configuration dictionary, and LoRA hooks.
    """
    # Load configuration from args.json
    with open(os.path.join(model_path, 'args.json')) as f:
        args = json.load(f)
    
    print("Loading model configuration...")
    print(f"Model: {args['model']}")
    print(f"Pruning method: {args['prune_method']}")
    print(f"Sparsity ratio: {args['sparsity_ratio']}")
    print(f"Sparsity type: {args['sparsity_type']}")
    print(f"Quantize weights: {args['quantize_weight']}")
    print(f"LoRA rank: {args['lora_rank']}")
    print(f"Separate LoRA: {args['separate_lora']}")
    print(f"Quantize LoRA: {args['quantize_lora']}")
    
    # Load base model arch
    print("Loading base model architecture...")
    model = AutoModelForCausalLM.from_pretrained(
        args['model'], 
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    
    # Determine if we need LoRA (simple check: lora_rank > 0)
    has_lora = args.get('lora_rank', 0) > 0
    lora_hooks = []
    if has_lora and args.get('separate_lora', False):
        # Note that add_empty_lora already attaches the hooks
        print("Adding LoRA architecture...")
        lora_tile_size = args['lora_tile_size'] if (args.get('quantize_lora', False) or args.get('pad_lora', False)) else None
        lora_hooks = add_empty_lora(
            model,
            lora_tile_size=lora_tile_size,
            lora_rank=args['lora_rank']
        )
    # Load the full state dict
    print("Loading state dictionary...")
    final_state_dict = {}
    
    # Use model.safetensors.index.json to figure out the filenames
    with open(os.path.join(model_path, 'model.safetensors.index.json')) as f:
        index_data = json.load(f)
        filenames = set()
        for file in index_data['weight_map'].values():
            filenames.add(file)
    
    files = list(filenames)
    for file in (pbar := tqdm(files, desc="Loading state dict files")):
        pbar.set_description_str(f"Loading file {file}")
        load_files_dict = safetensors.torch.load_file(os.path.join(model_path, file))
        final_state_dict.update(load_files_dict)
    
    # Load the state dict into the model
    print("Loading state dict into model...")
    matched_keys, unmatched_keys = model.load_state_dict(final_state_dict, strict=False)
    
    # For all unmatched keys, go to their respective layers and register them as buffers
    if unmatched_keys:
        print('Adding extra unmatched keys (scales and other metadata)')
        for key in (pbar := tqdm(unmatched_keys)):
            pbar.set_description_str(f"Registering buffer for {key}")

            splits = key.split('.')
            buffer_name = splits[-1]
            module_name = '.'.join(splits[:-1])

            device = next(model.get_submodule(module_name).parameters()).device

            buffer = final_state_dict[key].to(device)
            model.get_submodule(module_name).register_buffer(buffer_name, buffer)

    # Apply input quantization hooks if they were used
    if args.get('quantize_input', False):
        print("Setting up input quantization hooks...")
        from slim.quantization.quantization import attach_input_quantization_hooks
        attach_input_quantization_hooks(model,
                                        args['input_bitwidth'],
                                        args['input_group_size'])
    
    print("Model loaded successfully!")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args['model'], 
        local_files_only=args.get('local_files_only', False)
    )
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, args, lora_hooks


def quantized_slim_forward(module, x):
    """
    Forward pass for quantized slim model. If quantization is enabled, it will use quantized operations.

    Args:
        module: The module to forward through.
        x: The input tensor.

    Returns:
        The output tensor.
    """
    marlin_24_q_w_comp = module.marlin_24_q_w_comp
    marlin_24_s = module.marlin_24_s
    marlin_24_meta = module.marlin_24_meta
    quant_type = module.quant_type
    marlin_24_workspace = module.marlin_24_workspace
    
    d_out = module.out_features
    
    if x.dim() == 3:
        bs, seqlen, d_in = x.shape
    else:
        bs, d_in = x.shape
        seqlen = 1

    x_2d = x.view(-1, d_in)

    # Use gemm to carry out the forward
    output_2d = ops.gptq_marlin_24_gemm(
        x_2d,
        marlin_24_q_w_comp,
        marlin_24_meta,
        marlin_24_s,
        marlin_24_workspace.scratch,
        quant_type,
        bs * seqlen,
        d_out,
        d_in,
    )
    
    # Original lora
    if hasattr(module, "lora_left"):
        xl = torch.matmul(x_2d, module.lora_left)
        torch.addmm(output_2d, xl, module.lora_right, out=output_2d)
    elif hasattr(module, "marlin_q_L"):
        # Quantized lora
        xl_dout = module.marlin_q_L.shape[1] // 2
        xl = ops.gptq_marlin_gemm(
            a=x_2d,
            c=None,
            b_q_weight=module.marlin_q_L,
            b_scales=module.marlin_s_L,
            global_scale=None,
            b_zeros=None,
            g_idx=module.marlin_g_idx_L,
            perm=module.marlin_sort_indices_L,
            workspace=module.marlin_24_workspace.scratch,
            b_q_type=module.quant_type,
            size_m=bs*seqlen,
            size_n=xl_dout,
            size_k=d_in,
            is_k_full=False,
            use_atomic_add=False,
            use_fp32_reduce=False,
            is_zp_float=False,
        )

        xlr_dout = module.marlin_q_R.shape[1] // 2
        xlr = ops.gptq_marlin_gemm(
            a=xl,
            c=None,
            b_q_weight=module.marlin_q_R,
            b_scales=module.marlin_s_R,
            global_scale=None,
            b_zeros=None,
            g_idx=module.marlin_g_idx_R,
            perm=module.marlin_sort_indices_R,
            workspace=module.marlin_24_workspace.scratch,
            b_q_type=module.quant_type,
            size_m=bs*seqlen,
            size_n=xlr_dout,
            size_k=xl_dout,
            is_k_full=False,
            use_atomic_add=False,
            use_fp32_reduce=False,
            is_zp_float=False,
        )
        
        output_2d.add_(xlr)
    
    if x.dim() == 3:
        output = output_2d.view(bs, seqlen, d_out)
    else:
        output = output_2d
    
    if module.bias is not None:
        output.add_(module.bias)  # In-place add

    return output


def replace_module(args, 
                   quantizer, 
                   m,
                   lora_rank: float, 
                   quantize_lora: bool = False, 
                   lora_tile_size: int = 0, 
                   slim_quant: bool = False,
                   quant_type = scalar_types.uint4b8):
    """
    Replace the module with a sparse and/or quantized version.

    Args:
        quantizer: The quantizer to use.
        m: The module to replace.
        lora_rank: The rank for LoRA.
        quantize_lora: Whether to quantize the LoRA weights.
        lora_tile_size: The tile size for LoRA.
        slim_quant: Whether to use SLiM-Quant.
    
    Returns:
       The replaced module.
    """
    group_size = args.get('weight_tile_size', 128)
    min_q = -(2 ** (args.get('bitwidth', 4) - 1))
    max_q = 2 ** (args.get('bitwidth', 4) - 1) - 1
    midpoint = 2 ** (args.get('bitwidth', 4) - 1)

    # === Weight ===
    # Compute all relevant parameters
    weight = m.weight.data
    
    device = weight.device
    scales = m.quantization_scaling_factor
    
    if slim_quant:
        # We should use the saved quantization parameters for SLiM-Quant since they're based on the original weights
        int_weight = (weight * scales).round().clamp(min_q, max_q)
    else:
        int_weight = quantizer.quantize_weight(weight)
    
    d_out = weight.size(0)
    d_in = weight.size(1)
    
    # Create a marlin workspace
    marlin_workspace = MarlinWorkspace(d_out, GPTQ_MARLIN_24_MIN_THREAD_N, GPTQ_MARLIN_24_MAX_PARALLEL)

    marlin_workspace.scratch = marlin_workspace.scratch.to(device)

    # Attach the workspace
    m.marlin_24_workspace = marlin_workspace

    size_k = d_in
    size_k_comp = size_k // 2
    size_n = d_out

    # Compress quantized weight
    # Need to cast to int32
    q_w_24_comp, marlin_24_meta = compress_quantized_24_weight((int_weight.t() + midpoint).to(torch.int32), size_k, size_n, quant_type)

    # Reformat to marlin
    weight_perm = get_weight_perm_24(quant_type.size_bits)
    marlin_24_q_w_comp = marlin_weights(q_w_24_comp, size_k_comp, size_n,
                                        quant_type.size_bits, weight_perm)

    if slim_quant:
        # Slim quant has another definition of scales
        # (w * (scales / max_q)).round() = int_w
        
        # vllm scales expect to be of the form: 
        # (w / scales).round() = int_w
        scales = (max_q / scales).expand(d_out, d_in // group_size)

    # Regular quant's definition of scales is
    # (w / (scales * max_q)).round() = int_w
    # vllm expects scales to be of the form:
    # (w / scales).round() = int_w
    marlin_24_s = marlin_permute_scales_24((scales.t() / max_q).to(torch.float16), size_k, size_n, group_size)

    # Store the relevant information
    m.marlin_24_q_w_comp = torch.nn.Parameter(marlin_24_q_w_comp, requires_grad=False)
    m.marlin_24_meta = torch.nn.Parameter(marlin_24_meta, requires_grad=False)
    m.marlin_24_s = torch.nn.Parameter(marlin_24_s, requires_grad=False)
    m.quant_type = quant_type
    
    # === LORA ===
    # Deal with the lora component
    layer_rank = int(min(m.weight.shape) * lora_rank)
    if lora_tile_size is not None:
        residue = layer_rank % lora_tile_size
        if residue != 0:
            layer_rank = layer_rank + (lora_tile_size - residue)
        assert layer_rank % lora_tile_size == 0
        
    if hasattr(m, 'lora_left'):
        # Update the left and right matrices
        m.lora_left.data /= torch.sqrt(m.lora_rank)
        m.lora_right.data /= torch.sqrt(m.lora_rank)

    if quantize_lora:
        if lora_rank > 0:
            (marlin_ref_lora_left,
                    m.marlin_q_L,
                    m.marlin_s_L,
                    m.marlin_g_idx_L,
                    m.marlin_sort_indices_L,
                    m.marlin_rank_perfm_L) = marlin_quantize(m.lora_left,
                                                             quant_type,
                                                             lora_tile_size,
                                                             act_order=False)
            (marlin_ref_lora_right,
                m.marlin_q_R,
                m.marlin_s_R,
                m.marlin_g_idx_R,
                m.marlin_sort_indices_R,
                m.marlin_rank_perfm_R) = marlin_quantize(m.lora_right,
                                                         quant_type,
                                                         lora_tile_size,
                                                         act_order=False)
                
            m.lora_quant_type = quant_type
            
            # Remove the original lora
            del m.lora_left
            del m.lora_right
            
    # Replace the forward with the new forward
    m.forward = MethodType(quantized_slim_forward, m)
    
    # Remove the original parameters with the model
    del m.quantization_scaling_factor
    del m.weight


def load_and_accelerate_model(checkpoint_path):
    model, tokenizer, args, lora_hooks = load_compressed_model(checkpoint_path)
    
    if not args.get('column_wise_grouping', False):
        raise NotImplementedError("Row-wise grouping is not supported at the moment")

    if not args.get('slim_quant', False) and not args.get('tiled_weight_quantization', False):
        raise NotImplementedError("Non-tiled weight quantization is not supported at the moment if we're not using slim quantization")
        
    if args.get('bitwidth', False) != 4:
        raise NotImplementedError("Only 4-bit quantization is supported at the moment")

    if not args.get('slim_quant', False) and args.get('weight_tile_size', False) != 128:
        raise NotImplementedError("Only 128 weight tile size is supported at the moment if we're not using SLiM quantization")

    model = model.to(torch.float16)  # Need to cast the model to fp16 for the speedup
    
    # Get rid of all lora hooks, as we will use our own forward function
    for hook in lora_hooks:
        hook.remove()
    
    # Use a shared quantizer
    quantizer = AutoQuantizer(
        "weight",
        num_bits=args.get('bitwidth', 4),  # Default is 4 bits
        slim_quant=args.get('slim_quant', False),
        block_quantization=args.get('tiled_weight_quantization', False),
        block_dim=args.get('weight_tile_size', 128),  # Group size 128, 
        column_wise_grouping=args.get('column_wise_grouping', False), 
    )
    
    # This is limited to 4 bits symmetrical quant
    # We only support symmetric quant now
    quant_type = scalar_types.uint4b8

    # Filter out only the layers with quantization parameters stored
    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and hasattr(m, 'quantization_scaling_factor')]

    # Update all linear layers with the acceleration
    for n, m in tqdm(layers, desc='Converting to marlin accelerated layers'):
        replace_module(
            args,
            quantizer,
            m,
            lora_rank=args.get('lora_rank', 0.0),
            quantize_lora=args.get('quantize_lora', False),
            lora_tile_size=args.get('lora_tile_size', 0),
            slim_quant=args.get('slim_quant', False),
            quant_type=quant_type
        )
    
    return model, tokenizer


if __name__ == '__main__':
    # Example 1: Load model without acceleration
    parser = argparse.ArgumentParser(description='Load and evaluate a compressed model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint directory.')
    args = parser.parse_args()

    # Load the model
    model, tokenizer, cfg, lora_hooks = load_compressed_model(args.model_path)

    # Import evaluation components
    from slim.eval import eval_ppl_wikitext
    from slim.data import get_wikitext2

    try:
        print("Setting up model for evaluation...")
        model.eval()
        
        model.config.max_position_embeddings = 2048  # Set to eval ppl sequences with 2048
        
        print(f"Model dtype: {next(model.parameters()).dtype}")
        
        # Load WikiText2 test data
        print("Loading WikiText2 test dataset...")
        _, testenc = get_wikitext2(seed=0, tokenizer=tokenizer)
        
        print(f"Test dataset size: {testenc.input_ids.numel()} tokens")
        
        # Evaluate perplexity
        print("Evaluating perplexity on WikiText2...")
        ppl = eval_ppl_wikitext(
            model=model,
            testenc=testenc,
            bs=cfg.get('eval_batch_size', 1),
            device="cuda:0"
        )
        
        print(f"WikiText2 Perplexity: {ppl:.2f}")
        
    except Exception as e:
        print(f"Error during perplexity evaluation: {e}")
        import traceback
        traceback.print_exc()

    # Example 2: Load model with acceleration
    model, tokenizer = load_and_accelerate_model(args.model_path)

    try:
        print("Setting up model for evaluation...")
        model.eval()
        
        model.config.max_position_embeddings = 2048  # Set to eval ppl sequences with 2048

        # Load WikiText2 test data
        print("Loading WikiText2 test dataset...")
        _, testenc = get_wikitext2(seed=0, tokenizer=tokenizer)

        # Evaluate perplexity    
        start = time.time()
        
        torch.cuda.synchronize()
            
        ppl = eval_ppl_wikitext(
            model=model,
            testenc=testenc,
            bs=1,
            device="cuda:0"
        )
        
        torch.cuda.synchronize()
        
        end = time.time()
        
        print(f"Time taken: {end - start:.2f} seconds")
        print(f"WikiText2 Perplexity: {ppl:.2f}")
    
    except Exception as e:
        print(f"Error during perplexity evaluation: {e}")
        import traceback
        traceback.print_exc()