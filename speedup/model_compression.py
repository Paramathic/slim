import torch
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils_test_24 import (
    marlin_24_quantize,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    MarlinWorkspace,
    marlin_quantize,
)
from vllm.model_executor.layers.quantization.gptq_marlin_24 import (
    GPTQ_MARLIN_24_MAX_PARALLEL,
    GPTQ_MARLIN_24_MIN_THREAD_N,
)
from vllm import _custom_ops as ops
from types import MethodType


def slim_forward(module, input):
    if input.dim() == 3:
        bs, seqlen, d_in = input.shape
    else:
        bs, d_in = input.shape
        seqlen = 1
    xw = ops.gptq_marlin_24_gemm(
        input.view(-1, d_in),
        module.marlin_24_q_w_comp,
        module.marlin_24_meta,
        module.marlin_24_s,
        module.marlin_24_workspace.scratch,
        module.quant_type,
        bs * seqlen,
        module.d_out,
        d_in,
    )
    if hasattr(module, "lora_left"):
        xl = torch.matmul(input.view(-1, d_in), module.lora_left)
        torch.addmm(xw, xl, module.lora_right, out=xw)
    if input.dim() == 3:
        output = xw.view(bs, seqlen, module.d_out)
    else:
        output = xw

    if not module.bias is None:
        output += module.bias
    return output


def quantized_slim_forward(module, input):
    if input.dim() == 3:
        bs, seqlen, d_in = input.shape
    else:
        bs, d_in = input.shape
        seqlen = 1
    # if seqlen == 1:
    xw = ops.gptq_marlin_24_gemm(
        input.view(-1, d_in),
        module.marlin_24_q_w_comp,
        module.marlin_24_meta,
        module.marlin_24_s,
        module.marlin_24_workspace.scratch,
        module.quant_type,
        bs * seqlen,
        module.d_out,
        d_in,
    )
    if hasattr(module, "lora_left"):
        xl_dout = module.marlin_q_L.shape[1] // 2
        xl = ops.gptq_marlin_gemm(
            input.view(-1, d_in),
            module.marlin_q_L,
            module.marlin_s_L,
            torch.empty(1),
            module.marlin_g_idx_L,
            module.marlin_sort_indices_L,
            module.marlin_24_workspace.scratch,
            module.quant_type,
            bs * seqlen,
            xl_dout,
            d_in,
            is_k_full=False,
            has_zp=False,
            use_fpt32_reduce=False,
        )
        xlr_dout = module.marling_q_R.shape[1] // 2
        xlr = ops.gptq_marlin_gemm(
            xl,
            module.marlin_q_R,
            module.marlin_s_R,
            torch.empty(1),
            module.marlin_g_idx_R,
            module.marlin_sort_indices_R,
            module.marlin_24_workspace.scratch,
            module.quant_type,
            bs * seqlen,
            xlr_dout,
            xl_dout,
            is_k_full=False,
            has_zp=False,
            use_fpt32_reduce=False,
        )
        xw.add_(xlr)
    if input.dim() == 3:
        output = xw.view(bs, seqlen, module.d_out)
    else:
        output = xw

    if not module.bias is None:
        output += module.bias
    return output


def compress_model(
    model,
    lora_rank=0.1,
    pad_lora=True,
    quantize_lora=False,
    lora_group_size=128,
    quant_type=scalar_types.uint4b8,
    group_size=-1,
    skip_layers=[],
):
    marlin_workspaces = {}
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            if layer in skip_layers:
                continue
            weight = layer.weight.data.clone().detach().cuda()
            # Create a MarlinWorkspace object for the layer
            d_out = weight.shape[0]
            if not d_out in marlin_workspaces:
                marlin_workspaces[d_out] = MarlinWorkspace(
                    d_out, GPTQ_MARLIN_24_MIN_THREAD_N, GPTQ_MARLIN_24_MAX_PARALLEL
                )
            layer.quant_type = quant_type
            layer.marlin_24_workspace = marlin_workspaces[d_out]
            layer.d_out = weight.shape[0]
            (
                marlin_24_w_ref,
                layer.marlin_24_q_w_comp,
                layer.marlin_24_meta,
                layer.marlin_24_s,
            ) = marlin_24_quantize(weight.t(), quant_type, group_size)

            if lora_rank > 0:
                rank = int(min(layer.weight.shape) * lora_rank)
                if pad_lora:
                    residue = rank % lora_group_size
                    if residue != 0:
                        # rank = rank + (lora_group_size - residue)
                        rank = rank - residue
                    assert rank % lora_group_size == 0
                lora_left = (
                    torch.randn(
                        [weight.shape[1], rank],
                        device=weight.device,
                        dtype=weight.dtype,
                    )
                    / 100
                )
                lora_right = (
                    torch.randn(
                        [rank, weight.shape[0]],
                        device=weight.device,
                        dtype=weight.dtype,
                    )
                    / 100
                )

            del layer.weight, weight, marlin_24_w_ref
            torch.cuda.empty_cache()

            if quantize_lora:
                if lora_rank > 0:
                    (
                        marlin_ref_lora_left,
                        layer.marlin_q_L,
                        layer.marlin_s_L,
                        layer.marlin_g_idx_L,
                        layer.marlin_sort_indices_L,
                        layer.marlin_rank_perfm_L,
                    ) = marlin_quantize(
                        lora_left, quant_type, lora_group_size, act_order=False
                    )
                    (
                        marlin_ref_lora_right,
                        layer.marlin_q_R,
                        layer.marlin_s_R,
                        layer.marlin_g_idx_R,
                        layer.marlin_sort_indices_R,
                        layer.marlin_rank_perfm_R,
                    ) = marlin_quantize(
                        lora_right, quant_type, lora_group_size, act_order=False
                    )

                layer.forward = MethodType(quantized_slim_forward, layer)

            else:
                if lora_rank > 0:
                    layer.lora_left = lora_left
                    layer.lora_right = lora_right
                layer.forward = MethodType(slim_forward, layer)


if __name__ == "__main__":
    model = torch.nn.Linear(1024, 1024, dtype=torch.float16, device="cuda")
    compress_model(model)
