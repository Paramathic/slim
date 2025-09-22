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
import torch.utils.benchmark as benchmark
import csv


def generate_speedup_csv(input_string, output_file="results/speedup_results.csv"):
    """
    Parses the input table string, calculates speedups, and saves the results to a CSV file.

    Parameters:
    - input_string: str, the raw input table as a string.
    - output_file: str, the name of the output CSV file.
    """
    # Parse the input string
    rows = []
    for line in input_string.strip().split("\n"):
        if not line.startswith(" ") and "|" not in line:
            continue  # Ignore metadata or non-table lines
        parts = [part.strip() for part in line.split("|")]
        if len(parts) > 1:  # Ensure it contains data
            rows.append(parts)

    # Separate header and data
    header = rows[0]
    data = rows[1:]

    # Create a list for output with speedup calculations
    output_data = [
        [
            "Configuration",
            "gptq_marlin_24_gemm_speedup",
            "lora_linear_fp16_speedup",
            "lora_linear_marlin_int4_speedup",
        ]
    ]

    for row in data:
        configuration = row[0]
        pytorch_gemm = float(row[1])
        speedups = [
            configuration,
            pytorch_gemm / float(row[2]),
            pytorch_gemm / float(row[3]),
            pytorch_gemm / float(row[4]),
        ]
        output_data.append(speedups)

    # Write the output CSV
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(output_data)

    print(f"Speedup results saved to {output_file}")


sizes = []

quantize_only = True

model_list = {
    "LLaMA-2-7B": {(4096, 4096), (4096, 11008), (11008, 4096)},
    "LLaMA-2-13B": {(5120, 5120), (5120, 13824), (13824, 5120)},
    "LLaMA-2-70B": {(8192, 8192), (8192, 28672), (28672, 8192)},
    # "LLaMA-3.1-8B": {(4096, 4096), (4096, 14336), (14336, 4096)},
    # 'LLaMA-3.1-70B': {(8192, 8192), (8192, 28672), (28672, 8192)},
    # 'LLaMA-3.1-405B': {(16384, 16384), (16384, 53248), (53248, 16384)},
}

num_experiments = 100


quant_type = scalar_types.uint4b8
group_size = -1
lora_group_size = 128

results = []

for model in model_list:
    for d_in, d_out in model_list[model]:
        for bs in [16, 32, 64]:
            print(
                "Testing model: ",
                model,
                " with d_in: ",
                d_in,
                " d_out: ",
                d_out,
                " bs: ",
                bs,
            )
            marlin_24_workspace = MarlinWorkspace(
                d_out, GPTQ_MARLIN_24_MIN_THREAD_N, GPTQ_MARLIN_24_MAX_PARALLEL
            )

            x = torch.randn(bs, d_in, dtype=torch.half, device="cuda")
            w = torch.randn(d_in, d_out, dtype=torch.half, device="cuda")
            if quantize_only:
                (
                    marlin_w_ref,
                    marlin_q_w,
                    marlin_s_w,
                    marlin_g_idx_w,
                    marlin_sort_indices_w,
                    marlin_rand_perm_w,
                ) = marlin_quantize(w, quant_type, group_size, act_order=False)
                marlin_zp_w = torch.zeros_like(marlin_s_w, dtype=torch.int)
                (marlin_24_w_ref, marlin_24_q_w_comp, marlin_24_meta, marlin_24_s) = (
                    None,
                    None,
                    None,
                    None,
                )
            else:
                (marlin_24_w_ref, marlin_24_q_w_comp, marlin_24_meta, marlin_24_s) = (
                    marlin_24_quantize(w, quant_type, group_size)
                )
                (
                    marlin_w_ref,
                    marlin_q_w,
                    marlin_s_w,
                    marlin_g_idx_w,
                    marlin_sort_indices_w,
                    marlin_rand_perm_w,
                    marlin_zp_w,
                ) = (None, None, None, None, None, None, None)

            lora_type = torch.float16

            r = int(0.1 * min(d_in, d_out))
            r -= r % lora_group_size
            L = torch.randint(
                size=[d_in, r], high=10, low=0, dtype=lora_type, device="cuda"
            )
            R = torch.randint(
                size=[r, d_out], high=10, low=0, dtype=lora_type, device="cuda"
            )

            (
                marlin_L_ref,
                marlin_q_L,
                marlin_s_L,
                marlin_g_idx_L,
                marlin_sort_indices_L,
                marlin_rand_perm_L,
            ) = marlin_quantize(L.half(), quant_type, lora_group_size, act_order=False)
            marlin_zp_L = torch.zeros_like(marlin_s_L, dtype=torch.int)

            (
                marlin_R_ref,
                marlin_q_R,
                marlin_s_R,
                marlin_g_idx_R,
                marlin_sort_indices_R,
                marlin_rand_perm_R,
            ) = marlin_quantize(R.half(), quant_type, lora_group_size, act_order=False)
            marlin_zp_R = torch.zeros_like(marlin_s_R, dtype=torch.int)

            print(
                "Metadata Generate - Used GPU Memory: ",
                torch.cuda.memory_allocated() // 1024 // 1024,
                " MB",
            )

            def lora_linear_fp16(
                x,
                l,
                r,
                marlin_24_q_w_comp,
                marlin_24_meta,
                marlin_24_s,
                marlin_q_w,
                marlin_s_w,
                marlin_zp_w,
                marlin_g_idx_w,
                marlin_sort_indices_w,
                marlin_24_workspace,
                quantize_only,
                quant_type,
            ):
                bs, d_in = x.shape
                d_out = r.shape[1]
                if quantize_only:
                    xw = ops.gptq_marlin_gemm(
                        x,
                        marlin_q_w,
                        marlin_s_w,
                        marlin_zp_w,
                        marlin_g_idx_w,
                        marlin_sort_indices_w,
                        marlin_24_workspace.scratch,
                        quant_type,
                        bs,
                        d_out,
                        d_in,
                        is_k_full=True,
                        has_zp=False,
                        use_fp32_reduce=False,
                    )
                else:
                    xw = ops.gptq_marlin_24_gemm(
                        x,
                        marlin_24_q_w_comp,
                        marlin_24_meta,
                        marlin_24_s,
                        marlin_24_workspace.scratch,
                        quant_type,
                        bs,
                        d_out,
                        d_in,
                    )

                xl = torch.matmul(x, l)
                torch.addmm(xw, xl, r, out=xw)
                return xw

            def lora_linear_marlin_int4(
                x,
                marlin_24_q_w_comp,
                marlin_24_meta,
                marlin_24_s,
                marlin_q_w,
                marlin_s_w,
                marlin_zp_w,
                marlin_g_idx_w,
                marlin_sort_indices_w,
                quantize_only,
                quant_type,
                marlin_24_workspace,
                marlin_q_L,
                marlin_s_L,
                marlin_zp_L,
                marlin_g_idx_L,
                marlin_sort_indices_L,
                marlin_q_R,
                marlin_s_R,
                marlin_zp_R,
                marlin_g_idx_R,
                marlin_sort_indices_R,
            ):
                bs, d_in = x.shape
                if quantize_only:
                    d_out = marlin_q_w.shape[1] // 2
                    xw = ops.gptq_marlin_gemm(
                        x,
                        marlin_q_w,
                        marlin_s_w,
                        marlin_zp_w,
                        marlin_g_idx_w,
                        marlin_sort_indices_w,
                        marlin_24_workspace.scratch,
                        quant_type,
                        bs,
                        d_out,
                        d_in,
                        is_k_full=True,
                        has_zp=False,
                        use_fp32_reduce=False,
                    )
                else:
                    d_out = marlin_24_q_w_comp.shape[1] // 2
                    xw = ops.gptq_marlin_24_gemm(
                        x,
                        marlin_24_q_w_comp,
                        marlin_24_meta,
                        marlin_24_s,
                        marlin_24_workspace.scratch,
                        quant_type,
                        bs,
                        d_out,
                        d_in,
                    )

                bs, xl_din = x.shape
                xl_dout = marlin_q_L.shape[1] // 2
                xl = ops.gptq_marlin_gemm(
                    x,
                    marlin_q_L,
                    marlin_s_L,
                    marlin_zp_L,
                    marlin_g_idx_L,
                    marlin_sort_indices_L,
                    marlin_24_workspace.scratch,
                    quant_type,
                    bs,
                    xl_dout,
                    xl_din,
                    is_k_full=True,
                    has_zp=False,
                    use_fp32_reduce=False,
                )

                bs, xlr_din = xl.shape
                xlr_dout = marlin_q_R.shape[1] // 2
                xlr = ops.gptq_marlin_gemm(
                    xl,
                    marlin_q_R,
                    marlin_s_R,
                    marlin_zp_R,
                    marlin_g_idx_R,
                    marlin_sort_indices_R,
                    marlin_24_workspace.scratch,
                    quant_type,
                    bs,
                    xlr_dout,
                    xlr_din,
                    is_k_full=True,
                    has_zp=False,
                    use_fp32_reduce=False,
                )
                xw.add_(xlr)
                return xw

            globals = {
                # Gen params
                "quantize_only": quantize_only,
                "quant_type": quant_type,
                "group_size": group_size,
                "lora_group_size": lora_group_size,
                "bs": bs,
                "d_out": d_out,
                "d_in": d_in,
                "x": x,
                # Marlin_24 params
                "marlin_24_w_ref": marlin_24_w_ref,
                "marlin_24_q_w_comp": marlin_24_q_w_comp,
                "marlin_24_meta": marlin_24_meta,
                "marlin_24_s": marlin_24_s,
                "marlin_24_workspace": marlin_24_workspace,
                # Marlin params
                "marlin_w_ref": marlin_w_ref,
                "marlin_q_w": marlin_q_w,
                "marlin_s_w": marlin_s_w,
                "marlin_zp_w": marlin_zp_w,
                "marlin_g_idx_w": marlin_g_idx_w,
                "marlin_sort_indices_w": marlin_sort_indices_w,
                # Kernels
                "gptq_marlin_gemm": ops.gptq_marlin_gemm,
                "gptq_marlin_24_gemm": ops.gptq_marlin_24_gemm,
                "gptq_marlin_repack": ops.gptq_marlin_repack,
                # LoRA params
                "lora_linear_fp16": lora_linear_fp16,
                "lora_linear_marlin_int4": lora_linear_marlin_int4,
                "L_fp16": L.half(),
                "R_fp16": R.half(),
                "L_int8": L,
                "R_int8": R,
                "marlin_q_L": marlin_q_L,
                "marlin_s_L": marlin_s_L,
                "marlin_zp_L": marlin_zp_L,
                "marlin_g_idx_L": marlin_g_idx_L,
                "marlin_sort_indices_L": marlin_sort_indices_L,
                "marlin_rand_perm_L": marlin_rand_perm_L,
                "marlin_q_R": marlin_q_R,
                "marlin_s_R": marlin_s_R,
                "marlin_zp_R": marlin_zp_R,
                "marlin_g_idx_R": marlin_g_idx_R,
                "marlin_sort_indices_R": marlin_sort_indices_R,
                "marlin_rand_perm_R": marlin_rand_perm_R,
            }

            label = "Quantized Matmul"
            min_run_time = 10
            sub_label = f"{model} - group_size={group_size}, bs={bs}, d_out={d_out}, d_in={d_in}"

            print("Testing Torch Matmul")
            dense_cmd = (
                "torch.matmul(x, marlin_w_ref)"
                if quantize_only
                else "torch.matmul(x, marlin_24_w_ref)"
            )
            results.append(
                benchmark.Timer(
                    stmt=dense_cmd,
                    globals=globals,
                    label=label,
                    sub_label=sub_label,
                    description="pytorch_gemm",
                ).blocked_autorange(min_run_time=min_run_time)
            )

            print("Testing compressed Matmul")
            if quantize_only:
                compressed_cmd = (
                    "output = gptq_marlin_gemm("
                    "x, "
                    "marlin_q_w, "
                    "marlin_s_w, "
                    "marlin_zp_w, "
                    "marlin_g_idx_w, "
                    "marlin_sort_indices_w, "
                    "marlin_24_workspace.scratch, "
                    "quant_type, "
                    "bs, "
                    "d_out, "
                    "d_in, "
                    "is_k_full=True, "
                    "has_zp=False, "
                    "use_fp32_reduce=False)"
                )
            else:
                compressed_cmd = (
                    "output = gptq_marlin_24_gemm("
                    "x, "
                    "marlin_24_q_w_comp, "
                    "marlin_24_meta, "
                    "marlin_24_s, "
                    "marlin_24_workspace.scratch, "
                    "quant_type, "
                    "bs, "
                    "d_out, "
                    "d_in)"
                )

            results.append(
                benchmark.Timer(
                    stmt=compressed_cmd,
                    globals=globals,
                    label=label,
                    sub_label=sub_label,
                    description="gptq_marlin_24_gemm",
                ).blocked_autorange(min_run_time=min_run_time)
            )

            print("Testing LoRA FP16")
            fp16_cmd = (
                "lora_linear_fp16("
                "x, "
                "L_fp16, "
                "R_fp16, "
                "marlin_24_q_w_comp, "
                "marlin_24_meta, "
                "marlin_24_s, "
                "marlin_q_w, "
                "marlin_s_w, "
                "marlin_zp_w, "
                "marlin_g_idx_w, "
                "marlin_sort_indices_w, "
                "marlin_24_workspace, "
                "quantize_only, "
                "quant_type)"
            )
            results.append(
                benchmark.Timer(
                    stmt=fp16_cmd,
                    globals=globals,
                    label=label,
                    sub_label=sub_label,
                    description="lora_linear_fp16",
                ).blocked_autorange(min_run_time=min_run_time)
            )

            print("Testing LoRA Marlin Int4")
            int4_cmd = (
                "lora_linear_marlin_int4("
                "x, "
                "marlin_24_q_w_comp, "
                "marlin_24_meta, "
                "marlin_24_s, "
                "marlin_q_w, "
                "marlin_s_w, "
                "marlin_zp_w, "
                "marlin_g_idx_w, "
                "marlin_sort_indices_w, "
                "quantize_only, "
                "quant_type, "
                "marlin_24_workspace, "
                "marlin_q_L, "
                "marlin_s_L, "
                "marlin_zp_L, "
                "marlin_g_idx_L, "
                "marlin_sort_indices_L, "
                "marlin_q_R, "
                "marlin_s_R, "
                "marlin_zp_R, "
                "marlin_g_idx_R, "
                "marlin_sort_indices_R)"
            )
            results.append(
                benchmark.Timer(
                    stmt=int4_cmd,
                    globals=globals,
                    label=label,
                    sub_label=sub_label,
                    description="lora_linear_marlin_int4",
                ).blocked_autorange(min_run_time=min_run_time)
            )


compare = benchmark.Compare(results)
result_tab = str(compare)
output_file = (
    "results/speedup_results.csv"
    if not quantize_only
    else "results/speedup_results_quantize_only.csv"
)
generate_speedup_csv(result_tab, output_file=output_file)
