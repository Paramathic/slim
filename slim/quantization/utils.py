import torch

import triton
import triton.language as tl


@triton.jit
def pow(x, y):
    return tl.exp(tl.log(x) * y)


@triton.jit
def quantize(x_vals, alpha, beta, q):
    """ Returns (x[i*block_size1:(i+1)*block_size1, j*block_size2:(j+1)*block_size2] - betas[i, j]) / alphas[i, j]
        x_vals: 2D array of FP16
        alpha: 2D array of FP16
        beta: 2D array of FP16
        q: quantization bitwidth
       """
    max_val = pow(2., q-1) - 1

    if beta is not None:
        x_vals = (x_vals - beta) / alpha * 2.0 * max_val
    else:
        x_vals = x_vals / alpha * max_val
    x_vals = tl.clamp(x_vals, -(max_val+1), max_val)
    x_vals_int = x_vals.to(tl.int8).to(x_vals.dtype)
    round_up = ((x_vals - x_vals_int) > 0.5).to(tl.int8) - ((x_vals - x_vals_int) < -0.5).to(tl.int8)
    x_vals = x_vals_int.to(tl.int8) + round_up

    return x_vals


@triton.jit
def dequantize(x_vals, alpha, beta, q):
    """ Returns (x[i*block_size1:(i+1)*block_size1, j*block_size2:(j+1)*block_size2] - betas[i, j]) / alphas[i, j]
        x_vals: 2D array of INT8
        alpha: 2D array of FP16
        beta: 2D array of FP16
        q: quantization bitwidth
       """
    max_val = pow(2., q-1) - 1

    x_vals = x_vals.to(alpha.dtype) / max_val * alpha
    if beta is not None:
        x_vals = x_vals / 2.0 + beta
    return x_vals.to(alpha.dtype)


@triton.jit
def get_block_ptrs(x, block_size1, block_size2, row_size, i, j):
    """ Returns x[i*block_size1:(i+1)*block_size1, j*block_size2:(j+1)*block_size2]
        x: 2D array of FP16
        block_size1: int
        block_size2: int
        row_size: int
        i: int
        j: int
       """
    offset1 = i * block_size1 + tl.arange(0, block_size1)
    offset2 = j * block_size2 + tl.arange(0, block_size2)
    x_ptrs = x + offset1[:, None] * row_size + offset2[None, :]
    return x_ptrs


@triton.jit
def _quantize_tensor(x,
                 y,
                 alphas,
                 betas,
                 block_size1: tl.constexpr,
                 block_size2: tl.constexpr,
                 row_size: tl.constexpr,
                 col_size: tl.constexpr,
                 q):
    """ Returns (x[i*block_size1:(i+1)*block_size1, j*block_size2:(j+1)*block_size2] - betas[i, j]) / alphas[i, j]
        x: Input 2D array of FP16
        y: Output 2D array of int8
        alphas: 2D array of FP16
        betas: 2D array of FP16
        block_size1: int
        block_size2: int
        q: quantization bitwidth
       """

    # Compute the start and end indices of the block
    i, j = tl.program_id(0), tl.program_id(1)
    #x_ptrs = get_block_ptrs(x, block_size1, block_size2, row_size, i, j)
    x_ptrs = tl.make_block_ptr(
        base=x,
        shape=(col_size, row_size),
        strides=(row_size, 1),
        offsets=(i*block_size1, j*block_size2),
        block_shape=(block_size1, block_size2),
        order=(1,0)
    )
    alphas_row_size = row_size // block_size2
    alpha = tl.load(alphas + i * alphas_row_size + j).to(tl.float16)
    if betas is not None:
        beta = tl.load(betas + i * alphas_row_size + j)
    else:
        beta = None

    # Load the input values
    x_vals = tl.load(x_ptrs)

    # Quantize the input values
    x_vals = quantize(x_vals, alpha, beta, q)

    # Store the quantized values
    #y_ptrs = get_block_ptrs(y, block_size1, block_size2, row_size, i, j)
    y_ptrs = tl.make_block_ptr(
        base=y,
        shape=(col_size, row_size),
        strides=(row_size, 1),
        offsets=(i*block_size1, j*block_size2),
        block_shape=(block_size1, block_size2),
        order=(1,0)
    )
    tl.store(y_ptrs, x_vals)


def quantize_tensor(x, alphas, betas, q):
    row_block_size, col_block_size = x.shape[0] // alphas.shape[0], x.shape[1] // alphas.shape[1]
    y = torch.empty_like(x, dtype=torch.int8, device=x.device)
    grid = lambda meta: (alphas.shape[0], alphas.shape[1])
    
    with torch.cuda.device(x.device):
        _quantize_tensor[grid](x, y, alphas, betas, row_block_size, col_block_size, x.shape[1], x.shape[0], q)
    return y


@triton.jit
def _dequantize_tensor(x,
                 y,
                 alphas,
                 betas,
                 block_size1: tl.constexpr,
                 block_size2: tl.constexpr,
                 row_size: tl.constexpr,
                 col_size: tl.constexpr,
                 q):
    """ Returns (x[i*block_size1:(i+1)*block_size1, j*block_size2:(j+1)*block_size2] - betas[i, j]) / alphas[i, j]
        x: Input 2D array of FP16
        y: Output 2D array of int8
        alphas: 2D array of FP16
        betas: 2D array of FP16
        block_size1: int
        block_size2: int
        q: quantization bitwidth
       """

    # Compute the start and end indices of the block
    i, j = tl.program_id(0), tl.program_id(1)
    #x_ptrs = get_block_ptrs(x, block_size1, block_size2, row_size, i, j)
    x_ptrs = tl.make_block_ptr(
        base=x,
        shape=(col_size, row_size),
        strides=(row_size, 1),
        offsets=(i*block_size1, j*block_size2),
        block_shape=(block_size1, block_size2),
        order=(1,0)
    )
    alphas_row_size = row_size // block_size2
    alpha = tl.load(alphas + i * alphas_row_size + j)
    if betas is not None:
        beta = tl.load(betas + i * alphas_row_size + j)
    else:
        beta = None

    # Load the input values
    x_vals = tl.load(x_ptrs)

    # Quantize the input values
    x_vals = dequantize(x_vals, alpha, beta, q)

    # Store the quantized values
    #y_ptrs = get_block_ptrs(y, block_size1, block_size2, row_size, i, j)
    y_ptrs = tl.make_block_ptr(
        base=y,
        shape=(col_size, row_size),
        strides=(row_size, 1),
        offsets=(i*block_size1, j*block_size2),
        block_shape=(block_size1, block_size2),
        order=(1,0)
    )
    tl.store(y_ptrs, x_vals)


def dequantize_tensor(x, alphas, betas, q, dtype=torch.float16):
    row_block_size, col_block_size = x.shape[0] // alphas.shape[0], x.shape[1] // alphas.shape[1]
    y = torch.empty_like(x, dtype=dtype)
    grid = lambda meta: (alphas.shape[0], alphas.shape[1])
    with torch.cuda.device(x.device):
        _dequantize_tensor[grid](x, y, alphas, betas, row_block_size, col_block_size, x.shape[1], x.shape[0], q)
    return y


def compute_quantization_params_torch(x, block_size_row, block_size_col, symmetric=False):
    param_row_cnt, param_col_cnt = x.shape[0] // block_size_row, x.shape[1] // block_size_col
    alphas = torch.empty(param_row_cnt, param_col_cnt, device=x.device, dtype=x.dtype)
    if symmetric:
        betas = None
    else:
        betas = torch.empty(param_row_cnt, param_col_cnt, device=x.device, dtype=x.dtype)
    for i in range(param_row_cnt):
        for j in range(param_col_cnt):
            block = x[i*block_size_row:(i+1)*block_size_row, j*block_size_col:(j+1)*block_size_col]
            if symmetric:
                alphas[i, j] = block.abs().max()
            else:
                block_min = block.flatten().min()
                block_max = block.flatten().max()
                alphas[i, j] = (block_max - block_min)
                betas[i, j] = (block_max + block_min) / 2
    return alphas, betas


@triton.jit
def _compute_quantization_params(x,
                                 alphas,
                                 betas,
                                 block_size1: tl.constexpr,
                                 block_size2: tl.constexpr,
                                 row_size: tl.constexpr,
                                 col_size: tl.constexpr,
                                 ):
    # Compute the start and end indices of the block
    i, j = tl.program_id(0), tl.program_id(1)
    # x_ptrs = get_block_ptrs(x, block_size1, block_size2, row_size, i, j)
    x_ptrs = tl.make_block_ptr(
        base=x,
        shape=(col_size, row_size),
        strides=(row_size, 1),
        offsets=(i * block_size1, j * block_size2),
        block_shape=(block_size1, block_size2),
        order=(1, 0)
    )

    alphas_row_size = row_size // block_size2

    # Load the input values
    x_vals = tl.load(x_ptrs)

    if betas is not None:
        max = tl.max(x_vals)
        min = tl.min(x_vals)
        alpha = max - min
        beta = (max + min) / 2.
        tl.store(betas + i * alphas_row_size + j, beta)
    else:
        alpha = tl.max(tl.abs(x_vals))

    tl.store(alphas + i * alphas_row_size + j, alpha)


def compute_quantization_params(x, block_size_row, block_size_col, symmetric=False):
    param_row_cnt, param_col_cnt = x.shape[0] // block_size_row, x.shape[1] // block_size_col
    alphas = torch.empty(param_row_cnt, param_col_cnt, device=x.device, dtype=x.dtype)
    if symmetric:
        betas = None
    else:
        betas = torch.empty(param_row_cnt, param_col_cnt, device=x.device, dtype=x.dtype)

    # reshape input data into 2D tensor
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    # heuristics for number of warps
    grid_size = (triton.cdiv(M, block_size_row), triton.cdiv(N, block_size_col))
    with torch.cuda.device(x.device):
        _compute_quantization_params[grid_size](
            x_arg, alphas, betas,
            block_size_row, block_size_col, N, M,
            )
    return alphas, betas



if __name__ == "__main__":
    dim1 = 1024
    b1, b2 = 128, 1
    for dim2 in [512 * i for i in range(2, 32)]:
        
        for symmetric in [True, False]:
            # Allocate memory
            dtype = torch.bfloat16
            x = torch.randn(dim1, dim2).cuda(0).to(dtype) + (torch.randn(1).cuda(0).to(dtype) * 3.)
            q = 4


            alphas, betas = compute_quantization_params(x, b1, b2, symmetric=symmetric)
            alphas2, betas2 = compute_quantization_params_torch(x, b1, b2, symmetric=symmetric)
            assert torch.norm(alphas.float() - alphas2.float()) < 1e-2
            if not symmetric:
                assert torch.norm(betas.float() - betas2.float()) < 1e-2

            # Launch the kernel
            block_size1 = x.shape[0] // alphas.shape[0]
            block_size2 = x.shape[1] // alphas.shape[1]
            y = torch.empty_like(x, dtype=torch.int8)

            scale = (2.0 ** (q - 1) - 1)
            for i in range(alphas.shape[0]):
                for j in range(alphas.shape[1]):
                    block = x[i*block_size1:(i+1)*block_size1, j*block_size2:(j+1)*block_size2]
                    if betas is not None:
                        y[i*block_size1:(i+1)*block_size1, j*block_size2:(j+1)*block_size2] = torch.round((block - betas[i, j]) / alphas[i, j] * 2.0 * scale).to(torch.int8)
                    else:
                        y[i*block_size1:(i+1)*block_size1, j*block_size2:(j+1)*block_size2] = torch.round(block / alphas[i, j] * scale).to(torch.int8)

            y_triton = quantize_tensor(x, alphas, betas, q)

            error = torch.norm(y.float() - y_triton.float()) / torch.norm(y.float())
            print("Quantization - Triton vs Torch: ", error)
            if q == 4:
                assert error < 5e-2, error
            else:
                assert error < 2e-2, error

            x_dequant = torch.empty_like(x)

            for i in range(alphas.shape[0]):
                for j in range(alphas.shape[1]):
                    block = y[i*block_size1:(i+1)*block_size1, j*block_size2:(j+1)*block_size2]
                    if betas is not None:
                        x_dequant[i*block_size1:(i+1)*block_size1, j*block_size2:(j+1)*block_size2] = (block.to(torch.float16) / scale / 2.0 * alphas[i, j] + betas[i, j])
                    else:
                        x_dequant[i*block_size1:(i+1)*block_size1, j*block_size2:(j+1)*block_size2] = (block.to(torch.float16) / scale * alphas[i, j])

            dequant_error = torch.norm(x.float() - x_dequant.float()) / torch.norm(x.float())
            print("Dequantization - Torch: ", dequant_error)
            assert dequant_error < 5e-2
            x_dequant_triton = dequantize_tensor(y_triton, alphas, betas, q, dtype=x.dtype)
            dequant_error_triton = torch.norm(x.float() - x_dequant_triton.float()) / torch.norm(x.float())
            print("Dequantization - Triton: ", dequant_error_triton)
            assert dequant_error_triton < 5e-2