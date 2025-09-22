import torch
from slim.utils import get_layers_list, find_layers
from slim.quantization.utils import compute_quantization_params, quantize_tensor, dequantize_tensor


def compute_average_error(pdf, val, index, q):
    alpha = val[index]

    dx = val[1] - val[0]

    pdf_quantization = pdf[:index]
    accurate_val_quantization = val[:index]
    quantized_val_quantization = (val[:index] // (alpha / (2 ** (q)))) * (alpha / (2 ** (q)))
    quantization_loss = torch.sum(pdf_quantization * (accurate_val_quantization - quantized_val_quantization) ** 2) * dx

    pdf_clip = pdf[index:]
    val_clip = val[index:]
    clipping_loss = torch.sum(pdf_clip * (val_clip - alpha) ** 2) * dx
    total_loss = quantization_loss + clipping_loss
    return total_loss


def compute_error(mat, alpha, num_bits):
    max_q = 2 ** (num_bits - 1) - 1
    abs_max = alpha
    scaling_factor = max_q / abs_max
    quantized_mat = torch.round(mat * scaling_factor)
    quantized_mat = torch.clamp(quantized_mat, -max_q - 1, max_q)
    dequantized_mat = quantized_mat / scaling_factor
    norm = torch.norm(dequantized_mat - mat)
    return norm.item() if not torch.isnan(norm) else torch.inf


def find_optimal_quantiztion_cap(mat, num_bits=8, num_bins=4096, integrate=True):
    if integrate:
        pdf, val = torch.histogram(mat.data.abs().float().flatten().cpu(), bins=num_bins, density=True)
        pdf, val = pdf.cuda(0), val.cuda(0)

        val = (val[:-1] + val[1:]) / 2
        q = num_bits
        total_loss = torch.zeros(num_bins) + torch.inf

        losses = torch.zeros(11) + torch.inf
        indices = torch.zeros(11).to(torch.int)
        j = 0
        for i in range(0, num_bins, num_bins // 10):
            losses[j] = compute_average_error(pdf, val, i, q)
            indices[j] = i
            j += 1

        _, turning_point = torch.min(losses, 0)
        start = indices[max(turning_point - 1, 0)]
        end = indices[min(turning_point + 1, 10)]

        for i in range(start, end):
            total_loss[i] = compute_average_error(pdf, val, i, q)

        min_loss, idx = torch.min(total_loss, 0)

        # if not (idx < end) and (idx > start):
        #     print(f"{(idx < end) and (idx > start)} - ({start}, {end}) => {idx}")
        return val[idx].to(mat.device)
    else:
        max_val = mat.abs().max()
        val = torch.linspace(0, max_val, num_bins).to(mat.device)
        total_loss = torch.zeros(num_bins) + torch.inf

        losses = torch.zeros(11) + torch.inf
        indices = torch.zeros(11).to(torch.int)
        j = 0
        for i in range(0, num_bins, num_bins // 10):
            losses[j] = compute_error(mat, val[i], num_bits)
            indices[j] = i
            j += 1

        _, turning_point = torch.min(losses, 0)

        start = indices[max(turning_point - 1, 0)]
        end = indices[min(turning_point + 1, 10)]

        for i in range(start, end):
            total_loss[i] = compute_error(mat, val[i], num_bits)

        min_loss, idx = torch.min(total_loss, 0)
        return val[idx].to(mat.device)


class Quantizer:
    def __init__(
            self,
            matrix_type,
            num_bits=8,
            group_size=-1,
            symmetric=True,
            eps=1e-4,
            slim_quant=False,
            block_quantization=False,
            block_dim=16,
            column_wise_grouping=True,
    ):
        self.matrix_type = matrix_type
        self.num_bits = num_bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.eps = eps
        self.slim_quant = slim_quant
        self.block_quantization = block_quantization
        self.block_dim = block_dim
        self.important_columns_scaling_factor = (1 / 2.)
        self.column_wise_grouping = column_wise_grouping

    def quantize(
            self,
            mat,
            num_bits=8
    ):
        if self.matrix_type == "weight":
            return self.quantize_weight(mat, num_bits)

        elif self.matrix_type == "input":
            return self.quantize_input(mat)

    def get_dtype(
            self,
            num_bits
    ):
        if num_bits <= 8:
            dtype = torch.int8
        elif num_bits <= 16:
            dtype = torch.int16
        else:
            dtype = torch.int32
        return dtype

    def quantize_weight(
            self,
            mat,
            important_columns=None
    ):
        self.important_columns = important_columns
        if self.block_quantization:
            assert mat.shape[0] % self.block_dim == 0 and mat.shape[
                1] % self.block_dim == 0, "Input matrix size is not divisible by block size"
            if self.slim_quant:
                raise NotImplementedError("SLiM-Quant is not supported for block quantization")

            self.dtype = mat.dtype
            if self.column_wise_grouping:
                self.scaling_factor, _ = compute_quantization_params(mat, 1, self.block_dim, symmetric=True)
            else:
                self.scaling_factor, _ = compute_quantization_params(mat, self.block_dim, 1, symmetric=True)
            self.scale_important_columns(mat)
            quantized_mat = quantize_tensor(mat, self.scaling_factor, None, self.num_bits)
        else:
            quantized_mat, scaling_factor = self.quantize_block(
                mat,
                self.num_bits,
                self.slim_quant,
            )
            self.scaling_factor = scaling_factor.reshape(1, 1)
        self.scale_important_columns(mat, multiply=True)
        return quantized_mat
    

    def scale_important_columns(
            self,
            mat, 
            multiply=False):
        if self.important_columns is not None:
            if multiply:
                mat[:, self.important_columns] *= self.important_columns_scaling_factor
            else:
                mat[:, self.important_columns] /= self.important_columns_scaling_factor
        else:
            self.important_columns = None


    def quantize_block(
            self,
            mat,
            num_bits=8,
            slim_quant=False,
            important_columns=None
    ):
        """absmax quantization"""
        dtype = self.get_dtype(num_bits)
        max_q = 2 ** (num_bits - 1) - 1
        if slim_quant:
            abs_max = find_optimal_quantiztion_cap(
                mat,
                num_bits,
                num_bins=max(512, min(torch.numel(mat) // 1000, 20000))
            )
        else:
            abs_max = mat.abs().max()
        
        self.scale_important_columns(mat, important_columns)

        scaling_factor = max_q / abs_max
        quantized_mat = torch.round((mat * scaling_factor).float())
        if slim_quant:
            max_q = 2 ** (num_bits - 1) - 1
        quantized_mat = torch.clamp(quantized_mat, -max_q - 1, max_q)

        return quantized_mat.to(dtype), scaling_factor


    def dequantize_absmax(
            self,
            quantized_mat,
            scaling_factor=None
    ):
        if scaling_factor is None:
            scaling_factor = self.scaling_factor

        if scaling_factor.shape == (1, 1):
            deq_mat = quantized_mat / scaling_factor
        else:
            deq_mat = dequantize_tensor(quantized_mat, scaling_factor, None, self.num_bits, dtype=self.dtype)
        self.scale_important_columns(deq_mat, multiply=True)

        return deq_mat

    def quantize_input(
            self,
            mat
    ):
        if self.group_size != -1:
            mat_shape = mat.shape
            mat = mat.view(-1, self.group_size)

        max_q = 2 ** (self.num_bits - 1) - 1
        if self.symmetric:
            range = mat.abs().max(dim=-1, keepdim=True)[0]
            zero_locs = range.abs() < (self.eps * max_q)
            range[zero_locs] = 1.
            mid_point = torch.zeros_like(range)
            scale = max_q / range
        else:
            mat_max = mat.max(dim=-1, keepdim=True)[0]
            mat_min = mat.min(dim=-1, keepdim=True)[0]
            range = (mat_max - mat_min)
            zero_locs = range.abs() < (self.eps * max_q)
            range[zero_locs] = 1.
            mid_point = (mat_max + mat_min) / 2
            # TODO: Fix the asymmetric issue
            scale = (2 * max_q + 1) / range
        quantized_mat = torch.round((mat - mid_point) * scale)
        quantized_mat = torch.clamp(quantized_mat, -max_q - 1, max_q)

        self.zero_point = mid_point
        self.scaling_factor = scale
        if self.group_size != -1:
            quantized_mat = quantized_mat.view(mat_shape)

        return quantized_mat

    def dequantize_input(
            self,
            mat
    ):
        if self.group_size != -1:
            mat_shape = mat.shape
            mat = mat.view(-1, self.group_size)
        dequantized_mat = mat / self.scaling_factor + self.zero_point
        if self.group_size != -1:
            dequantized_mat = dequantized_mat.view(mat_shape)
        return dequantized_mat


def attach_input_quantization_hooks(
        model,
        num_bits=8,
        input_group_size=-1
):
    """
    Attach input quantization hooks to the model.

    Args:
        model: nn.Module, The model to attach the hooks to
        num_bits: int, The number of bits to quantize the input to
        input_group_size: int, The number of elements to quantize together 
            when num_bits!=8. If -1, per-token quantization is applied.
    """
    def input_quantization_pre_hook(module, input):
        if module.quantize_input:
            if num_bits != 8:
                quantized_input = module.quantizer.quantize(input[0])
                dequantized_input = module.quantizer.dequantize_input(quantized_input)
            else:
                max_val = input[0].abs().max()
                dtype = torch.float8_e4m3fn if max_val < 488 else torch.float8_e5m2
                dtype_maxval = torch.finfo(dtype).max
                quantized_input = (input[0] / max_val * (dtype_maxval - 1)).to(dtype)
                dequantized_input = (quantized_input).to(input[0].dtype) / (dtype_maxval - 1) * max_val

            input[0].data = dequantized_input

    layers = get_layers_list(model)
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            subset[name].register_forward_pre_hook(input_quantization_pre_hook)
            subset[name].quantize_input = True
            if num_bits != 8:
                subset[name].input_group_size = input_group_size
                subset[name].quantizer = Quantizer("input", num_bits=num_bits, group_size=input_group_size)


class QuantizedMatmul(torch.autograd.Function):
    """
    Both forward and backward are static methods.
    """

    @staticmethod
    def forward(ctx, input, weight, quantizer, mask=None, transpose_weight=False):
        if mask is not None:
            weight.data[mask] = 0.
        if quantizer is not None:
            quantized_weight = quantizer.quantize_weight(weight.data)
            dequantized_weight = quantizer.dequantize_absmax(quantized_weight)
        else:
            dequantized_weight = weight
        if transpose_weight:
            dequantized_weight = dequantized_weight.t()
        ctx.save_for_backward(input, dequantized_weight, torch.tensor(transpose_weight))
        result = torch.matmul(input, dequantized_weight)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, transpose_weight = ctx.saved_tensors
        grad_input = torch.matmul(grad_output, weight.t())
        grad_weight = torch.matmul(input.view(-1, input.shape[-1]).t(), grad_output.view(-1, grad_output.shape[-1]))
        if transpose_weight:
            grad_weight = grad_weight.t()
        return grad_input, grad_weight, None, None, None