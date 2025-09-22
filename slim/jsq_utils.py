import torch
import copy


def clip_matrix(matrix, abs=True, clip_l=0, clip_h=0, channel=False):
    if clip_l == 0 and clip_h == 0:
        return matrix

    if channel:
        print("Channel wise clip!")
        matrix_flatten = torch.t(matrix[0])
        if abs:
            matrix_flatten = torch.abs(torch.t(matrix[0]))
        max_threshold = None
        min_threshold = None

        if clip_h != 0:
            max_threshold = torch.quantile(matrix_flatten.double(), q=1 - clip_h, dim=0)
        clipped_matrix = torch.clamp(
            torch.t(matrix[0]), min=-max_threshold, max=max_threshold
        )
        return torch.t(clipped_matrix).unsqueeze(0).half()
    else:
        num_elements = matrix.numel()
        if abs:
            matrix_flatten = torch.abs(matrix).flatten()
        else:
            matrix_flatten = matrix.flatten()

        max_threshold = None
        min_threshold = None

        if clip_l != 0:
            low_index = int(clip_l * num_elements)
            min_threshold, _ = torch.topk(matrix_flatten, largest=False, k=low_index)
            min_threshold = min_threshold[-1]
        if clip_h != 0:
            high_index = int(clip_h * num_elements)
            max_threshold, _ = torch.topk(matrix_flatten, largest=True, k=high_index)
            max_threshold = max_threshold[-1]

        if abs:
            clipped_matrix = torch.clamp(matrix, -max_threshold, max_threshold)
        else:
            clipped_matrix = torch.clamp(matrix, min_threshold, max_threshold)

        return clipped_matrix


def generate_ss(activation, weight):
    cin, cout = weight.shape
    ss = torch.zeros_like(weight)
    for i in range(cout):
        w = copy.deepcopy(weight)
        w[:, i] = 0
        out = activation @ (w.t())
        max_values, _ = torch.max(out, dim=0)
        min_values, _ = torch.min(out, dim=0)
        row_ss = max_values - min_values
        ss[:, i] = row_ss
    ss = torch.where(torch.isinf(ss), torch.tensor(100, device=ss.device), ss)
    return ss
