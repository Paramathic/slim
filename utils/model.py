from slim.utils import get_layers_list, find_layers
import torch
import lm_eval
import subprocess
import re
import psutil
from accelerate import infer_auto_device_map, dispatch_model

from accelerate.hooks import remove_hook_from_submodules
from copy import deepcopy

def get_llm(model_name,
            local_files_only=False,
            hf_token=None,
            seqlen=2048,
            ):
    """
    Load a model from transformers
    model_name: str, the name of the model to load
    local_files_only: bool, whether to only load local files
    hf_token: str, the huggingface token to use
    seqlen: int, the maximum sequence length to use
    """
    model_args = f"pretrained={model_name},dtype=half,local_files_only={local_files_only},low_cpu_mem_usage=True,token={hf_token}"
    lm_eval_model = lm_eval.api.registry.get_model("hf").create_from_arg_string(
        model_args,
        {
            'parallelize': True, # parallelize across gpu
            "device": None, # We load the model to GPU for proper inference through LM-Eval
        },
    )
    model = lm_eval_model._model

    orig_device_map = deepcopy(model.hf_device_map)

    device_map = {'': 'cpu'}
    model = dispatch_model(model, device_map=device_map)
    
    # Get rid of all the hooks that accelerate adds
    remove_hook_from_submodules(model)
    
    # model = lm_eval_model._model
    
    torch.cuda.empty_cache()
    model.config.max_position_embeddings = seqlen
    model.seqlen = seqlen
    return model, lm_eval_model, orig_device_map


def add_empty_lora(
        model,
        lora_tile_size=None,
        lora_rank=0.01,
):
    layer_list = get_layers_list(model)
    for i in range(len(layer_list)):
        layer = layer_list[i]
        subset = find_layers(layer)
        for name in subset:
            layer_rank = int(min(subset[name].weight.shape) * lora_rank)
            if lora_tile_size is not None:
                tile_dim = lora_tile_size
                residue = layer_rank % tile_dim
                if residue != 0:
                    layer_rank = layer_rank + (tile_dim - residue)
                assert layer_rank % tile_dim == 0
            subset[name].lora_left = torch.nn.Parameter(
                torch.zeros((subset[name].weight.shape[1], layer_rank), device=subset[name].weight.device).half())
            subset[name].lora_right = torch.nn.Parameter(
                torch.zeros((layer_rank, subset[name].weight.shape[0]), device=subset[name].weight.device).half())

            def add_lora_hook(module, input, output):
                output += torch.matmul(
                    torch.matmul(input[0].to(module.lora_left.dtype) / torch.sqrt(module.lora_rank), module.lora_left),
                    module.lora_right) / torch.sqrt(module.lora_rank)

            subset[name].lora_rank = torch.tensor(layer_rank)
            subset[name].register_forward_hook(add_lora_hook)


def contigous_model(model):
    for layer in get_layers_list(model):
        for param in layer.parameters():
            param.data = param.data.contiguous()
    return model

def get_gpu_info_torch():
    if not torch.cuda.is_available():
        return {}
    gpu_info = {}
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        total_memory_bytes = torch.cuda.get_device_properties(i).total_memory
        total_memory_gb = total_memory_bytes / 1024**3
        gpu_info[i] = f"{total_memory_gb:.2f}GB"
    return gpu_info


def get_cpu_memory():
    mem = psutil.virtual_memory()
    free_memory_gb = mem.available / 1024**3
    return f"{free_memory_gb:.2f}GB"


def get_max_memory():
    gpu_info = get_gpu_info_torch()

    max_memory = gpu_info
    max_memory["cpu"] = get_cpu_memory()
    return max_memory


def distribute_model(model, activation_buffer_percentage=0.30, device_map=None):
    """
    Distribute the model across all available GPUs
    """
    max_memory = get_max_memory()
    for device in max_memory:
        if device == "cpu":
            continue
        mem = max_memory[device]
        mem = re.sub(r'[^0-9.]', '', mem)
        mem = float(mem) * 1024**3
        mem = int(mem * (1 - activation_buffer_percentage))
        max_memory[device] = f"{mem // 1e9}GB"
    layer_list = get_layers_list(model)

    if device_map is None:
        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=[str(type(layer_list[0])).split('.')[-1]],
        )
        if any(d == 'meta' for d in device_map.values()):
            raise ValueError("Device map contains 'meta'. This shouldn't happen if model is already on CPU.")
        
    model = dispatch_model(model, device_map=device_map)
    
    return model



if __name__ == "__main__":
    from transformers import AutoTokenizer

    token = None

    def load_model_and_tokenizer(model_name):
        print("Loading model", model_name)
        get_llm(model_name, hf_token=token)
        AutoTokenizer.from_pretrained(model_name, token=token)

    #Load OPT models
    for size in ["125m", "350m", "1.3b", "2.7b", "6.7b", "13b"]:
        model_name = f"facebook/opt-{size}"
        load_model_and_tokenizer(model_name)

    #Load LLaMA-2 models
    for size in ["7b", "13b"]:
        model_name = f"meta-llama/Llama-2-{size}-hf"
        load_model_and_tokenizer(model_name)

    #Load LLaMA-3.1 models
    for size in ["8B"]:
        model_name = f"meta-llama/Llama-3.1-{size}"
        load_model_and_tokenizer(model_name)

    #Load LLaMA-3.2 models
    for size in ["1B", "3B"]:
        model_name = f"meta-llama/Llama-3.2-{size}"
        load_model_and_tokenizer(model_name)

    # Load Gemma-3 models
    for size in ["1b", "4b", "12b"]:
        model_name = f"google/gemma-3-{size}-pt"
        load_model_and_tokenizer(model_name)
