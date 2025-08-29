import torch
import transformers
from slim.quantization.quantization import Quantizer as AutoQuantizer
import json


def save_model(model, checkpoint_dir, args):
    """
    Save the model to the specified directory.
    
    Args:
        model (torch.nn.Module): The model to save.
        output_dir (str): The directory where the model will be saved.
    """

    for name, module in model.named_modules():
        if hasattr(module, 'lora_quantizer'):
            quantized_lora_left = module.lora_quantizer.quantize_weight(module.lora_left.data)
            module.lora_left.data = module.lora_quantizer.dequantize_absmax(quantized_lora_left).to(torch.bfloat16)
            module.register_buffer("lora_left_quantization_scaling_factor", module.lora_quantizer.scaling_factor)
        
            quantized_lora_right = module.lora_quantizer.quantize_weight(module.lora_right.data)
            module.lora_right.data = module.lora_quantizer.dequantize_absmax(quantized_lora_right).to(torch.bfloat16)
            module.register_buffer("lora_right_quantization_scaling_factor", module.lora_quantizer.scaling_factor)
    
    model.save_pretrained(checkpoint_dir)
    
    # Dump the args used to generate this save
    with open(f"{checkpoint_dir}/args.json", "w") as f:
        json.dump(vars(args), f)