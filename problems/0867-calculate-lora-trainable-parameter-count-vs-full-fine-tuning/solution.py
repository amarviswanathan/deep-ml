def lora_param_count(num_layers: int, d_model: int, num_target_matrices: int, rank: int) -> dict:
    """Compute LoRA vs full fine-tuning parameter counts."""

    # Full param = (d_model * d_model) * num_target_matrices * num_layers
    # LoRA param = ((d_model * r) + (d_model * r)) * num_target_matrices * num_layers
    # Full param / LoRA param = (d_model * d_model)/(d_model * r * 2) = d_model/(r*2)

    

    full_param_count = (d_model * d_model) * num_target_matrices * num_layers
    lora_param_count = ((d_model * rank) + (d_model * rank)) * num_target_matrices * num_layers
    compression_ratio = full_param_count/lora_param_count

    return {
        'lora_params' : lora_param_count,
        'full_params' : full_param_count,
        'compression_ratio': compression_ratio
    }
