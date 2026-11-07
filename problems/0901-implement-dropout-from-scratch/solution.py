import torch

def dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    # TODO: implement inverted dropout
    if not training or p == 0.0:
        return x

    if p == 1.0:
        return torch.zeros_like(x)

    keep = 1 - p

    mask = (torch.rand_like(x) < keep).to(x.dtype)

    
    return (x * mask)/keep 
