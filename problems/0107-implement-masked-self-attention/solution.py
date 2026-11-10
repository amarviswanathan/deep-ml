import torch
import math

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return torch.matmul(X, W_q), torch.matmul(X, W_k), torch.matmul(X, W_v)

def masked_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute masked self-attention.
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2,-1)
    scores = scores/math.sqrt(d_k)

    if mask is not None:
        scores = scores + mask
    
    # nrmalize
    attention_weights = torch.softmax(scores,dim=-1)

    return attention_weights @ V