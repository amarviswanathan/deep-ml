import torch
import math
import torch.nn.functional as F
from typing import Tuple

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Query, Key, and Value matrices.

    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)

    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q, K, V

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)

    Returns:
        Attention output of shape (seq_len, d_k)
    """
    d_k = Q.shape[-1] # seq_len,d_k
    scores = Q @ K.transpose(-2,-1)/math.sqrt(d_k) # seq_len,seq_len
    weights = F.softmax(scores,dim=-1)
    return weights @ V # seq_len,d_k

def multi_head_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Compute multi-head attention.

    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads

    Returns:
        Attention output of shape (seq_len, d_model)
    """
    seq_len, d_model = Q.shape
    d_k = d_model // n_heads
    assert d_model % n_heads == 0

    def split(x):
        return x.view(seq_len,n_heads,d_k).transpose(0,1) # n_heads,seq_len,d_k
    
    out = self_attention(split(Q),split(K),split(V)) # n_heads,seq_len,d_k
    return out.transpose(0,1).reshape(seq_len,d_model) # seq_len, d_model