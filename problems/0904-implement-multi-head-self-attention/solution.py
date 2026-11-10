import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        # TODO: store d_head, create q/k/v/out projections (all bias=False)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.dim = d_model
        self.n_heads = num_heads
        self.head_dim = d_model // num_heads

        self.to_q = nn.Linear(self.dim,self.dim,bias=False)
        self.to_k = nn.Linear(self.dim,self.dim,bias=False)
        self.to_v = nn.Linear(self.dim,self.dim,bias=False)

        self.out_prj = nn.Linear(self.dim,self.dim,bias=False)


    def forward(self, x, mask=None):
        # x: (B, T, d_model); mask: (T, T) of 0 and -inf, or None

        B, T, d_model = x.shape
        Q = self.to_q(x)    # (B,T,d_model)
        K = self.to_k(x)
        V = self.to_v(x)

        Q = Q.view(B, T, self.n_heads, self.head_dim)
        Q = Q.transpose(1,2) # B, n_heads, T, head_dim

        K = K.view(B,T,self.n_heads,self.head_dim)
        K = K.transpose(1,2)

        V = V.view(B,T,self.n_heads,self.head_dim)
        V = V.transpose(1,2)

        scores = Q @ K.transpose(-1,-2) #B, n_heads, T, T 
        scores = scores/math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask # 0 and -inf, additive mask
        
        weights = F.softmax(scores,dim=-1)
        out = weights @ V # B, n_heads, T, head_dim 
        out = out.transpose(1,2) # B,T, n_heads,head_dim
        out = out.contiguous().view(B,T,d_model)

        return self.out_prj(out)
