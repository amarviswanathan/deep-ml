import numpy as np

def softmax(x:np.ndarray) -> np.ndarray:
    shift_x = x - np.max(x,axis=-1,keepdims=True)
    exps = np.exp(shift_x)
    return exps/np.sum(exps,axis=-1,keepdims=True)


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None) -> tuple:
    """
    Compute Scaled Dot-Product Attention.
    
    Args:
        Q: Query matrix of shape (seq_len_q, d_k)
        K: Key matrix of shape (seq_len_k, d_k)
        V: Value matrix of shape (seq_len_k, d_v)
        mask: Optional binary mask of shape (seq_len_q, seq_len_k) 0s for positions to block and 1s for positions to attend to
    
    Returns:
        Tuple of (output, attention_weights)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-2,-1)/np.sqrt(d_k) #seq_len,seq_len
    if mask is not None:
        scores = np.where(mask==0,-1e9,scores)

    weights = softmax(scores)
    
    return weights @ V, weights