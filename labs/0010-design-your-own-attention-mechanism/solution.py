import numpy as np


def attention(Q, K, V):
    """
    Compute attention over a sequence.
    
    Args:
        Q: Query matrix, shape (batch_size, query_len, dim)
        K: Key matrix, shape (batch_size, key_len, dim)
        V: Value matrix, shape (batch_size, key_len, dim)
    
    Returns:
        output: Attended values, shape (batch_size, query_len, dim)
    
    The attention mechanism should:
    1. Compute compatibility between queries and keys
    2. Convert to attention weights (non-negative, sum to 1)
    3. Use weights to compute weighted sum of values
    """
    dim = Q.shape[-1]
    scores = np.matmul(Q,K.transpose(0,2,1))
    scaled_scores = scores/np.sqrt(dim)
    e_x = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
    weights = e_x / e_x.sum(axis=-1, keepdims=True)
    outputs = np.matmul(weights,V)
    # batch_size, query_len, dim = Q.shape
    # _, key_len, _ = K.shape
    
    # Your implementation here
    # Hint: Think about how queries "ask questions" and keys "provide answers"
    
    return outputs
