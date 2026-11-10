import math
import numpy as np

def sigmoid(z: float) -> float:
    # Implementing stable sigmoid
    x = np.asarray(z,dtype=np.float64)
    out = np.zeros_like(x,dtype=np.float64)
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    out[pos_mask] = 1/(1 + np.exp(-x[pos_mask]))
    out[neg_mask] = np.exp(x[neg_mask])/(1 + np.exp(x[neg_mask]))
    return out