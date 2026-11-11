# --- cell 1 ---
import numpy as np

def stable_sigmoid(x):
    
    out = np.zeros_like(x)
    pos = (x >= 0)
    neg = (x < 0)
    
    out[pos] = 1/(1 + np.exp(-x[pos]))
    out[neg] = np.exp(x[neg])/(1 + np.exp(x[neg]))
    
    return out
    
def relu(x):
    return np.maximum(0,x)
    

def activation(x):
    '''
    Apply an activation function element-wise.
    
    Args:
        x: numpy array of any shape (raw neuron outputs)
    
    Returns:
        numpy array of same shape (activated outputs)
    
    Requirements:
        - Must be non-linear (not just returning x)
        - Must work on arrays of any shape
        - Must be deterministic
    '''
    # TODO: Implement your activation function
    
    result = relu(x)  # Replace with your activation
    
    return result