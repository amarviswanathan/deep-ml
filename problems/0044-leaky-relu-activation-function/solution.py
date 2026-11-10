import numpy as np
def leaky_relu(z: float, alpha: float = 0.01) -> float|int:
    # Your code here
    z = np.asarray(z,dtype=np.float64)
    
    return np.where(z>0,z,alpha * z)
