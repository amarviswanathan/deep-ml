import numpy as np

def optimizer_step(param, grad, state, lr):
    '''
    Update a parameter using its gradient.
    
    Args:
        param: numpy array - current parameter values (any shape)
        grad: numpy array - gradient of loss w.r.t. param (same shape)
        state: dict - persists between calls, use to store any needed values
                      Example: state = {'step': 5, 'momentum': np.array([...])}
                      First call: state = {} (empty dict)
        lr: float - learning rate
    
    Returns:
        new_param: numpy array - updated parameter (must be same shape as param)
        state: dict - updated state dictionary
    '''
    # TODO: Implement your optimizer update rule
    # Hint: Think about gradient descent and how to use the gradient to update the parameter
    
    new_param = param - (lr * grad)  # Replace with your update rule
    
    return new_param, state
