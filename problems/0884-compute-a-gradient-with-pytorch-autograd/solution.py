import torch

def grad_of_quadratic(x_value: float) -> float:
    # TODO: build a tracked leaf for x, compute f(x), run backprop, return df/dx as a float

    x = torch.tensor(x_value,dtype=torch.float64,requires_grad=True)
    f_x = x ** 2 + 3*x + 2
    f_x.backward()
    # gradients are in x.grad and the scalar is .item()
    return x.grad.item()
