import torch

def epsilon_greedy(Q: torch.Tensor, epsilon: float = 0.1) -> int:
    """
    Selects an action using epsilon-greedy policy.
    Q: torch.Tensor of shape (n,) -- estimated action values
    epsilon: float in [0, 1]
    Returns: int, selected action index
    """

    n_arms = Q.shape[-1]
    if torch.rand(1).item() < epsilon:
        return torch.randint(0,n_arms,(1,)).item()
    else:
        return torch.argmax(Q).item()

