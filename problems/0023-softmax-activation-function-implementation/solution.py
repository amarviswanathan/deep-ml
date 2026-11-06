import math

def softmax(scores: list[float]) -> list[float]:

    max_logit = max(scores)
    shifted = [score - max_logit for score in scores]
    e = [math.exp(v) for v in shifted]
    return [v/sum(e) for v in e]
