import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	# Get max score
	max_score = max(scores)

	# shifted scores
	shifted = np.array(scores) - max_score

	# Log sum exp
	exp_shifted = np.exp(shifted)
	sum_exp = np.sum(exp_shifted)
	log_sum = np.log(sum_exp)

	return shifted - log_sum