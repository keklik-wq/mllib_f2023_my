import numpy as np


def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    n = len(predictions)
    squared_difference = (predictions - targets) ** 2
    return (1 / n) * np.sum(squared_difference)
