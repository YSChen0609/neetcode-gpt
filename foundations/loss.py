import numpy as np
from numpy.typing import NDArray

"""
Cross Entropy Loss is actually dirived from KL-div where we try to minimize the KL-div
between the Ground-truth distribution and the predicted distribution.
And since some terms of it is not depending on model parameter, we can reduce it to
[-(p*)(logp)] where p* is the ground truth prob. of the class given the features and p is the predicted one.

we can do:
1. element-wise mul. + sum
    more preferred, for it is more efficient
2. mat. mul. (y_true @ np.log(y_pred).T)
    more elegant, but require trace and extra steps to get, and computationally more expensive.
"""

class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-7
        y_pred += epsilon
        return np.round(
            -np.mean(
                y_true * np.log(y_pred) + \
                (1-y_true) * np.log(1-y_pred)
            ),
            4
        )

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-7
        y_pred += epsilon

        return  np.round(
                    -np.mean(
                        np.sum(
                            y_true * np.log(y_pred), # shape will be: n_samples x n_classes
                            axis=1 
                        ) 
                    ), # shape will be: (n_samples, )
                    4
                )


