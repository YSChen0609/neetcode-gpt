import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))

        # init the weights and bias
        self.W, self.b = np.zeros(X.shape[1]), 0
        
        def train_step():
            # forward pass
            y_pred = X @ self.W.T + self.b 
            # loss = np.mean((y_pred-y)**2) # get loss

            # get gradients
            dL_dW = 2/(X.shape[0]) * X.T @ (y_pred-y)
            dL_db = 2/(X.shape[0]) * np.sum(y_pred-y)

            # update weights and bias
            self.W -= lr * dL_dW
            self.b -= lr * dL_db

        for _ in range(epochs):
            train_step()

        return np.round(self.W, 5), np.round(self.b, 5)





