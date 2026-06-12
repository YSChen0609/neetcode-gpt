import numpy as np
from typing import Tuple, List

"""
BatchNorm normalizes across the batch for each feature (axis=0). 
This means BatchNorm depends on batch statistics and needs special handling for inference.
"""

class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        
        # cast lists to ndarray
        running_mean = np.array(running_mean)
        running_var = np.array(running_var)
        
        if training:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            # update the running mean and var
            running_mean = (1-momentum) * running_mean + momentum * batch_mean
            running_var = (1-momentum) * running_var + momentum * batch_var

            # get the normed batch (training)
            normed_batch = np.round(
                gamma * ((x-batch_mean) / np.sqrt(batch_var+eps)) + beta,
                4
            )
        else: # inferencing
            normed_batch = np.round(
                gamma * ((x-running_mean) / np.sqrt(running_var+eps)) + beta,
                4
            )
        
        return normed_batch.tolist(), \
               np.round(running_mean, 4).tolist(), \
               np.round(running_var, 4).tolist()







