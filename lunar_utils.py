from functools import partial

from icecream import ic
from jax import jit
import jax
import jax.numpy as jnp
import numpy as onp

def classify_goal(x):
    # result = onp.zeros_like(x, dtype=onp.int32)
    # result[x < -0.33] = 0
    # result[(x >= -0.33) & (x < 0.33)] = 1
    # result[x >= 0.33] = 2
    # return result
    return onp.select(
        [x < -0.33, (x >= -0.33) & (x < 0.33), x >= 0.33],
        [0, 1, 2],
    )

def embedding_1he(x):
    goals = classify_goal(x)
    result = jax.nn.one_hot(goals, 3)
    return result, None
    
def normalize_freq_matrix(joint_freq_matrix):
    if jnp.sum(joint_freq_matrix) < 1e-9:
        return jnp.zeros((3, 3))
    
    return joint_freq_matrix / onp.sum(joint_freq_matrix)

def mutual_information(joint_freq_matrix):
    # Step 1: Normalize the joint frequency matrix
    joint_prob_matrix = joint_freq_matrix / onp.sum(joint_freq_matrix)
    
    # Step 2: Calculate marginal probability distributions
    marginal_x = onp.sum(joint_prob_matrix, axis=1)
    marginal_y = onp.sum(joint_prob_matrix, axis=0)
    
    # Step 3: Compute mutual information
    mutual_info = 0
    for i in range(joint_prob_matrix.shape[0]):
        for j in range(joint_prob_matrix.shape[1]):
            if joint_prob_matrix[i, j] > 0:  # Avoid log(0)
                mutual_info += joint_prob_matrix[i, j] * onp.log(joint_prob_matrix[i, j] / (marginal_x[i] * marginal_y[j]))
    
    return mutual_info
