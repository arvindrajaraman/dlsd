import jax
import jax.numpy as jnp

def grad_norm(grads):
    grad_norms = jax.tree_util.tree_map(jnp.linalg.norm, grads)
    grad_norm_avg = jax.tree_util.tree_reduce(lambda x, y: x + y, grad_norms) / len(grad_norms)
    return grad_norm_avg
