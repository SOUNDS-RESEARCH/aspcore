"""Functions for Monte Carlo integration

References
----------
"""
import jax
import jax.numpy as jnp
from functools import partial



@partial(jax.jit, static_argnames=['num_points'])
def uniform_random_on_sphere(num_points, key):
    """Generate uniformly random points on the unit sphere.

    num_points : int
        The number of points to generate
    key : jax random key
        the random key which determines the state of the PRNG 

    Returns
    -------
    points : ndarray of shape (num_points, 3)
        The points on the unit sphere
    """

    points = jax.random.normal(key, shape=(num_points, 3))
    points = points / jnp.linalg.norm(points, axis=-1)[:,None]
    return points

