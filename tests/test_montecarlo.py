import aspcore.montecarlo_jax as mc_jax
import aspcore.montecarlo as mc_numpy

import jax.numpy as jnp
import numpy as np
from jax import random
import matplotlib.pyplot as plt


def show_vonmises_fisher_distribution_in_jax_is_similar_to_scipy_implentation():
    key = random.PRNGKey(0)
    rng = np.random.default_rng()

    mean_dir = mc_numpy.uniform_random_on_sphere(1, rng)[0]
    kappa = 10.0

    num_points = 4096
    p_jax = mc_jax.vonmises_fisher_on_sphere(num_points, mean_dir, kappa, key)
    p_numpy = mc_numpy.vonmises_fisher_on_sphere(num_points, mean_dir, kappa, rng)

    #scatter plots in different 2D projections
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 2, 1)
    plt.title("JAX Implementation")
    plt.scatter(p_jax[:, 0], p_jax[:, 1])
    plt.subplot(3, 2, 2)
    plt.title("NumPy Implementation")
    plt.scatter(p_numpy[:, 0], p_numpy[:, 1])

    plt.subplot(3, 2, 3)
    plt.title("JAX Implementation (XZ Plane)")
    plt.scatter(p_jax[:, 0], p_jax[:, 2])
    plt.subplot(3, 2, 4)
    plt.title("NumPy Implementation (XZ Plane)")
    plt.scatter(p_numpy[:, 0], p_numpy[:, 2])

    plt.subplot(3, 2, 5)
    plt.title("JAX Implementation (YZ Plane)")
    plt.scatter(p_jax[:, 1], p_jax[:, 2])
    plt.subplot(3, 2, 6)
    plt.title("NumPy Implementation (YZ Plane)")
    plt.scatter(p_numpy[:, 1], p_numpy[:, 2])

    plt.tight_layout()
    plt.show()