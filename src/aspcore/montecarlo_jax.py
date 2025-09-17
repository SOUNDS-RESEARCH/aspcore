"""Functions for Monte Carlo integration

References
----------
"""
import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=['num_points'])
def uniform_random_on_circle(num_points, key):
    """Generate uniformly random points on the unit circle.

    num_points : int
        The number of points to generate
    key : jax random key
        the random key which determines the state of the PRNG
    """
    u = jax.random.uniform(key, shape=(num_points,))
    theta = 2 * jnp.pi * u

    points = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    return points



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



@partial(jax.jit, static_argnames=['num_points'])
def vonmises_fisher_on_sphere(num_points, mean_direction, kappa, key):
    """Generate points on the unit sphere according to the von Mises-Fisher distribution.

    This means it samples from the PDF p(x) = e^{kappa * mean_direction^T x}.
    The directions will be centered on mean_direction, with a concentration 
    given by kappa.

    Parameters
    ----------
    num_points : int
        The number of points to generate
    mean_direction : ndarray of shape (3,) or (1,3)
        The mean direction of the distribution
    kappa : float
        The concentration parameter of the distribution. Must be [0, +inf)
    key : jax random key
        The random key which determines the state of the PRNG

    Returns
    -------
    points : ndarray of shape (num_points, 3)
        The points on the unit sphere

    References
    ----------
    https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
    """
    if mean_direction.ndim == 2:
        assert mean_direction.shape[0] == 1, "Not implemented yet for batch directions"
        mean_direction = mean_direction[0,:]
    mean_direction = jnp.asarray(mean_direction)
    mean_direction = mean_direction / jnp.linalg.norm(mean_direction)

    kappa_inv = jnp.where(kappa != 0, 1 / kappa, 0)

    key, subkey = jax.random.split(key)
    V = uniform_random_on_circle(num_points, subkey)

    key, subkey = jax.random.split(key)
    xi = jax.random.uniform(subkey, shape=(num_points,))

    W = 1 + kappa_inv * jnp.log(xi + (1 - xi) * jnp.exp(-2 * kappa))

    # vMF distributed variables with mean direction (0, 0, 1)
    w = jnp.concatenate([jnp.sqrt(1 - W**2)[:, None] * V, W[:, None]], axis=-1)

    # Rotate the points to the desired mean direction
    # rotvec = jnp.cross(jnp.array([0, 0, 1]), mean_direction)
    # rot = jax.scipy.spatial.transform.Rotation.from_rotvec(rotvec)
    # w = rot.apply(w)

    z_axis = jnp.array([0., 0., 1.])
    R = rotation_matrix_from_a_to_b(z_axis, mean_direction)

    u_v = (R @ w.T).T
    return u_v


def rotation_matrix_from_a_to_b(a, b):
    v = jnp.cross(a, b)
    s = jnp.linalg.norm(v)
    c = jnp.dot(a, b)

    v_unit = v / (s + 1e-15)

    K = jnp.array([[0, -v_unit[2], v_unit[1]],
                   [v_unit[2], 0, -v_unit[0]],
                   [-v_unit[1], v_unit[0], 0]])

    R = jnp.eye(3) + K * s + (K @ K) * (1 - c)

    # Handle special cases
    R_parallel = jnp.eye(3)
    perp = jnp.where(jnp.abs(a[0]) < 0.9,
                     jnp.array([1., 0., 0.]),
                     jnp.array([0., 1., 0.]))
    axis_180 = jnp.cross(a, perp)
    axis_180 /= jnp.linalg.norm(axis_180)
    K180 = jnp.array([[0, -axis_180[2], axis_180[1]],
                      [axis_180[2], 0, -axis_180[0]],
                      [-axis_180[1], axis_180[0], 0]])
    R_antiparallel = jnp.eye(3) + 2 * (K180 @ K180)

    R = jnp.where(s < 1e-8, jnp.where(c > 0, R_parallel, R_antiparallel), R)
    return R


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    key = jax.random.PRNGKey(0)
    num_points = 512
    p = vonmises_fisher_on_sphere(num_points, jnp.array([1, 0, 0]), 10.0, key)
    plt.plot(p[:,0])
    plt.show()