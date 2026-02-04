import jax.numpy as jnp
import aspcore.quadrature as quad


def t_design(order):
    """Returns the t-design directions for the given order.

    Supported orders are odd positive integers from 1 to 15 (inclusive).

    Parameters
    ----------
    order : int
        The order of the t-design. 

    Returns
    -------
    directions : ndarray of shape (num_directions, 3)
        The directions of the t-design points.

    References
    ----------
    The current t-designs are from Rob Womersley at:
    https://web.maths.unsw.edu.au/~rsw/Sphere/EffSphDes/
    """
    assert isinstance(order, int), "Order must be an integer."
    
    designs = {
        1: quad._t_design_1,
        3: quad._t_design_3,
        5: quad._t_design_5,
        7: quad._t_design_7,
        9: quad._t_design_9,
        11: quad._t_design_11,
        13: quad._t_design_13,
        15: quad._t_design_15,
    }

    assert order in designs, "The requested order is not supported. Supported orders are odd positive integers from 1 to 15 (inclusive)."
    return jnp.asarray(designs[order]())
    
