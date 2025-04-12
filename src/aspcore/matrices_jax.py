"""Functions for common operations on matrices.

Some examples include constructing a block matrix, ensure positive definiteness, applying a function to individual blocks of a block matrix. 

References
----------
"""
import numpy as np

import jax.numpy as jnp
import jax
from functools import partial


def matmul_param(mat1, mat2):
    """Multiplies two parametrized block matrices without explicitly converting to full matrices.

    Is equivalent to _blockmat2param(_param2blockmat(mat1) @ _param2blockmat(mat2), num_mic, ir_len). 

    Parameters
    ----------
    mat1 : np.ndarray of shape (dim1, dim2, ir_len, ir_len)
        The first matrix.
    mat2 : np.ndarray of shape (dim2, dim3, ir_len, ir_len)
        The second matrix.

    Returns
    -------
    np.ndarray of shape (dim1, dim3, ir_len, ir_len)
        The product matrix.
    
    """
    dim1, dim2, ir_len, _ = mat1.shape
    dim2b, dim3, _, _ = mat2.shape
    assert dim2 == dim2b, "The inner dimensions must match."
    assert mat1.dtype == mat2.dtype, "The matrices must have the same dtype."
    def _matmul_param_inner(m1, m2):
        return m1[:,None,:,:] @ m2[None,:,:,:]
    
    all_outer_products = jax.vmap(_matmul_param_inner, in_axes=(1, 0))(mat1, mat2)
    matmul_result = jnp.sum(all_outer_products, axis=0)
    return matmul_result


@jax.jit
def param2blockmat(param):
    """Turns an array of blocks and returns a single blocks matrix

    Parameters
    ----------
    param : ndarray of shape (num_blocks1, num_blocks2, ir_len, ir_len)

    Returns
    -------
    block matrix : ndarray of shape (num_blocks1 * ir_len, num_blocks2 * ir_len)
    """
    return jnp.concatenate(jnp.concatenate(param, axis=1), axis=1)


def regularize_matrix_with_condition_number(mat, max_cond= 1e10):
    """Adds a scaled identity matrix to the matrix in order to ensure a maximum condition number

    Parameters
    ----------
    mat : ndarray of shape (a, a)
        Matrix to be regularized
    max_cond : float, optional
        maximum condition number allowed. Must be positive. The default is 1e10.

    Returns
    -------
    mat_reg : ndarray of shape (a, a)
        Regularized matrix
    """
    all_evs = jnp.linalg.eigvalsh(mat)#, subset_by_index=(mat.shape[-1]-1, mat.shape[-1]-1))
    max_ev = all_evs[-1]
    identity_scaling = max_ev / max_cond
    mat_reg = mat + identity_scaling * np.eye(mat.shape[-1])
    return mat_reg

@partial(jax.jit, static_argnames=["num_blocks"])
def block_diagonal_same(block, num_blocks):
    """Creates a block diagonal matrix from a single block. 
    
    Parameters
    ----------
    block : ndarray of shape (block_size, block_size)
        The block to be repeated.
    num_blocks : int
        The number of times the block is repeated.
    
    Returns
    -------
    block_diag : ndarray of shape (num_blocks * block_size, num_blocks * block_size)
        The block diagonal matrix.
    """
    return jnp.kron(jnp.eye(num_blocks, dtype = int), block)