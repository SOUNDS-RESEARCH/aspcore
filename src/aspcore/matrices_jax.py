"""Functions for common operations on matrices.

Some examples include constructing a block matrix, ensure positive definiteness, applying a function to individual blocks of a block matrix.

References
----------
"""

from functools import partial

import jax
import jax.numpy as jnp


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
    if mat1.ndim > 5 or mat2.ndim > 5:
        raise NotImplementedError("param must be 4 or 5 dimensions")
    assert mat1.ndim == mat2.ndim, (
        "Both matrices must have the same number of dimensions."
    )

    def _matmul_param_broadcast(m1, m2):
        dim1, dim2, ir_len, _ = m1.shape
        dim2b, dim3, _, _ = m2.shape
        assert dim2 == dim2b, "The inner dimensions must match."

        # assert m1.dtype == m2.dtype, "The matrices must have the same dtype."
        def _matmul_param_inner(m1, m2):
            return m1[:, None, :, :] @ m2[None, :, :, :]

        all_outer_products = jax.vmap(_matmul_param_inner, in_axes=(1, 0))(m1, m2)
        mm_res = jnp.sum(all_outer_products, axis=0)
        return mm_res

    if mat1.ndim == 4:
        return _matmul_param_broadcast(mat1, mat2)
    matmul_result = jax.vmap(_matmul_param_broadcast, in_axes=(0, 0))(mat1, mat2)
    return matmul_result


@jax.jit
def param2blockmat(param):
    """Turns an array of blocks and returns a single blocks matrix

    Parameters
    ----------
    param : ndarray of shape (num_blocks1, num_blocks2, block_len, block_len) or (num_mats, num_blocks1, num_blocks2, block_len, block_len)
        In the latter case the operation is applied to each matrix independently.

    Returns
    -------
    block matrix : ndarray of shape (num_blocks1 * block_len, num_blocks2 * block_len)
        or (num_mats, num_blocks1 * block_len, num_blocks2 * block_len)
    """
    if param.ndim > 5:
        raise NotImplementedError("param must be 4 or 5 dimensions")

    def _param2blockmat_inner(mat):
        return jnp.concatenate(jnp.concatenate(mat, axis=1), axis=1)

    if param.ndim == 4:
        return _param2blockmat_inner(param)
    return jax.vmap(_param2blockmat_inner, in_axes=0)(param)


def blockmat2param(R, num_blocks, block_len):
    """Converts from a standard block matrix to a parametrized form

    Parameters
    ----------
    R : ndarray of shape (num_blocks * block_len, num_blocks * block_len)
        The input block matrix in standard form.
        Assumes for now that the matrix and the blocks are square.
    num_blocks : int
        The number of blocks in each dimension.
    block_len : int
        The length of each block.

    Returns
    -------
    param_mat : ndarray of shape (num_blocks, num_blocks, block_len, block_len)
        The output block matrix in parametrized form.

    Notes
    -----
    The parametrized form is a block matrix where each (identically sized) block is accessed as R[i,j,:,:].
    The standard form is a matrix, where the same data is accessed as R[i*len1:(i+1)*len1, j*len2:(j+1)*len2].
    """
    new_mat = R.reshape((num_blocks * block_len, num_blocks, block_len))
    new_mat = new_mat.reshape((num_blocks, block_len, num_blocks, block_len))
    new_mat = jnp.moveaxis(new_mat, 1, 2)
    return new_mat


def regularize_matrix_with_condition_number(mat, max_cond=1e10):
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
    all_evs = jnp.linalg.eigvalsh(
        mat
    )  # , subset_by_index=(mat.shape[-1]-1, mat.shape[-1]-1))
    max_ev = all_evs[-1]
    identity_scaling = max_ev / max_cond
    mat_reg = mat + identity_scaling * jnp.eye(mat.shape[-1])
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
    return jnp.kron(jnp.eye(num_blocks, dtype=int), block)


def generalized_eigh(A, B):
    """Computes the generalized eigenvalue decomposition of a pair of positive semidefinite matrices.

    Returns the same as scipy.linalg.eigh(A, B)

    Parameters
    ----------
    A : ndarray of shape (M, M)
        Hermitian matrix
    B : ndarray of shape (M, M)
        Positive definite matrix

    Returns
    -------
    eigenvalues : ndarray of shape (M,)
        Eigenvalues in ascending order
    eigenvectors : ndarray of shape (M, M)
        Eigenvectors in the columns
    """
    L = jnp.linalg.cholesky(B)
    L_inv = jnp.linalg.inv(L)
    C = L_inv @ A @ jnp.conj(L_inv.T)
    eigenvalues, eigenvectors_transformed = jnp.linalg.eigh(C)
    eigenvectors_original = jnp.conj(L_inv.T) @ eigenvectors_transformed
    return eigenvalues, eigenvectors_original


def generalized_eigvalsh(A, B):
    """Computes the generalized eigenvalue decomposition of a pair of positive semidefinite matrices.

    This can be used if only the eigenvalues are of interest, and not the eigenvectors.
    Returns the same as scipy.linalg.eigvalsh(A, B)

    Parameters
    ----------
    A : ndarray of shape (M, M)
        Hermitian matrix
    B : ndarray of shape (M, M)
        Positive definite matrix

    Returns
    -------
    eigenvalues : ndarray of shape (M,)
        Eigenvalues in ascending order
    """
    L = jnp.linalg.cholesky(B)
    L_inv = jnp.linalg.inv(L)
    C = L_inv @ A @ jnp.conj(L_inv.T)
    eigenvalues = jnp.linalg.eigvalsh(C)
    return eigenvalues


def matrix_sqrt(A):
    """Computes the square root of a positive definite matrix A.

    Clips eigenvalues below 1e-12 to avoid numerical instability.

    Parameters
    ----------
    A : ndarray of shape (M, M)
        Positive definite matrix

    Returns
    -------
    sqrt_A : ndarray of shape (M, M)
        The square root of A. Such that sqrt_A @ sqrt_A = A

    """
    eigvals, eigvecs = jnp.linalg.eigh(A)
    eigvals = jnp.maximum(eigvals, 1e-12)
    return eigvecs @ jnp.sqrt(jnp.diag(eigvals)) @ jnp.conj(eigvecs).T
