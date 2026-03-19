"""Calculates distance measures for some different types of quantities.

For all types of arrays, the mean square error can be computed. Vectors can be compared using angular distance or cosine similarity. For positive definite matrices, the correlation matrix distance, the affine invariant Riemannian metric, and the Kullback Leibler divergence between zero-mean Gaussian densities described by the compared matrices can be computed.

References
----------
[herdinCorrelation2005] M. Herdin, N. Czink, H. Ozcelik, and E. Bonek, 'Correlation matrix distance, a meaningful measure for evaluation of non-stationary MIMO channels,' in 2005 IEEE 61st Vehicular Technology Conference, May 2005, pp. 136-140 Vol. 1. doi: 10.1109/VETECS.2005.1543265. `[link] <https://doi.org/10.1109/VETECS.2005.1543265>`__ \n
[forstnermetric2003] W. Förstner and B. Moonen, 'A metric for covariance matrices,' in Geodesy-The Challenge of the 3rd Millennium, E. W. Grafarend, F. W. Krumm, and V. S. Schwarze, Eds., Berlin, Heidelberg: Springer Berlin Heidelberg, 2003, pp. 299–309. doi: 10.1007/978-3-662-05296-9_31. `[link] <https://doi.org/10.1007/978-3-662-05296-9_31>`__ \n
[duchiDerivations2016] J. Duchi, 'Derivations for Linear Algebra and Optimization, 2016. `[link] <https://web.stanford.edu/~jduchi/projects/general_notes.pdf>`__ \n
[absilOptimization2008] P.-A. Absil, R. Mahony, and R. Sepulchre, Optimization algorithms on matrix manifolds. Princeton, N.J. ; Woodstock: Princeton University Press, 2008. \n
"""

# import numpy as np
from functools import partial

import jax
import jax.numpy as jnp
from jax._src.basearray import Array

import aspcore.matrices_jax as matop


def mse(var1, var2):
    """The normalized mean square error

    Normalized by the second variable

    Parameters
    ----------
    var1 : np.ndarray of any shape
        First variable
    var2 : np.ndarray of the same shape as var1
        Second variable. Cannot be zero, as it is used as the denominator

    Returns
    -------
    mse : float
        The normalized mean square error
    """
    return jnp.sum(jnp.abs(var1 - var2) ** 2) / jnp.sum(jnp.abs(var2) ** 2)


# ============== FOR VECTORS ======================
# @jax.jit(static_argnames=["sign_invariant"])
def angular_distance(vec1, vec2, sign_invariant=False):
    """A distance metric based on the cosine similary, that retains the
        scale invariant property, but is also a proper distance metric

    Parameters
    ----------
    vec1 : np.ndarray of shape (N,)
        First vector
    vec2 : np.ndarray of shape (N,)
        Second vector
    sign_invariant : bool, optional
        if True, the angle is first adjusted to a range between 1 and 0
        meaning that parallell vectors and opposite vectors are both considered to be the same
        If False, the same shape but opposite signs gives maximum distance

    Returns
    -------
    ang_dist : float
        The angular distance between the two vectors, in the range [0, 1]
    """
    similarity: Array = cos_similary(vec1, vec2)
    if sign_invariant:
        similarity: Array = jnp.abs(similarity)
    return jnp.arccos(similarity) / jnp.pi


def cos_similary(vec1, vec2):
    """Computes <vec1, vec2> / (||vec1|| ||vec2||) which is cosine of the angle between the two vectors.
        1 is paralell vectors, 0 is orthogonal, and -1 is opposite directions

    Is currently not possible to jit, due to the zero checking. Can be fixed if needed.

    Parameters
    ----------
    vec1 : np.ndarray of any shape
        First vector. If the arrays have more than 1 axis, it will be flattened.
    vec2 : np.ndarray of same shape as vec1
        Second vector. If the arrays have more than 1 axis, it will be flattened.

    Returns
    -------
    cos_sim : float
        The cosine similarity between the two vectors
    """
    assert vec1.shape == vec2.shape
    vec1 = jnp.ravel(vec1)
    vec2 = jnp.ravel(vec2)
    norms = jnp.linalg.norm(vec1) * jnp.linalg.norm(vec2)
    if norms == 0:
        return jnp.nan
    ip: Array = vec1.T @ vec2
    return ip / norms


def spatial_similarity(vec1, vec2):
    """Measures the spatial similarity between two vectors. Also known as the modal assurance criterion (MAC).

    1 is identical, 0 is fully dissimilar

    Implements |p^H q|^2 / (||p||^2 ||q||^2)

    Parameters
    ----------
    vec1 : np.ndarray of shape (..., N)
        First vector
    vec2 : np.ndarray of shape (..., N)
        Second vector

    Returns
    -------
    sim : float or ndarray of shape (...)
        The spatial similarity between the two vectors

    References
    ----------
    (25) in M. Hahmann and E. Fernandez-Grande, “A convolutional plane wave model for sound field reconstruction.” Aug. 24, 2022.
    """
    assert vec1.shape == vec2.shape

    denom = jnp.linalg.norm(vec1, axis=-1) ** 2 * jnp.linalg.norm(vec2, axis=-1) ** 2
    return jnp.abs(jnp.sum(vec1.conj() * vec2, axis=-1)) ** 2 / denom


# =============== FOR COVARIANCE MATRICES =============
def airm(mat1, mat2):
    """The affine invariant Riemannian metric distance between two positive definite matrices.

    Parameters
    ----------
    A : ndarray of shape (M, M)
        Positive definite matrix
    B : ndarray of shape (M, M)
        Positive definite matrix

    Returns
    -------
    distance : float
        The distance between A and B in AIRM distance
    """
    eigvals = matop.generalized_eigvalsh(mat1, mat2)
    return jnp.real(jnp.sqrt(jnp.sum(jnp.log(eigvals) ** 2)))


def wasserstein_distance(A, B):
    """Computes the Wasserstein distance between two zero-mean Gaussian distributions defined by two positive definite matrices.

    Parameters
    ----------
    A : ndarray of shape (M, M)
        Positive definite matrix
    B : ndarray of shape (M, M)
        Positive definite matrix

    Returns
    -------
    distance : float
        The distance between A and B in Wasserstein distance
    """
    A_sqrt = matop.matrix_sqrt(A)
    mix_term = A_sqrt @ B @ A_sqrt
    mix_term_sqrt = matop.matrix_sqrt(mix_term)
    return jnp.real(jnp.trace(A + B - 2 * mix_term_sqrt))


def corr_matrix_distance(mat1, mat2):
    """Computes the correlation matrix distance

    0 means that the matrices are equal up to a scaling
    1 means that they are maximally different (orthogonal in NxN dimensional space)

    Currently not possible to jit, due to the zero checking. Can be fixed if needed.

    Parameters
    ----------
    mat1 : np.ndarray of shape (..., N, N)
        First covariance matrix, should be symmetric and positive definite
    mat2 : np.ndarray of shape (..., N, N)
        Second covariance matrix, should be symmetric and positive definite

    References
    ----------
    Correlation matrix distaince, a meaningful measure for evaluation of
    non-stationary MIMO channels - Herdin, Czink, Ozcelik, Bonek
    """
    assert mat1.shape == mat2.shape
    norm1: Array = jnp.linalg.norm(mat1, ord="fro", axis=(-2, -1))
    norm2: Array = jnp.linalg.norm(mat2, ord="fro", axis=(-2, -1))
    if norm1 * norm2 == 0:
        return jnp.array(jnp.nan)
    return jnp.real(1 - jnp.trace(mat1 @ mat2) / (norm1 * norm2))


def covariance_distance_riemannian(mat1, mat2):
    """
    Computes the covariance matrix distance

    Parameters
    ----------
    mat1 : np.ndarray of shape (N, N)
        First covariance matrix, should be symmetric and positive definite
    mat2 : np.ndarray of shape (N, N)
        Second covariance matrix, should be symmetric and positive definite

    Returns
    -------
    dist : float
        The distance between the two matrices

    Notes
    -----
    It is the distance of a canonical invariant Riemannian metric on the space
    Sym+(n, R) of real symmetric positive definite matrices.

    Invariant to affine transformations and inversions.
    It is a distance measure, so 0 means equal and then it goes to infinity
    and the matrices become more unequal.

    When the metric of the space is the fisher information metric, this is the
    distance of the space. See COVARIANCE CLUSTERING ON RIEMANNIAN MANIFOLDS
    FOR ACOUSTIC MODEL COMPRESSION - Shinohara, Masukp, Akamine

    References
    ----------
    [forstnermetric2003]
    [absilOptimization2008]
    """
    assert mat1.shape == mat2.shape
    assert mat1.shape[0] == mat1.shape[1]
    assert mat1.ndim == 2
    eigvals: Array = jax.scipy.linalg.eigh(mat1, mat2, eigvals_only=True)
    return jnp.real(jnp.sqrt(jnp.sum(jnp.log(eigvals) ** 2)))


def covariance_distance_kl_divergence(mat1, mat2):
    """The Kullback Leibler divergence between two Gaussian
    distributions that has mat1 and mat2 as their covariance matrices.

    Assumes both of these distributions has zero mean.

    It is a distance measure, so 0 means equal and then it goes to infinity
    and the matrices become more unequal.

    Parameters
    ----------
    mat1 : np.ndarray of shape (N, N)
        First covariance matrix, should be symmetric and positive definite
    mat2 : np.ndarray of shape (N, N)
        Second covariance matrix, should be symmetric and positive definite

    Returns
    -------
    dist : float
        The distance between the two matrices

    """
    assert mat1.shape == mat2.shape
    assert mat1.shape[0] == mat1.shape[1]
    assert mat1.ndim == 2
    N: int = mat1.shape[0]
    eigvals: Array = jax.scipy.linalg.eigh(mat1, mat2, eigvals_only=True)
    det1: Array = jax.scipy.linalg.det(mat1)
    det2: Array = jax.scipy.linalg.det(mat2)
    common_trace: Array = jnp.sum(eigvals)
    return jnp.real(jnp.sqrt((jnp.log(det2 / det1) + common_trace - N) / 2))


@partial(jax.jit, static_argnames=["rank"])
def frob_gevd_weighted(A, B, rank="full"):
    """The frobenious distance between A and B, weighted by the generalized eigenvectors.

    Defined by $lVert W (A - B) W^H rVert_F$, where $W$ is the matrix of eigenvectors of the generalized eigenvalue decomposition of $A$ and $B$.

    Parameters
    ----------
    A : ndarray of shape (M, M)
        Positive semi-definite matrix
    B : ndarray of shape (M, M)
        Positive definite matrix

    Returns
    -------
    distance : float
        The distance between A and B in weighted Frobenius distance

    Notes
    -----
    The defintion of the eigenvector matrix is in terms of the simultaneous diagonalization of $A$ and $B$.
    $W A W^H = \Sigma$
    $W B W^H = I$

    The distance is equivalent to \lVert \Sigma - I \rVert_F, which is just the sum of the squared
    differences of the eigenvalues from 1.
    """
    eigvals = matop.generalized_eigvalsh(A, B)
    if rank != "full":
        eigvals = jnp.flip(eigvals, axis=-1)[
            :rank
        ]  # only take the rank largest eigenvalues

    return jnp.sqrt(jnp.sum((eigvals - jnp.ones_like(eigvals)) ** 2))


def wishart_log_likelihood(mat_variable, cov, N, regularization=1e6):
    """Wishart log likelihood function.

    It is not a true likelihood, since the mass is not 1. It is however proportional to true likelihood with regards to the covariance matrix. Anything constant with regards to the covariance is not taken into account. Therefore, the maximum likelihood estimator can be found by maximizing this function.

    Parameters
    ----------
    mat_variable : ndarray of shape (M, M)
        positive definite matrix
    cov : ndarray of shape (M, M)
        positive definite matrix
    N : int
        the degree of freedom parameter for the wishart distribution
    regularization : float, optional
        regularization parameter for the covariance matrix. The default is 1e6.
        The matrix is regularized by adding a scaled identity matrix to the covariance matrix,
        such that the condition number becomes at most regularization.

    Returns
    -------
    l : float
        The log likelihood of the data given the covariance matrix
    """
    cov = matop.regularize_matrix_with_condition_number(cov, regularization)
    f1 = -N * jnp.log(jnp.linalg.det(cov))
    f2 = -jnp.trace(jnp.linalg.solve(cov, mat_variable))
    likelihood = f1 + f2
    return jnp.real(likelihood)
