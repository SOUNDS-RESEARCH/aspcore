"""Routines for solving linear systems with various penalties



"""
import jax 
import numpy as np
import jax.numpy as jnp


def find_best_reg_param_gcv(A, y, reg_params):
    """Finds the best regularization parameter using GCV

    Parameters
    ----------
    A : array of shape (m x n)
        the system matrix
    y : array of shape (m,)
        the data vector
    reg_params : array of shape (k,)
        candidate regularization parameters
    
    Returns
    -------
    best_reg_param : float
        the regularization parameter with the lowest GCV score
    scores : array of shape (k,)
        the GCV scores for each candidate regularization parameter
    """
    scores = jnp.array([gcv_score(A, y, rp) for rp in reg_params])
    best_idx = jnp.argmin(scores)
    best_reg_param = reg_params[best_idx]
    return best_reg_param, scores

def gcv_score(A, y, reg_param):
    """Computes the score for the generalized cross validation method
    
    For a quadratic problem like ||Ax - y||_2^2 + reg_param * ||x||_2^2, with optimal
    solution x_reg = (A^H A + reg_param I)^{-1} A^H 

    smoothing matrix is S = A (A^H A + reg_param I)^{-1} A^H

    Parameters
    ----------
    A : array of shape (m x n)
        the system matrix
    y : array of shape (m,)
        the data vector
    reg_param : float
        regularization parameter
    
    Returns
    -------
    score : float
        the GCV score
    """
    m = A.shape[0]
    n = A.shape[1]

    system_mat = A.conj().T @ A + reg_param * jnp.eye(A.shape[1], dtype=A.dtype)
    S = A @ jax.scipy.linalg.solve(system_mat, A.conj().T, assume_a="pos")

    numerator = jnp.linalg.norm((jnp.eye(m) - S) @ y)**2 / m
    denominator = jnp.abs((jnp.trace(jnp.eye(m) - S) / m))**2
    score = numerator / denominator
    return score


def lsq_with_l2_regularization(A, y, lamb=1e-10):
    """Solves a linear least squares problem with L2 regularization

    Parameters
    ----------
    A : ndarray of shape (m, n)
        matrix in the least squares problem
    y : ndarray of shape (m)
        vector in the least squares problem
    
    Returns
    -------
    x : ndarray of shape (n)
        solution to the least squares problem   
    """
    system_mat = A.conj().T @ A + lamb * jnp.eye(A.shape[1], dtype=A.dtype)
    rhs = A.conj().T @ y
    return jax.scipy.linalg.solve(system_mat, rhs, assume_a="pos")



def irls(A, y, reg_param=0):
    """Solves a linear system of form Ax = y with l1 penalty

    A is m x N where m is smaller. Therefore A^{-1}y is a nontrivial set of vectors. 
    We find x in A^{-1}y such that x has minimal l1 norm. 

    Parameters
    ----------
    A : array of shape (m x N)
        the system matrix
    y : array of shape (m,)
        the data vector
    
    Returns
    -------
    x : array of shape (N,)
        the result of the algorithm
        
    References
    ----------
    Iteratively reweighted least squares minimization for sparse recovery
    INGRID DAUBECHIES, RONALD DEVORE, MASSIMO FORNASIER, C. S˙INAN GÜNTÜRK
    """

    MAXITER = 16
    N = A.shape[-1]
    m = A.shape[0]

    #suggested default values
    K = N // 2
    gamma = 0.9

    def irls_update(w, eps):
        D = jnp.diag(1/w)
        system_mat = A @ D @ A.conj().T 
        system_mat = system_mat + reg_param * jnp.eye(m, dtype=system_mat.dtype)
        v = jax.scipy.linalg.solve(system_mat, y)
        x = D @ A.conj().T @ v #x_{n+1}

        r_K = jnp.sort(jnp.abs(x))[-K]
        eps = jnp.minimum(eps, r_K / N)

        w = 1 / jnp.sqrt((jnp.abs(x)**2 + eps**2))
        return x, w, eps

    w = jnp.ones(N)
    eps = 1

    l1_norm = []
    residual = []

    for i in range(MAXITER):
        x, w, eps = irls_update(w, eps)

        #diagnostics
        l1_norm.append(jnp.sum(jnp.abs(x)))
        residual.append(jnp.linalg.norm(A @ x - y))

    l1_norm = jnp.array(l1_norm)
    residual = jnp.array(residual)

    return x, l1_norm, residual



def irls_reg(A, y, snr=1, max_iter = 16):
    """Solves a linear system of form Ax = y with l1 penalty

    A is m x N where m is smaller. Therefore A^{-1}y is a nontrivial set of vectors. 
    We find x in A^{-1}y such that x has minimal l1 norm. 

    Can have high compilation time for large max_iter, as it uses a native python loop. If 
    this is a problem, the method should be reimplemented using jax.lax.scan. 

    Parameters
    ----------
    A : array of shape (m x N)
        the system matrix
    y : array of shape (m,)
        the data vector
    snr : float
        signal-to-noise ratio
    
    Returns
    -------
    x : array of shape (N,)
        the result of the algorithm
        
    References
    ----------
    [1] V. Pulkki, S. Delikaris-Manias, and A. Politis, Parametric time-frequency domain spatial audio. 2017. doi: 10.1002/9781119252634. Chapter 3. 
    """
    N = A.shape[-1] #number of plane waves
    m = A.shape[0] #number of measurements

    #suggested default value
    K = N // 2

    def irls_update(w, eps):
        D = jnp.diag(1/w)
        system_mat = A @ D @ A.conj().T

        signal_power_estimate = jnp.trace(system_mat) / m
        #print(f"Signal power estimate: {signal_power_estimate}")
        reg_param = jnp.abs(signal_power_estimate / snr) # should be positive real valued
        system_mat = system_mat + reg_param * jnp.eye(m, dtype=system_mat.dtype)

        v = jax.scipy.linalg.solve(system_mat, y)
        x = D @ A.conj().T @ v #x_{n+1}

        r_K = jnp.sort(jnp.abs(x))[-K]
        eps = jnp.minimum(eps, r_K / N)

        w = 1 / jnp.sqrt((jnp.abs(x)**2 + eps**2))
        return x, w, eps

    w = jnp.ones(N)
    eps = 1

    l1_norm = []
    residual = []

    for i in range(max_iter):
        x, w, eps = irls_update(w, eps)

        #diagnostics
        l1_norm.append(jnp.sum(jnp.abs(x)))
        residual.append(jnp.linalg.norm(A @ x - y))

    l1_norm = jnp.array(l1_norm)
    residual = jnp.array(residual)

    return x, l1_norm, residual