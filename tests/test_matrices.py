import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as splin

import aspcore.matrices as aspmat
import aspcore.matrices_jax as aspmat_jax


def test_blockmat2param_numpy_and_jax_versions_are_equivalent():
    rng = np.random.default_rng()
    num_blocks = 5
    block_len = 3
    mat_size = num_blocks * block_len
    mat = rng.normal(size=(mat_size, mat_size))
    param_numpy = aspmat.blockmat2param(mat, num_blocks, block_len)
    param_jax = aspmat_jax.blockmat2param(mat, num_blocks, block_len)

    assert np.allclose(param_numpy, np.array(param_jax))


def test_param2blockmat_numpy_and_jax_versions_are_equivalent():
    rng = np.random.default_rng()
    num_blocks = 5
    block_len = 3
    param = rng.normal(size=(num_blocks, num_blocks, block_len, block_len))
    mat_numpy = aspmat.param2blockmat(param)
    mat_jax = aspmat_jax.param2blockmat(param)

    assert np.allclose(mat_numpy, np.array(mat_jax))


def test_matmul_toeplitz_real_tuple_equivalent_to_scipy():
    rng = np.random.default_rng(0)
    num_rows = 9
    num_cols = 6
    num_rhs = 4

    c = rng.normal(size=num_rows)
    r = rng.normal(size=num_cols)
    x = rng.normal(size=(num_cols, num_rhs))

    y_scipy = splin.matmul_toeplitz((c, r), x)
    y_jax = aspmat_jax.matmul_toeplitz((jnp.asarray(c), jnp.asarray(r)), jnp.asarray(x))

    assert np.allclose(np.asarray(y_jax), y_scipy, atol=1e-6, rtol=1e-6)


def test_matmul_toeplitz_complex_implicit_row_equivalent_to_scipy():
    rng = np.random.default_rng(1)
    mat_dim = 8

    c = rng.normal(size=mat_dim) + 1j * rng.normal(size=mat_dim)
    x = rng.normal(size=mat_dim) + 1j * rng.normal(size=mat_dim)

    y_scipy = splin.matmul_toeplitz(c, x)
    y_jax = aspmat_jax.matmul_toeplitz(jnp.asarray(c), jnp.asarray(x))

    assert np.allclose(np.asarray(y_jax), y_scipy, atol=1e-6, rtol=1e-6)


def test_matmul_toeplitz_jitted_equivalent_to_scipy():
    rng = np.random.default_rng(2)
    num_rows = 10
    num_cols = 7
    num_rhs = 3

    c = rng.normal(size=num_rows)
    r = rng.normal(size=num_cols)
    x = rng.normal(size=(num_cols, num_rhs))

    jitted_matmul_toeplitz = jax.jit(aspmat_jax.matmul_toeplitz)
    y_scipy = splin.matmul_toeplitz((c, r), x)
    y_jitted = jitted_matmul_toeplitz((jnp.asarray(c), jnp.asarray(r)), jnp.asarray(x))

    assert np.allclose(np.asarray(y_jitted), y_scipy, atol=1e-6, rtol=1e-6)


def test_matmul_toeplitz_is_faster_than_standard_matmul_for_large_matrices():
    rng = np.random.default_rng(3)
    num_rows = 3000
    num_cols = 3000
    num_rhs = 1

    c = rng.normal(size=num_rows)
    r = rng.normal(size=num_cols)
    x = rng.normal(size=(num_cols, num_rhs))

    # Time the standard matrix multiplication

    start_time = time.time()
    A = splin.toeplitz(c, r)
    y_standard = A @ x
    end_time = time.time()
    scipy_time = end_time - start_time

    @jax.jit
    def jax_standard_matmul(c, r, x):
        A = jax.scipy.linalg.toeplitz(c, r)
        return A @ x

    # compile the function
    y_jax = jax.block_until_ready(
        jax_standard_matmul(jnp.asarray(c), jnp.asarray(r), jnp.asarray(x))
    )

    start_time = time.time()
    y_jax = jax.block_until_ready(
        jax_standard_matmul(jnp.asarray(c), jnp.asarray(r), jnp.asarray(x))
    )
    end_time = time.time()
    jax_time = end_time - start_time

    # compile the function
    y_toeplitz = jax.block_until_ready(
        aspmat_jax.matmul_toeplitz((jnp.asarray(c), jnp.asarray(r)), jnp.asarray(x))
    )

    # Time the matmul_toeplitz function
    start_time = time.time()
    y_toeplitz = jax.block_until_ready(
        aspmat_jax.matmul_toeplitz((jnp.asarray(c), jnp.asarray(r)), jnp.asarray(x))
    )
    end_time = time.time()
    toeplitz_matmul_time = end_time - start_time

    print(f"Scipy matmul time: {scipy_time:.6f} seconds")
    print(f"JAX standard matmul time: {jax_time:.6f} seconds")
    print(f"Matmul_toeplitz time: {toeplitz_matmul_time:.6f} seconds")

    assert toeplitz_matmul_time < scipy_time, (
        "matmul_toeplitz should be faster than standard matmul for large matrices"
    )
    assert toeplitz_matmul_time < jax_time, (
        "matmul_toeplitz should be faster than JAX standard matmul for large matrices"
    )
