


import numpy as np
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