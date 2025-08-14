
import numpy as np
import aspcore.fouriertransform as ft_numpy
import aspcore.fouriertransform_jax as ft_jax

import matplotlib.pyplot as plt


def test_rdft_weighting_is_identical_for_numpy_and_jax():
    rng = np.random.default_rng()
    vec_len = rng.integers(1, 100)
    num_real_freqs = vec_len // 2 + 1

    weights_numpy = ft_numpy.rdft_weighting(num_real_freqs, vec_len)
    weights_jax = ft_jax.rdft_weighting(vec_len)

    plt.plot(weights_numpy, label='Numpy Weights')
    plt.plot(weights_jax, label='Jax Weights')
    plt.legend()
    plt.show()

    assert np.allclose(weights_numpy, weights_jax)


def test_real_vec_to_dft_domain_is_identical_for_numpy_and_jax():
    rng = np.random.default_rng()
    vec_len = rng.integers(1, 100)
    num_real_freqs = vec_len // 2 + 1

    signal = rng.normal(size=(vec_len,))
    dft_vec_numpy = ft_numpy.real_vec_to_dft_domain(signal, scale = True)
    dft_vec_jax = ft_jax.real_vec_to_dft_domain(signal, scale = True)
    assert np.allclose(dft_vec_numpy, dft_vec_jax)

def test_dft_domain_to_real_vec_is_identical_for_numpy_and_jax():
    rng = np.random.default_rng()
    vec_len = rng.integers(1, 100)
    num_real_freqs = vec_len // 2 + 1

    signal = rng.normal(size=(vec_len,)) + 1j * rng.normal(size=(vec_len,))
    signal[0] = np.real(signal[0])  # Ensure the first element is real
    even = vec_len % 2 == 0
    if even:
        signal[-1] = np.real(signal[-1])

    dft_vec_numpy = ft_numpy.dft_domain_to_real_vec(signal, scale = True, even = even)
    dft_vec_jax = ft_jax.dft_domain_to_real_vec(signal, scale = True, even = even)

    assert np.allclose(dft_vec_numpy, dft_vec_jax)
