"""Functions associated with perfect periodic sequences

Such sequences are deterministic periodic sequences with an impulse as periodic autocorrelation. When used for system identification, which in an audio context often is room impulse response estimation, the deconvolution is particularly simple. For an ideal LTI system without noise, the impulse response can be identified perfectly after only one period of the signal, by cross-correlating the output signal with the perfect sequence. In addition, it is possible to identify the impulse response of a MISO system by using a shifted version of the same sequence for each source.


References
----------
[antweilerNLMStype2008] C. Antweiler, A. Telle, and P. Vary, “NLMS-type system identification of MISO systems with shifted perfect sequences,” Proceedings of the International Workshop on Acoustic Echo and Noise Control (IWAENC), Seattle, WA, Sep. 2008. \n
[antweilerSystem2014] C. Antweiler, S. Kuehl, B. Sauert, and P. Vary, “System identification with perfect sequence excitation - efficient NLMS vs. inverse cyclic convolution,” in Speech Communication; 11. ITG Symposium, Sep. 2014, pp. 1–4.\n
[hahnSimultaneous2018] N. Hahn and S. Spors, “Simultaneous measurement of spatial room impulse responses from multiple sound sources using a continuously moving microphone,” in 2018 26th European Signal Processing Conference (EUSIPCO), Sep. 2018, pp. 2180–2184. doi: 10.23919/EUSIPCO.2018.8553532. `[link] <https://doi.org/10.23919/EUSIPCO.2018.8553532>`__ \n
"""

import jax.numpy as jnp

import aspcore.matrices_jax as aspmat


def decorrelate(sig, pseq, v=0):
    """Can be used to identify a LTI system from a PSEQ input signal

    Parameters
    ----------
    sig : ndarray of shape (num_channels, num_samples)
        the signal that has been convolved with the system
    pseq : ndarray of shape (1, num_samples,)
        the perfect sequence that was used as input to the system
    v : int
        index between 0 and PSEQ period length
        declares which index should be considered the start of the sequence

    Returns
    -------
    ir : ndarray of shape (num_channels, num_samples)
        the estimated impulse response of the system

    """
    assert sig.ndim == 2
    assert sig.shape[-1] == pseq.shape[-1]
    if pseq.ndim == 2:
        pseq = jnp.squeeze(pseq, axis=0)

    normalize_factor = jnp.sum(pseq**2)
    p_n = jnp.flip(jnp.roll(pseq, -1 - v))
    p_n = p_n / normalize_factor
    pn_rev = jnp.concatenate((jnp.array([0]), jnp.flip(p_n[1:])))

    rir_est = aspmat.matmul_toeplitz((p_n, pn_rev), sig.T).T

    # system_mat = jax.scipy.linalg.toeplitz(p_n, pn_rev)
    # rir_est = (system_mat @ sig.T).T

    # rir_est = splin.matmul_toeplitz((p_n, pn_rev), sig.T)
    return rir_est
