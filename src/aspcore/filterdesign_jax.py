"""Functions to design and compute filters using JAX

"""
import jax.numpy as jnp

def frac_dly_windowed_sinc(frac_delay, filt_len, window_type='hamming'):
    """Fractional delay filter using windowed sinc method

    The returned filter has a delay of frac_delay, plus a fixed integer delay of filt_len // 2.

    Parameters
    ----------
    frac_delay : float
        the fractional delay in samples
    filt_len : int
        the length of the filter (should be odd)
    window_type : str
        the type of window to use. Options are 'hamming', 'hann', 'blackman'

    Returns
    -------
    h : ndarray of shape (filt_len,)
        the impulse response of the filter
    """
    if filt_len % 2 == 0:
        raise ValueError("Filter length should be odd")
    #assert frac_delay >= 0.0 and frac_delay < filt_len, "frac_delay should be in [0, filt_len)"

    n = jnp.arange(filt_len)
    mid = filt_len // 2
    h = jnp.sinc(n - mid - frac_delay)

    if window_type == 'hamming':
        window = jnp.hamming(filt_len)
    elif window_type == 'hann':
        window = jnp.hanning(filt_len)
    elif window_type == 'blackman':
        window = jnp.blackman(filt_len)
    else:
        raise ValueError(f"Unknown window type: {window_type}")

    h = h * window
    h = h / jnp.sum(h)
    return h


# if __name__ == "__main__":
#     frac_delay = 6.8
#     order = 64
#     h = frac_dly_windowed_sinc(frac_delay, 2*order + 1, window_type='hamming')
#     center = order
#     import matplotlib.pyplot as plt
#     plt.plot(h)
#     plt.plot(center + frac_delay, 0, 'ro')
#     #plt.show()

#     import scipy.signal as signal
#     w, gp = signal.group_delay((h, 1))
#     plt.figure()
#     plt.plot(w, gp)
#     plt.show()