"""Functions to design and compute filters

Can be used to obtain FIR filters from frequency responses. Also contains
helper functions associated with the discrete Fourier transform. 

References
----------
[benestyStudy2013] J. Benesty and J. Chen, Study and design of differential microphone arrays, vol. 6. in Springer Topics in Signal Processing, vol. 6. Springer, 2013.
"""
import numpy as np
import scipy.signal as signal
import itertools as it
import matplotlib.pyplot as plt

import aspcore.fouriertransform as ft


# ============== DESIGN FIR FILTERS ====================
def fir_from_frequency_function(freq_function, ir_len, samplerate, window=None):
    """Constructs a FIR filter from a frequency function

    In order to make the filter casual, the impulse response is centered in the middle, meaning that an
    extra ir_len // 2 samples of delay is added by the filter. To obtain the true response of freq_function, 
    filter with the FIR filter and then shift the signal by ir_len // 2 samples.

    If freq_function is real-valued the filter will be a linear-phase filter with a delay of ir_len // 2 samples.
    
    Parameters
    ----------
    freq_function : function
        a function that takes a 1D array of frequencies as input and returns the complex frequency response
    ir_len : int
        the length of the impulse response. Currently only works for odd lengths.
    samplerate : int
        the sampling rate of the signal
    
    Returns
    -------
    ir : ndarray of shape (ir_len)
    """
    if ir_len % 2 == 0:
       raise NotImplementedError("Function only implemented for odd impulse response lengths (will give incorrect group delay for even lengths)")

    num_freqs = ir_len
    freqs = ft.get_real_freqs(num_freqs, samplerate)
    freq_response = freq_function(freqs)
    freq_response[0] = np.real(freq_response[0])
    if num_freqs % 2 == 0:
        freq_response[-1] = np.real(freq_response[-1])

    ir = ft.irfft(freq_response, n=num_freqs)
    ir = np.real_if_close(ir)

    mid_point = ir_len // 2 + 1
    ir = np.concatenate((ir[...,mid_point:], ir[...,:mid_point]), axis=-1)

    if window == "hamming":
            ir = ir * signal.windows.hamming(ir_len).reshape(
            (1,) * (ir.ndim - 1) + ir.shape[-1:]
            )
    elif window is not None:
        raise NotImplementedError("Only Hamming supported currently.")
    return ir



def fir_from_freqs_window(freq_filter, ir_len, two_sided=True, window="hamming"):
    """Constructs a FIR filter from frequency values
    
    Currently works only for two_sided=True and odd ir_len

    Parameters
    ----------
    freq_filter : ndarray of shape (num_freqs, ...)
        should include both positive and negative frequencies
    ir_len : int
        the length of the impulse response
    two_sided : bool
        if True, the impulse response is assumed to be centered in the middle
    window : str or None
        the window to apply to the impulse response
        
    Returns
    -------
    trunc_filter : ndarray of shape (ir_len, ...)
    trunc_error : float
        the relative truncation error of the impulse response 
    """
    assert ir_len % 1 == 0
    assert freq_filter.shape[0] % 2 == 0
    if two_sided:
        #halfLen = irLen // 2
        mid_point = freq_filter.shape[0] // 2

        time_filter = np.real_if_close(ft.ifft(freq_filter))
        time_filter = np.concatenate((time_filter[...,-mid_point:], time_filter[...,:mid_point]), axis=-1)

        trunc_filter, trunc_error = truncate_filter(time_filter, ir_len, True)


        if window == "hamming":
            trunc_filter = trunc_filter * signal.windows.hamming(ir_len).reshape(
            (1,) * (trunc_filter.ndim - 1) + trunc_filter.shape[-1:]
            )
        elif window is None:
            pass
        else:
            raise ValueError("Invalid value for window argument")
    else:
        raise NotImplementedError

    return trunc_filter, trunc_error

def truncate_filter(ir, ir_len, two_sided):
    """Truncates the impulse response to the desired length
    Currently only works for two_sided=True and odd ir_len

    Parameters
    ----------
    ir : ndarray of shape (..., ir_len_original)
    ir_len : int
        the length of the truncated impulse response
    two_sided : bool
        if True, the impulse response is assumed to be centered in the middle

    Returns
    -------
    trunc_filter : ndarray of shape (..., ir_len)
    """
    if two_sided:
        assert ir_len % 2 == 1
        assert ir.shape[-1] % 2 == 0
        half_len = ir_len // 2
        mid_point = ir.shape[-1] // 2
        trunc_filter = ir[..., mid_point-half_len:mid_point+half_len+1]

        trunc_power = np.sum(ir[...,:mid_point-half_len]**2) + np.sum(ir[...,mid_point+half_len+1:]**2)
        total_power = np.sum(ir**2)
        rel_trunc_error = 10 * np.log10(trunc_power / total_power)
    else:
        raise NotImplementedError
    return trunc_filter, rel_trunc_error



def calc_truncation_error(ir, ir_len, two_sided=True):
    """Calculates the relative truncation error of an impulse response
    The relative error is how much of the power of the impulse response that is lost by truncating.

    Parameters
    ----------
    ir : ndarray of shape (..., ir_len_original)
    ir_len : int
        the length of the truncated impulse response
    two_sided : bool
        if True, the impulse response is assumed to be centered in the middle
    
    Returns
    -------
    rel_trunc_error : float
    """
    if two_sided:
        assert ir_len % 2 == 1
        half_len = ir_len // 2
        mid_point = ir.shape[-1]
        trunc_power = np.sum(ir[...,:mid_point-half_len]**2) + np.sum(ir[...,mid_point+half_len:]**2)
        total_power = np.sum(ir**2)
        rel_trunc_error = 10 * np.log10(trunc_power / total_power)
    else:
        raise NotImplementedError
    return rel_trunc_error
    

def min_truncated_length(ir, two_sided=True, max_rel_trunc_error=1e-3):
    """Calculates the minimum length you can truncate a filter to.
    The minimum length will be calculated independently for all impulse responses, and
    the longest length chosen. The relative error is how much of the
    power of the impulse response that is lost by truncating.
    
    Parameters
    ----------
    ir : ndarray of shape (..., ir_len)
    two_sided : bool
        if True, the impulse response is assumed to be centered in the middle
    max_rel_trunc_error : float
        the maximum relative truncation error allowed

    Returns
    -------
    req_filter_length : int
        the minimum length you can truncate the filter to
    """
    ir_len = ir.shape[-1]
    ir_shape = ir.shape[:-1]

    total_energy = np.sum(ir ** 2, axis=-1)
    energy_needed = (1 - max_rel_trunc_error) * total_energy

    if two_sided:
        center_idx = ir_len // 2
        casual_sum = np.cumsum(ir[..., center_idx:] ** 2, axis=-1)
        noncausal_sum = np.cumsum(np.flip(ir[..., :center_idx] ** 2, axis=-1), axis=-1)
        energy_sum = casual_sum
        energy_sum[..., 1:] += noncausal_sum
    else:
        energy_sum = np.cumsum(ir ** 2, axis=-1)

    enough_energy = energy_sum > energy_needed[..., None]
    trunc_indices = np.zeros(ir_shape, dtype=int)
    for indices in it.product(*[range(dimSize) for dimSize in ir_shape]):
        trunc_indices[indices] = np.min(
            np.nonzero(enough_energy[indices + (slice(None),)])[0]
        )

    req_filter_length = np.max(trunc_indices)
    if two_sided:
        req_filter_length = 2 * req_filter_length + 1
    return req_filter_length




def fractional_delay_filter(delay, order):
    """Returns impulse response of a fractional delay filter using lagrange interpolation

    WARNING: currently untested for delays that are not between 0 and 1.

    The factional delay filter h is non-causal, and adds an integer delay, which is returned as added_integer_delay.
    To get the correct fractional delay after filtering, the signal should be extracted as
    signal[n - delay] = (signal * h)[n - added_integer_delay]

    Parameters
    ----------
    delay : float
        the delay in samples
    order : int
        the interpolation order. 
    
    Returns
    -------
    ir : ndarray of shape (max_filt_len)
        the impulse response of the filter
    added_integer_delay : int
        the integer delay added with the fractional delay filter

    Notes
    -----
    An example that will correctly compensate for the added integer delay:
    import scipy.signal as spsig
    delay_filter, int_delay = fractional_delay_filter(samples_delay, filter_order)
    delayed_cardioid = spsig.fftconvolve(delay_filter[None,:], signal, axes=-1)
    delayed_cardioid = delayed_cardioid[...,int_delay:]
    """
    #ir = np.zeros(max_filt_len)
    frac_dly, dly = np.modf(delay)
    dly = int(dly)

    #order = int(np.min((max_order, 2 * dly, 2 * (max_filt_len - 1 - dly))))

    filt_len = 2*order + 1
    delta = order + frac_dly
    h = _lagrange_interpol(filt_len, delta)

    added_integer_delay = order - dly

    if dly > added_integer_delay:
        raise NotImplementedError("Not correctly implemented for larger delays")

    # diff = delay - delta
    # start_idx = int(np.floor(diff))
    # ir[start_idx : start_idx + order + 1] = h
    return h, added_integer_delay


def _lagrange_interpol(N, delta):
    """Lagrange interpolation
    
    Parameters
    ----------
    N : int
        the order of the interpolation
    delta : float
        the fractional delay
    
    Returns
    -------
    h : ndarray of shape (N+1)
        the impulse response of the filter
    """
    ir_len = int(N + 1)
    h = np.zeros(ir_len)

    for n in range(ir_len):
        k = np.arange(ir_len)
        k = k[np.arange(ir_len) != n]

        h[n] = np.prod((delta - k) / (n - k))
    return h






# ===================== APPLIED FILTERING FUNCTIONS ============================
def filterbank_third_octave(sig, sr, min_freq = 40, plot=False):
    """Filters the provided signal into third-octave bands using butterworth filters
    
    Parameters
    ----------
    sig : ndarray of shape (num_channels, num_samples)
        the signal to filter
    sr : int
        the sampling rate of the signal
    min_freq : int
        the minimum frequency of the lowest band
    plot : bool
        if True, plots the frequency response of the filters

    Returns
    -------
    sig_filtered : ndarray of shape (num_bands, num_channels, num_samples)
        the filtered signal
    freq_lims : ndarray of shape (num_bands, 2)
        the frequency limits of the bands
    
    References
    ----------
    The frequency bands are based on ANSI S1.11: Specification for Octave, Half-Octave, and Third Octave Band Filter Sets
    https://law.resource.org/pub/us/cfr/ibr/002/ansi.s1.11.2004.pdf
    
    """
    max_freq = sr / 2
    ref_freq = 1000 # from ANSI S1.11
    G = 2 # from ANSI S1.11
    b = 3 # third octave band
    num_prototype_bands = 100
    midband_freq = G ** ((np.arange(num_prototype_bands) - 30) / b) * ref_freq
    #assert np.all(midband_freq > min_freq) and np.all(midband_freq < max_freq) 
    assert min_freq < max_freq
    assert min_freq > midband_freq[0], "Gives incorrect results for too small min_freq"

    freq_lims = np.zeros((num_prototype_bands, 2))
    freq_lims[:,0] = G ** (-1 / (2*b)) * midband_freq
    freq_lims[:,1] = G ** (1 / (2*b)) * midband_freq

    freq_idxs = np.logical_and(freq_lims[:,0] > min_freq, freq_lims[:,1] < max_freq)
    freq_lims = freq_lims[freq_idxs,:]

    num_bands = freq_lims.shape[0]
    sos = [signal.butter(N=4, Wn=freq_lims[i,:], btype='bandpass', analog=False, output='sos', fs = sr) for i in range(num_bands)]

    if plot:
        freq_response = [signal.sosfreqz(sos[i]) for i in range(num_bands)]
        fig, ax = plt.subplots(1,1, figsize=(8,6))
        for i in range(num_bands):
            w, h = freq_response[i]
            f = w * sr / (2 * np.pi)
            ax.plot(f, 20 * np.log10(abs(h)), 'b')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Amplitude [dB]')
        ax.set_title('Frequency response of the filter')
        plt.show()

    sig_filtered = np.stack([signal.sosfiltfilt(sos[i], sig) for i in range(num_bands)], axis=0)
    return sig_filtered, freq_lims