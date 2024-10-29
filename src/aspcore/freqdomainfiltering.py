"""Methods for linear convolution and correlation in the frequency domain

The methods can be helpful for filtering using frequency-domain filters
when multiple channels are involved. When possible, for correlation and convolution 
the functions will zero-pad the signals correctly in order to avoid circular convolution. 

In particular, the functions can be useful frequency domain adaptive filters, where overlap-save
is used for both linear convolutions and correlations. It can be tricky there to get 
the zero-padding right. 
"""
import numpy as np


def fft_transpose(time_sig, n=None, add_empty_dim=False):
    """Applies the FFT to the last axis, and moves it to the first axis
    
    Parameters
    ----------
    time_sig : ndarray
        The signal to be transformed. The last axis should correspond to time
    n : int, optional
        length of the FFT. If None, the length is the length of the last axis of time_sig
    add_empty_dim : bool, optional
        If True, an empty dimension is added as the last axis
        Useful if the return value should be considered a vector
    
    Returns
    -------
    freq_signal : ndarray
        The transformed signal
    """
    freq_signal = np.fft.fft(time_sig, n=n, axis=-1)

    new_axis_order = np.concatenate(
        ([time_sig.ndim - 1], np.arange(time_sig.ndim - 1))
    )
    freq_signal = np.transpose(freq_signal, new_axis_order)
    if add_empty_dim:
        freq_signal = np.expand_dims(freq_signal, -1)
    return freq_signal


def ifft_transpose(freq_signal, remove_empty_dim=False):
    """Inverse FFT, and moves the first axis to the last axis

    Parameters
    ----------
    freq_signal : ndarray
        The signal to be transformed. The first axis should 
        correspond to frequency
    remove_empty_dim : bool, optional
        If True, the last axis is removed if it is empty
        Will raise an error if the last axis is not empty and 
        remove_empty_dim is set to True. Should be set to True if used with
        fft_transpose with add_empty_dim=True. 
    
    Returns
    -------
    time_signal : ndarray
        The transformed signal
    """
    if remove_empty_dim:
        freq_signal = np.squeeze(freq_signal, axis=-1)
    time_signal = np.fft.ifft(freq_signal, axis=0)
    new_axis_order = np.concatenate((np.arange(1, freq_signal.ndim), [0]))
    time_signal = np.transpose(time_signal, new_axis_order)
    return time_signal


def correlate_sum_tt(time_filter, time_signal):
    """Computes linear correlation between two time-domain signals

    The next to last dimension is summed over. 
    Pads the filter before the signal, which ensures no artifacts due
    to circular convolution. 
    
    Parameters
    ----------
    time_filter : ndarray
        The filter. The last axis should correspond to time
    time_signal : ndarray
        The signal. The last axis should correspond to time
        Should be exactly twice as long as the filter
    
    Returns
    -------
    correlation : ndarray
        The linear correlation between the filter and the signal
    """
    assert 2 * time_filter.shape[-1] == time_signal.shape[-1]
    # assert(timeFilter.shape[-2] == timeSignal.shape[-2])
    freq_filter = fft_transpose(
        np.concatenate((np.zeros_like(time_filter), time_filter), axis=-1),
        add_empty_dim=False,
    )
    return correlate_sum_ft(freq_filter, time_signal)


def correlate_sum_ft(freq_filter, time_signal):
    """Computes linear correlation between a time-domain signal and a frequency-domain filter
    
    Last dimension of the signal is summed over.
    Next to last dimension of the filter is summed over.

    Parameters
    ----------
    freq_filter : ndarray
        The filter. The first axis should correspond to frequency.
        Before it was transformed, it should have been padded with zeros
        equal to the length of the impulse response before the impulse response. 
    time_signal : ndarray
        The signal. The last axis should correspond to time. 
        Should be exactly twice as long as the filter.
    
    Returns
    -------
    correlation : ndarray
        The linear correlation between the filter and the signal
    """
    freq_signal = fft_transpose(time_signal, add_empty_dim=False)
    return correlate_sum_ff(freq_filter, freq_signal)

def correlate_sum_tf(time_filter, freq_signal):
    """Computes linear correlation between a frequency-domain signal and a time-domain filter
    
    Last dimension of the filter is summed over.
    Next to last dimension of the signal is summed over.

    Parameters
    ----------
    time_filter : ndarray
        The filter. The last axis should correspond to time.
    freq_signal : ndarray
        The signal. The first axis should correspond to frequency.
        Before being transformed into the frequency domain, it should 
        have been twice as long as the filter
        
    Returns
    -------
    correlation : ndarray
        The linear correlation between the filter and the signal
    """
    freq_filter = fft_transpose(
        np.concatenate((np.zeros_like(time_filter), time_filter), axis=-1),
        add_empty_dim=False,
    )
    return correlate_sum_ff(freq_filter, freq_signal)


def correlate_sum_ff(freq_filter, freq_signal):
    """Computes linear correlation between two frequency-domain signals
    
    The last dimension is summed over. 

    Parameters
    ----------
    freq_filter : ndarray
        The filter. The first axis should correspond to frequency.
        Before it was transformed, it should have been padded with zeros
        equal to the length of the impulse response before the impulse response.
    freq_signal : ndarray
        The signal. The first axis should correspond to frequency.
        Before being transformed into the frequency domain, it should 
        have been twice as long as the impulse response of the filter.
    
    Returns
    -------
    correlation : ndarray   
        The linear correlation between the filter and the signal
    """
    assert freq_filter.shape[0] == freq_signal.shape[0]
    assert freq_filter.shape[0] % 2 == 0
    output_len = freq_filter.shape[0] // 2
    filtered_signal = ifft_transpose(np.sum(freq_filter * freq_signal.conj(), axis=-1))
    return np.real_if_close(filtered_signal[..., :output_len])


def correlate_cartesian_tt(time_filter, time_signal):
    """Computes the linear correlation between two time-domain signals

    Computes the correlation for all combinations of channels, as in
    the cartesian product of the channels.
    
    Parameters
    ----------
    time_filter : ndarray
        The filter. The last axis should correspond to time
    time_signal : ndarray
        The signal. The last axis should correspond to time
        Should be exactly twice as long as the filter. It should be 
        twice as long as the impulse response of the filter. 
    
    Returns
    -------
    correlation : ndarray of shape (filter.shape[:-1], signal.shape[:-1], num_samples)
        The linear correlation between the filter and the signal
    """
    assert 2 * time_filter.shape[-1] == time_signal.shape[-1]
    freq_filter = fft_transpose(
        np.concatenate((np.zeros_like(time_filter), time_filter), axis=-1),
        add_empty_dim=False,
    )
    return correlate_euclidian_ft(freq_filter, time_signal)


def correlate_euclidian_ft(freq_filter, time_signal):
    """Correlates every channel of input with every channel of the filter

    Parameters
    ----------
    freq_filter : ndarray
        The filter. The first axis should correspond to frequency.
        Before it was transformed, it should have been padded with zeros
        equal to the length of the impulse response *before* the impulse response.
    time_signal : ndarray
        The signal. The last axis should correspond to time.
        Should be exactly twice as long as the impulse response of the filter.

    Returns
    -------
    filtered_signal : ndarray of shape (filter.shape[:-1], signal.shape[:-1], num_samples)
        The linear correlation between the filter and the signal
    """
    freq_signal = fft_transpose(time_signal, add_empty_dim=False)
    return correlate_euclidian_ff(freq_filter, freq_signal)

def correlate_euclidian_tf(time_filter, freq_signal):
    """Correlates every channel of input with every channel of the filter
    
    Parameters
    ----------
    time_filter : ndarray of shape (f_1, f_2, ..., ir_len)
        The filter. The last axis should correspond to time.
    freq_signal : ndarray of shape (num_freq, s_1, s_2, ..., s_n)
        The signal. The first axis should correspond to frequency.
        Before being transformed into the frequency domain, it should
        have been twice as long as the impulse response of the filter.
    
    Returns
    -------
    filtered_signal : ndarray of shape (filter.shape[:-1], signal.shape[:-1], num_samples)
        The linear correlation between the filter and the signal
    """
    freq_filter = fft_transpose(
        np.concatenate((np.zeros_like(time_filter), time_filter), axis=-1),
        add_empty_dim=False,
    )
    return correlate_euclidian_ff(freq_filter, freq_signal)


def correlate_euclidian_ff(freq_filter, freq_signal):
    """Correlates every channel of input with every channel of the filter

    Parameters
    ----------
    freq_filter : ndarray of shape (num_freq, f_1, f_2, ..., f_n)
        The filter. The first axis should correspond to frequency.
        Before it was transformed, it should have been padded with zeros
        equal to the length of the impulse response *before* the impulse response.
    freq_signal : ndarray of shape (num_freq, s_1, s_2, ..., s_n)
        The signal. The first axis should correspond to frequency.
        Before being transformed into the frequency domain, it should
        have been twice as long as the impulse response of the filter.
    
    Returns
    -------
    filtered_signal : ndarray of shape (filter.shape[:-1], signal.shape[:-1], num_samples)
        The linear correlation between the filter and the signal
    """
    assert freq_filter.shape[0] % 2 == 0
    output_len = freq_filter.shape[0] // 2

    filtered_signal = (
        freq_filter.reshape(freq_filter.shape + (1,) * (freq_signal.ndim - 1))
        * freq_signal.reshape(
            freq_signal.shape[0:1] + (1,) * (freq_filter.ndim - 1) + freq_signal.shape[1:]
        ).conj()
    )
    filtered_signal = ifft_transpose(filtered_signal)
    return np.real_if_close(filtered_signal[..., :output_len])



def convolve_sum(freq_filter, time_signal):
    """Performs linear convolution between a time-domain signal and a frequency-domain filter
    
    The last dimension of the filter is summed over.
    The next to last dimension of the signal is summed over.

    Parameters
    ----------
    freq_filter : ndarray
        The filter. The first axis should correspond to frequency.
        Before it was transformed, it should have been padded with zeros
        equal to the length of the impulse response *after* the impulse response.
    time_signal : ndarray
        The signal. The last axis should correspond to time.
        Should be exactly twice as long as the impulse response of the filter.
    
    Returns
    -------
    filtered_signal : ndarray
        The signal filtered through the frequency domain filter
    """
    assert freq_filter.shape[-1] == time_signal.shape[-2]
    assert freq_filter.shape[0] == time_signal.shape[-1]
    assert freq_filter.shape[0] % 2 == 0
    output_len = freq_filter.shape[0] // 2

    filtered_signal = freq_filter @ fft_transpose(time_signal, add_empty_dim=True)
    filtered_signal = ifft_transpose(filtered_signal, remove_empty_dim=True)
    return np.real_if_close(filtered_signal[..., output_len:])


def convolve_euclidian_ff(freq_filter, freq_signal):
    """Convolves every channel of input with every channel of the filter
    
    Parameters
    ----------
    freq_filter : ndarray of shape (num_freq, f_1, f_2, ..., f_n)
        The filter. The first axis should correspond to frequency.
        Before it was transformed, it should have been padded with zeros
        equal to the length of the impulse response *after* the impulse response.
    freq_signal : ndarray of shape (num_freq, s_1, s_2, ..., s_n)
        The signal. The first axis should correspond to frequency.
        Before being transformed into the frequency domain, it should
        have been twice as long as the impulse response of the filter.

    Returns
    -------
    filtered_signal : ndarray of shape (filter.shape[:-1], signal.shape[:-1], num_samples)
        The signal filtered through the frequency domain filter
    """
    assert freq_filter.shape[0] % 2 == 0
    output_len = freq_filter.shape[0] // 2

    filtered_signal = freq_filter.reshape(
        freq_filter.shape + (1,) * (freq_signal.ndim - 1)
    ) * freq_signal.reshape(
        freq_signal.shape[0:1] + (1,) * (freq_filter.ndim - 1) + freq_signal.shape[1:]
    )
    filtered_signal = ifft_transpose(filtered_signal)
    return np.real_if_close(filtered_signal[..., output_len:])


def convolve_euclidian_ft(freq_filter, time_signal):
    """Convolves every channel of input with every channel of the filter
    
    Parameters
    ----------
    freq_filter : ndarray of shape (num_freq, f_1, f_2, ..., f_n)
        The filter. The first axis should correspond to frequency.
        Before it was transformed, it should have been padded with zeros
        equal to the length of the impulse response *after* the impulse response.
    time_signal : ndarray of shape (s_1, s_2, ..., s_n, num_samples)
        The signal. The last axis should correspond to time.
        Should be exactly twice as long as the impulse response of the filter.

    Returns
    -------
    filtered_signal : ndarray of shape (filter.shape[:-1], signal.shape[:-1], num_samples)
        The signal filtered through the frequency domain filter
    
    """
    assert freq_filter.shape[0] == time_signal.shape[-1]

    freq_signal = fft_transpose(time_signal, add_empty_dim=False)
    return convolve_euclidian_ff(freq_filter, freq_signal)
