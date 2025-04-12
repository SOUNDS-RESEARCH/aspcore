"""Methods for discrete Fourier transform

The forward transforms will operate on the last axis, and put the resulting frequency axis first. The inverse transforms will operate on the first axis and put the resulting time axis last. 
A time domain sample can be accessed as time_signal[...,n]
A frequency domain sample can be accessed as freq_signal[f,...]

The motivation is that the methods are build for multichannel audio processing, where the assumption is that only 1-D FFTs are needed. The most common use case is that time domain processing operates directly on the last axis, and frequency domain processing operates on the channel axes, but broadcasts over the frequency axis. This behaviour for the transforms results in much fewer transpositions, and makes the code more readable.

Important
---------
Note the time convention, which is not the same as in the numpy.fft module. This is to achieve consistency with 
the acoustics literature. The definition of the DFT is
u(k) = \sum_{n=0}^{N-1} u[n] e^{i 2pi k n / N}, for frequency bins k = 0, 1, ..., N-1,
and the definition of the Inverse DFT is
u[n] = \frac{1}{N} \sum_{k=0}^{N-1} u(k) e^{-i 2pi k n / N}.

It can be viewed as a direct consequence of choosing the definition of a plane wave propagating in the d direction to be
u(r) = e^{i k r^T d}, where k is the wavenumber.

References
----------

"""
import numpy as np
import scipy.linalg as splin

def fft(time_sig, n=None):
    """Computes the FFT

    Computes along the last axis, and moves the resulting frequency axis to the first axis. 
    Uses the opposite time convention as numpy.fft.
    
    Parameters
    ----------
    time_sig : ndarray of shape (..., num_samples)
        The signal to be transformed. The last axis should correspond to time
    n : int, optional
        length of the FFT. If None, the length is the length of the last axis of time_sig
    
    Returns
    -------
    freq_signal : ndarray (num_freqs, ...)
        The transformed signal. 
    """
    if n is None:
        n = time_sig.shape[-1]
    freq_signal = np.fft.ifft(time_sig, n=n, axis=-1) * n
    freq_signal = np.moveaxis(freq_signal, -1, 0)
    return freq_signal

def ifft(freq_signal):
    """Computes the inverse FFT

    Computes along the first axis, and moves the resulting time axis to the last axis. 
    Uses the opposite time convention as numpy.fft.

    Parameters
    ----------
    freq_signal : ndarray of shape (num_freqs, ...)
        The signal to be transformed. The first axis should 
        correspond to frequency

    Returns
    -------
    time_signal : ndarray of shape (..., num_samples)
        The transformed signal
    """
    n = freq_signal.shape[0]
    time_signal = np.fft.fft(freq_signal, axis=0) / n
    time_signal = np.moveaxis(time_signal, 0, -1)
    return time_signal

def rfft(time_sig, n=None, num_freqs_removed_low = 0):
    """Computes the real FFT
    
    Parameters
    ----------
    time_sig : ndarray
        The signal to be transformed. The last axis should correspond to time
    n : int, optional
        length of the FFT. If None, the length is the length of the last axis of time_sig
    num_freqs_removed_low : int, optional
        The number of frequencies to remove from the low end of the spectrum. Default
        option is 0, which corresponds to the full real DFT. 
    
    Returns
    -------
    freq_signal : ndarray
        The transformed signal

    Notes
    -----
    Corresponds to numpy.fft.rfft, but with the time convention used in this package. 
    """
    if n is None:
        n = time_sig.shape[-1]
    freq_signal = np.fft.ifft(time_sig, n=n, axis=-1) * n
    freq_signal = np.moveaxis(freq_signal, -1, 0)

    num_real_freqs = n // 2 + 1
    freq_signal = freq_signal[:num_real_freqs,...]

    if num_freqs_removed_low > 0:
        freq_signal = freq_signal[num_freqs_removed_low:,...]
    return freq_signal

def irfft(freq_signal, n=None, num_freqs_removed_low = 0):
    """Inverse FFT, and moves the first axis to the last axis

    Parameters
    ----------
    freq_signal : ndarray
        The signal to be transformed. The first axis should 
        correspond to frequency
    n : int, optional
        length of the FFT. This is the length of the resulting time domain signal, not the frequency domain input.
        If not supplied, it is assumed to be 2 * (freq_signal.shape[0] - 1), corresponding to the 
        output of rfft without argument. To get an odd output length, you need to supply n.
    num_freqs_removed_low : int, optional
        The number of frequencies that were removed from the low end of the spectrum. 
        Default option is 0, which corresponds to the full real DFT. The low frequencies are filled in
        with zeros. 

    Returns
    -------
    time_signal : ndarray
        The transformed signal
    """
    if num_freqs_removed_low > 0:
        padding_shape = list(freq_signal.shape)
        padding_shape[0] = num_freqs_removed_low
        freq_signal = np.concatenate((np.zeros(padding_shape), freq_signal), axis=0)

    if n is None:
        freq_signal = insert_negative_frequencies(freq_signal, even = True)
        #freq_signal = np.concatenate((freq_signal, np.flip(freq_signal[1:-1, ...].conj(), axis=0)), axis=0)
        n = freq_signal.shape[0]
    else:
        if n == 2 * (freq_signal.shape[0] - 1):
            freq_signal = insert_negative_frequencies(freq_signal, even = True)
            n = freq_signal.shape[0]
        elif n == 2 * (freq_signal.shape[0] - 1) + 1:
            freq_signal = insert_negative_frequencies(freq_signal, even = False)
            n = freq_signal.shape[0]
        else:  
            raise NotImplementedError("irfft with arbitrary output length not implemented")
    time_signal = np.fft.fft(freq_signal, axis=0) / n
    time_signal = np.moveaxis(time_signal, 0, -1)
    return np.real_if_close(time_signal)

def insert_negative_frequencies(freq_signal, even):
    """Inserts the values associated with negative frequencies.
    
    Requires the assumption of conjugate symmetry, i.e. the original signal was real. 
    Can be used in conjunction with get_real_freqs
    
    Parameters
    ----------
    freq_signal : ndarray of shape (num_real_freq, ...)
    even : bool
        if True, the full dft length was even, if False it was odd.

    Returns
    -------
    freq_signal_full : ndarray of shape (num_freq, ...)
        num_freq is even if even=True, and odd if even=False
    
    """
    if even:
        num_real_freqs = freq_signal.shape[0]
        all_freq_values = np.concatenate((freq_signal, np.flip(freq_signal[1:-1, ...].conj(), axis=0)), axis=0)
        all_freq_values[0, ...] = np.real(all_freq_values[0, ...])
        all_freq_values[num_real_freqs-1, ...] = np.real(all_freq_values[num_real_freqs-1, ...])
        return all_freq_values
    else:
        num_real_freqs = freq_signal.shape[0]
        all_freq_values = np.concatenate((freq_signal, np.flip(freq_signal[1:, ...].conj(), axis=0)), axis=0)
        all_freq_values[0, ...] = np.real(all_freq_values[0, ...])
        return all_freq_values



def dft_mat(dft_len : int):
    """DFT matrix corresponding to the real DFT.
    
    Parameters
    ----------
    dft_len : int
        Length of the DFT.
    num_freqs_removed_low : int
        Number of frequencies to remove.

    Returns
    -------
    F : np.ndarray of shape (L, L)
        DFT matrix

    Notes 
    -----
    The matrix represents the forward transform such that F @ a = rfft(a). 
    Note that the matrix is not square, so the inverse transform is not the 
    hermitian transpose of the forward transform.
    """
    F = splin.dft(dft_len, scale = None).conj()
    return F

def rdft_mat(dft_len : int, num_freqs_removed_low = 0):
    """DFT matrix corresponding to the real DFT.
    
    Parameters
    ----------
    dft_len : int
        Length of the DFT.
    num_freqs_removed_low : int
        Number of frequencies to remove.

    Returns
    -------
    F : np.ndarray of shape (L/2 + 1 - num_freqs_removed_low, L)
        DFT matrix with the first num_to_remove frequencies removed.

    Notes 
    -----
    The matrix represents the forward transform such that F @ a = rfft(a). 
    Note that the matrix is not square, so the inverse transform is not the 
    hermitian transpose of the forward transform.
    """
    F = splin.dft(dft_len, scale = None).conj()
    F = F[num_freqs_removed_low:dft_len//2 + 1, :]
    return F

def irdft_mat(dft_len : int, num_freqs_removed_low = 0):
    """DFT matrix corresponding to the inverse real DFT.

    The inverse real DFT is defined as Re[B @ a], where B is the dft_len x num_freqs matrix returned by this function.
    Therefore, a matrix multiplication is not sufficient to equal irfft(a), but the real part of 
    the result of the matrix multiplication is.
    
    Parameters
    ----------
    dft_len : int
        Length of the DFT.
    num_freqs_removed_low : int
        Number of frequencies to remove.

    Returns
    -------
    F : np.ndarray of shape (L/2 + 1 - num_freqs_removed_low, L)
        DFT matrix with the first num_to_remove frequencies removed.

    Notes 
    -----
    The matrix represents the forward transform such that F @ a = rfft(a). 
    Note that the matrix is not square, so the inverse transform is not the 
    hermitian transpose of the forward transform.
    """
    Finv = splin.dft(dft_len, scale = None).T
    num_freqs = dft_len // 2 + 1 - num_freqs_removed_low

    B = Finv[:, num_freqs_removed_low:dft_len//2 + 1]
    B *= rdft_weighting(num_freqs, dft_len, num_freqs_removed_low)[None,:]

    #F = F[num_freqs_removed_low:dft_len//2 + 1, :]
    return B

def rdft_weighting(num_freqs, dft_len, freqs_to_remove_low):
    """The weighting required for the real DFT to be the same as the complex DFT.

    np.diag(c_diag) corresponds to C in the paper

    Parameters
    ----------
    dft_len : int
        The length of the DFT. 
    num_freqs : int
        The number of frequencies in the resulting real-valued DFT. Technically redundant information
        as this is the same as dft_len // 2 + 1 - freqs_to_remove, so will not be necessary later. 
    freqs_to_remove_low : int
        The number of frequencies that were remove from the lower end of the DFT.

    Returns
    -------
    c_diag : np.ndarray of shape (num_freqs,)
    """
    c_diag = np.ones(num_freqs) * 2 / dft_len
    c_diag[-1] = 1 / dft_len
    if freqs_to_remove_low == 0:
        c_diag[0] = 1 / dft_len
    #c_diag = np.tile(c_diag, num_pos)
    #C = np.diag(c_diag)
    return c_diag



def dft_vector (freq_idx : int, dft_len : int):
    """All exponential values to calculate the DFT for a specific frequency bin

    Returns exp(i 2pi k n / N) / N for all time steps n = 0, 1, ..., N-1

    Parameters
    ----------
    freq_idx : int
        the index of the frequency bin. Symbol k in the formula above. 
    dft_len : int
        the number of frequency bins. Symbol N in the formula above.
    
    Returns
    -------
    exp_vector : ndarray of shape (dft_len,)

    Notes
    -----
    The values corresponds to the DFT definition used here which is u(k) = sum_n u[n] exp(i 2pi k n / N). See documentation for the fft function.

    Although inefficient, the DFT can be calculated by multiplying the input signal with this vector and summing. Preferably, use the fft and ifft functions. 
    """
    exp_factor = 2 * np.pi * 1j * freq_idx / dft_len
    return np.exp(np.arange(dft_len) * exp_factor)

def idft_vector (freq_idx : int, dft_len : int):
    """All exponential values to calculate the IDFT for a specific output time step

    Returns exp(i 2pi k n / N) for all frequency bins k = 0, 1, ..., N-1
    Which corresponds to the IDFT definition used here which is u(n) = sum_k u(k) exp(i 2pi k n / N) / N

    Parameters
    ----------
    time_idx : int
        the index of the time step bin. Symbol n in the formula above.
    dft_len : int
        the number of frequency bins. Symbol N in the formula above.
    
    Returns
    -------
    exp_vector : ndarray of shape (dft_len,)

    Notes
    -----
    The values corresponds to the DFT definition used here which is u(n) = \frac{1}{N} sum_k u(k) exp(-i 2pi k n / N). See documentation for the ifft function.

    Although inefficient, the DFT can be calculated by multiplying the input signal with this vector and summing. Preferably, use the fft and ifft functions. 
    """
    exp_factor = -2 * np.pi * 1j * freq_idx / dft_len
    return np.exp(np.arange(dft_len) * exp_factor) / dft_len



def get_freqs(num_freq : int, samplerate : int):
    """Returns the sampled frequencies in Hz for a discrete Fourier transform

    Parameters
    ----------
    num_freq : int
        should equal the length of the sequence 
        so it includes the number of negative frequencies
    samplerate : int

    Returns
    -------
    freqs : ndarray of shape (num_freq,)    
    """
    return np.arange(num_freq) * samplerate / num_freq

def get_wavenum(num_freq : int, samplerate : int, c : float):
    """Returns the wave numbers associated with the sampled frequencies of the DFT

    See documentation for get_freqs
    """
    return get_angular_freqs(num_freq, samplerate) / c

def get_angular_freqs(num_freq : int, samplerate : int):
    """Returns the angular frequencies associated with the sampled frequencies of the DFT

    See documentation for get_freqs
    """
    return 2 * np.pi * get_freqs(num_freq, samplerate)

def get_real_freqs(num_freq : int, samplerate : int):
    """Returns the real sampled frequencies in Hz for a discrete Fourier transform

    Parameters
    ----------
    num_freq :int 
        should equal the length of the sequence, so it includes 
        the number of negative frequencies
    samplerate : int
    
    Returns
    -------
    freqs : ndarray of shape (num_real_freq,)
        if num_freq is even, num_real_freq = num_freq // 2 + 1
        if num_freq is odd, num_real_freq = (num_freq + 1) // 2
    """
    if num_freq % 2 == 0:
        num_real_freq = num_freq // 2 + 1
        return (samplerate / (num_freq)) * np.arange(num_real_freq)
    elif num_freq % 2 == 1:
        num_real_freq = (num_freq + 1) // 2
        return (samplerate / (num_freq)) * np.arange(num_real_freq)
        #f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)
        #raise NotImplementedError
    else:
        raise ValueError

def get_real_wavenum(num_freq : int, samplerate : int, c : float):
    """Get wave numbers associated with the real frequencies of the DFT

    See documentation for get_real_freqs
    """
    return get_real_angular_freqs(num_freq, samplerate) / c

def get_real_angular_freqs(num_freq : int, samplerate : int):
    """Returns angular frequencies associated with the real frequencies of the DFT

    See documentation for get_real_freqs
    """
    return 2 * np.pi * get_real_freqs(num_freq, samplerate)






def freqs_to_wavenum(freqs : np.ndarray, c : float):
    """Converts an array of frequencies to wave numbers

    Parameters
    ----------
    freqs : ndarray
        the frequencies in Hz
    c : float
        the speed of sound
    
    Returns
    -------
    wavenums : ndarray
        the wavenumbers
    """
    return 2 * np.pi * freqs / c

def freqs_to_angular_freqs(freqs : np.ndarray):
    """Converts an array of frequencies to angular frequencies

    Parameters
    ----------
    freqs : ndarray
        the frequencies in Hz
    
    Returns
    -------
    angular_freqs : ndarray
        the angular frequencies
    """
    return 2 * np.pi * freqs

def angular_freqs_to_wavenum(angular_freqs : np.ndarray, c : float):
    """Converts an array of angular frequencies to wave numbers

    Parameters
    ----------
    angular_freqs : ndarray
        the angular frequencies
    c : float
        the speed of sound
    
    Returns
    -------
    wavenums : ndarray
        the wavenumbers
    """
    return angular_freqs / c