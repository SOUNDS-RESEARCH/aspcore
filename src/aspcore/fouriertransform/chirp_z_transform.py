"""Methods for the chirp z transform

The axis convention is the as for the DFT, where time samples are assumed to be on the last axis, 
and the frequency bins are on the first axis.
The time convention is the same as for the DFT, which is opposite from the numpy.fft module. 


References
----------
[SukhoyGeneralizing2019] Generalizing the inverse FFT of the unit circle, Sukhoy V., Stoytchev A., https://www.nature.com/articles/s41598-019-50234-9.pdf
""" 
import numpy as np
import aspcore.fouriertransform.dft as dft

def czt(time_sig, M, w=None, a = 1+0j):
    """The Chirp-z Transform

    The chirp z transform is a generalization of the DFT, where the chirp z transform can be computed for a larger set of
    z values in the complex plane. The DFT corresponds to equal values on the unit circle. 
    
    Parameters
    ----------
    time_sig : np.ndarray of shape (..., N)
        The time signal to be transformed. The transform is applied to the last axis
    M : int
        The number of frequency bins to compute. 
    w : complex, optional
        The complex number to use as the base for the frequency bins. If not specified, the default is exp(2j*pi/M), 
        which correponds to equally spaced valued around the unit circle. This is the same as the DFT.
    a : complex, optional
        The starting point for the chirp z transform. The default is 1+0j, which corresponds to the DFT.

    Returns
    -------
    np.ndarray of shape (M, ...)
        The chirp z transform of the input signal. The first axis corresponds to the frequency bins, and the remaining
        axes correspond to the input shape.
        
    Notes
    -----
    Implemented using a matrix multiplication approach, which has O(n^2) complexity, analogous to computing the
    DFT using the DFT matrix. 

    The chirp z transform is defined as
    X_k = sum_{n=0}^{N-1} x_n * A^{-n} * W^{nk}
    or written as a matrix multiplication 
    X = W A x
    where A is a N x N diagonal matrix with A^{-n} on the diagonal, and W is a MxN matrix with the elements
    (W)_{mn} = W^{mn}.

    References
    ----------
    [SukhoyGeneralizing2019]

    """
    if w is None:
        w = np.exp(2j * np.pi / M)

    N = time_sig.shape[-1]
    input_dims = time_sig.ndim - 1
    assert input_dims > 0

    # Expand the transform matrix to the same number of axes as the time signal
    mat = _czt_matrix(N, M, w, a)
    mat = np.expand_dims(mat, np.arange(input_dims).tolist())

    freq_sig = mat @ time_sig[..., None]
    freq_sig = np.squeeze(freq_sig, -1)
    freq_sig = np.moveaxis(freq_sig, -1, 0)
    return freq_sig

def _czt_matrix(N, M, w, a):
    powers = np.arange(M)[: , None] * np.arange(N)[None,:]
    w_mat = w ** powers
    a_mat = a ** -np.arange(N) # this is not converted to a diagonal matrix explicitly
    mat = w_mat * a_mat[None,:] # corresponds to w_mat @ np.diag(a_mat)
    return mat

def iczt(freq_sig, N, w=None, a = 1+0j):
    """The Inverse Chirp-z Transform

    The inverse chirp z transform is the inverse of the chirp z transform. The inverse
    is only defined when M = N. 

    Parameters
    ----------
    freq_sig : np.ndarray of shape (M, ...)
        The frequency signal to be transformed. The transform is applied to the first axis
    N : int
        The number of time samples to compute. 
    w : complex, optional
        The complex number to use as the base for the frequency bins. If not specified, the default is exp(2j*pi/M), 
        which correponds to equally spaced valued around the unit circle. This is the same as the DFT.
    a : complex, optional
        The starting point for the chirp z transform. The default is 1+0j, which corresponds to the DFT.

    Returns
    -------
    np.ndarray of shape (..., N)
        The inverse chirp z transform of the input signal. The last axis corresponds to the time samples, and the remaining
        axes correspond to the input shape.

    Notes
    -----
    Implemented using a matrix multiplication approach, which has O(n^2) complexity, analogous to computing the
    DFT using the DFT matrix. 

    The inverse chirp z transform is defined as
    x = (W A)^{-1} X
    where the matrices are defined in the notes of the czt function.

    References
    ----------
    [SukhoyGeneralizing2019]
    """
    if w is None:
        w = np.exp(2j * np.pi / freq_sig.shape[0])

    M = freq_sig.shape[0]
    assert M == N, "The inverse chirp z transform is only defined when M = N"

    input_dims = freq_sig.ndim - 1
    assert input_dims > 0, "The input signal must have at least one axis"

    mat = _czt_matrix(N, M, w, a)
    mat = np.expand_dims(mat, np.arange(input_dims).tolist())

    time_sig = np.linalg.solve(mat, np.moveaxis(freq_sig, 0, -1))
    return time_sig


def czt_unit_circle(time_sig, M, w_angle, a_angle):
    """Chirp-z transform on the unit circle

    Can be viewed as a special case of the chirp z transform, where both a and w are on the unit circle. 
    This is a generalization of the DFT, where the frequency bins do not necessarily cover the whole unit circle.

    Parameters
    ----------
    time_sig : np.ndarray of shape (..., N)
        The time signal to be transformed. The transform is applied to the last axis
    a_angle : float
        The starting point for the chirp z transform. The value a in the chirp z transform is a complex number
        defined as exp(2j * pi * a_angle).
    w_angle : float
        The base for the frequency bins. The value w in the chirp z transform is a complex number
        defined as exp(2j * pi * w_angle).
    """
    N = time_sig.shape[-1]
    input_dims = time_sig.ndim - 1
    assert input_dims > 0

    # Expand the transform matrix to the same number of axes as the time signal
    mat = _czt_unit_circle_matrix(N, M, w_angle, a_angle)
    mat = np.expand_dims(mat, np.arange(input_dims).tolist())

    freq_sig = mat @ time_sig[..., None]
    freq_sig = np.squeeze(freq_sig, -1)
    freq_sig = np.moveaxis(freq_sig, -1, 0)
    return freq_sig

def _czt_unit_circle_matrix(N, M, w_angle, a_angle):
    a_angles = -np.arange(N) * a_angle

    powers = np.arange(M)[: , None] * np.arange(N)[None,:]
    w_angles = powers * w_angle

    tot_angles = w_angles + a_angles[None,:]
    mat = np.exp(2j * np.pi * tot_angles)
    return mat

def iczt_unit_circle(freq_sig, N, w_angle, a_angle):
    """Inverse Chirp-z transform on the unit circle

    Parameters
    ----------
    freq_sig : np.ndarray of shape (M, ...)
        The frequency signal to be transformed. The transform is applied to the first axis
    N : int
        The number of time samples to compute. 
    a_angle : float
        The starting point for the chirp z transform. The value a in the chirp z transform is a complex number
        defined as exp(2j * pi * a_angle).
    w_angle : float
        The base for the frequency bins. The value w in the chirp z transform is a complex number
        defined as exp(2j * pi * w_angle).
    """
    M = freq_sig.shape[0]
    input_dims = freq_sig.ndim - 1
    assert input_dims > 0, "The input signal must have at least one axis"

    mat = _czt_unit_circle_matrix(N, M, w_angle, a_angle)
    mat = np.expand_dims(mat, np.arange(input_dims).tolist())

    time_sig = np.linalg.solve(mat, np.moveaxis(freq_sig, 0, -1))
    return time_sig




def zoom_dft(time_sig, M, freq_limits, samplerate):
    """Zoom Discrte Fourier Transform
    
    Equivalent to czt with unit circle parameters, but here with parameters in terms of hertz to make for 
    more intuitive use. Samples the unit circle on evenly spaced points on the interval [freq_limits[0], freq_limits[1]].

    Parameters
    ----------
    time_sig : np.ndarray of shape (..., N)
        The time signal to be transformed. The transform is applied to the last axis
    M : int
        The number of frequency bins to compute.
    freq_limits : tuple of floats
        The lower and upper frequency limits to sample.
    samplerate : int
        The samplerate of the time signal.
    """
    a_angle = freq_limits[0] / samplerate
    w_angle = (freq_limits[1] - freq_limits[0]) / (samplerate * M)

    return czt_unit_circle(time_sig, M, w_angle, a_angle)


def zoom_idft(freq_sig, N, freq_limits, samplerate):
    """Zoom Inverse Discrte Fourier Transform
    
    Equivalent to iczt with unit circle parameters, but here with parameters in terms of hertz to make for 
    more intuitive use. 
    
    """
    a_angle = freq_limits[0] / samplerate
    w_angle = (freq_limits[1] - freq_limits[0]) / (samplerate * N)
    return iczt_unit_circle(freq_sig, N, w_angle, a_angle)






if __name__ == "__main__":
    import scipy.signal as spsig
    rng = np.random.default_rng(3456)
    sig = rng.normal(0, 1, size=(3,4,5,100))

    freq_sig = czt(sig, 100)
    time_sig = iczt(freq_sig, 100)
    print(np.allclose(sig, time_sig))
    freq_sig_2 = np.fft.fft(sig, axis=-1)
    freq_sig_3 = dft.fft(sig)
    freq_sig_4 = spsig.czt(sig, 100)
    freq_sig_5 = spsig.zoom_fft(sig, 2, axis=-1)

    print(np.allclose(np.moveaxis(freq_sig, 0, -1), freq_sig_2))
    print(np.allclose(freq_sig, freq_sig_3))
    print(np.allclose(freq_sig_2, freq_sig_4))
    print(np.allclose(freq_sig_2, freq_sig_5))
