"""A collection of classes implementing linear convolution. 

The functionality offered beyond what is available in numpy and scipy is
inherent support for MIMO filters in different forms, as well as
streaming filtering, where only a part of the signal is known at a time. 
It is also possible to filter with a time-varying impulse response.

The classes are JIT compiled using numba's experimental jitclass to keep
computational cost low.

The main function of the module is create_filter, which constructs the appropriate 
filter object according to the chosen parameters. The only required parameters is
either an impulse response or the dimensions of the impulse response. In the 
latter case, the impulse response is initialized to zero.


In addition to linear convolution, the module also contains implementations of
- Weighted overlap-add (WOLA) [crochiereWeighted1980, ruizComparison2021]
- IIR filter
- Mean with forgetting factor

References
----------
[crochiereWeighted1980] R. Crochiere, “A weighted overlap-add method of short-time Fourier analysis/synthesis,” IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 28, no. 1, pp. 99–102, Feb. 1980, doi: 10.1109/TASSP.1980.1163353. `[link] <https://doi.org/10.1109/TASSP.1980.1163353>`__ \n
[ruizComparison2021] S. Ruiz, T. Dietzen, T. van Waterschoot, and M. Moonen, “A comparison between overlap-save and weighted overlap-add filter banks for multi-channel Wiener filter based noise reduction,” in 2021 29th European Signal Processing Conference (EUSIPCO), Aug. 2021, pp. 336–340. doi: 10.23919/EUSIPCO54536.2021.9616352. `[link] <https://doi.org/10.23919/EUSIPCO54536.2021.9616352>`__ \n



"""
import numpy as np
import itertools as it

import aspcore.fouriertransform as ft
import aspcore.filterdynamic as fcd
import aspcore.utilities as utils

import scipy.signal as spsig
import numba as nb
import numexpr as ne



def create_filter(
    ir=None, 
    num_in=None, 
    num_out=None, 
    ir_len=None, 
    broadcast_dim=None, 
    sum_over_input=True, 
    dynamic=False):
    """
    Returns the appropriate filter object for the desired use.

    All filters has a method called process which takes one parameter 
    signal : ndarray of shape (num_in, num_samples)
    and returns an ndarray of shape (out_dim, num_samples), where 
    out_dim depends on which filter. 

    Either ir must be provided or num_in, num_out and ir_len. In the
    latter case, the filter coefficients are initialized to zero.  

    Parameters
    ----------
    ir : ndarray of shape (num_in, num_out, ir_len)
    num_in : int
        the number of input channels
    num_out : int
        the number of output channels
    ir_len : int
        the length of the impulse response
    broadcast_dim : int
        Supply this value if the filter should be applied to more than one set of signals
        at the same time. 
    sum_over_input : bool
        Choose true to sum over the input dimension after filtering. For a canonical MIMO
        system choose True, for example when simulating sound in a space with multiple 
        loudspeakers.
        Choose False otherwise, in which case filter.process will return a ndarray of 
        shape (num_in, num_out, num_samples)
    dynamic : bool
        Choose true to use a convolution specifically implemented for time-varying systems.
        More accurate, but takes more computational resources. 

    Returns
    -------
    filter : Appropriate filter class from this module
    """
    if num_in is not None and num_out is not None and ir_len is not None:
        assert ir is None
        ir = np.zeros((num_in, num_out, ir_len))

    if ir is not None:
        #assert numIn is None and numOut is None and irLen is None
        if broadcast_dim is not None:
            if dynamic:
                raise NotImplementedError
            if sum_over_input:
                raise NotImplementedError
            assert ir.ndim == 3 and isinstance(broadcast_dim, int) # A fallback to non-numba MD-filter can be added instead of assert
            return FilterBroadcast(broadcast_dim, ir)
        
        if sum_over_input:
            if dynamic:
                return fcd.FilterSumDynamic(ir)
            return FilterSum(ir)
        return FilterNosum(ir)

spec_filtersum = [
    ("ir", nb.float64[:,:,:]),
    ("num_in", nb.int32),
    ("num_out", nb.int32),
    ("ir_len", nb.int32),
    ("buffer", nb.float64[:,:]),
]
@nb.experimental.jitclass(spec_filtersum)
class FilterSum:
    """A class for implementing an LTI MIMO system. 

    The input dimension will be summed over. 
    The signal can be processed in chunks of arbitrary size, and the
    internal buffer will be updated accordingly.

    Parameters
    ----------
    ir : ndarray of shape (num_in, num_out, ir_len)
        The impulse response of the filter
    """
    def __init__(self, ir):
        self.ir = ir
        self.num_in = ir.shape[0]
        self.num_out = ir.shape[1]
        self.ir_len = ir.shape[2]
        self.buffer = np.zeros((self.num_in, self.ir_len - 1))

    def process(self, data_to_filter):
        """Filter the data with the impulse response.
        
        Parameters
        ----------
        data_to_filter : ndarray of shape (num_in, num_samples)
            The data to be filtered
        
        Returns
        -------
        filtered : ndarray of shape (num_out, num_samples)
            The filtered data
        """
        num_samples = data_to_filter.shape[-1]
        buffered_input = np.concatenate((self.buffer, data_to_filter), axis=-1)

        filtered = np.zeros((self.num_out, num_samples))
        for out_idx in range(self.num_out):
            for i in range(num_samples):
                filtered[out_idx,i] += np.sum(self.ir[:,out_idx,:] * np.fliplr(buffered_input[:,i:self.ir_len+i]))

        self.buffer[:, :] = buffered_input[:, buffered_input.shape[-1] - self.ir_len + 1 :]
        return filtered


spec_filtermd = [
    ("ir", nb.float64[:,:,:]),
    ("ir_dim1", nb.int32),
    ("ir_dim2", nb.int32),
    ("ir_len", nb.int32),
    ("data_dims", nb.int32),
    ("buffer", nb.float64[:,:]),
]
@nb.experimental.jitclass(spec_filtermd)
class FilterBroadcast:
    """Filters all channels of the input signal with all channels of the impulse response.
    
    Parameters
    ----------
    data_dims : int
        The number of channels of the input signal
    ir : ndarray of shape (ir_dim1, ir_dim2, ir_len)
        The impulse response of the filter
    """
    def __init__(self, data_dims, ir):
        self.ir = ir
        self.ir_dim1 = ir.shape[0]
        self.ir_dim2 = ir.shape[1]
        self.ir_len = ir.shape[-1]
        self.data_dims = data_dims

        self.buffer = np.zeros((self.data_dims, self.ir_len - 1))

    def process(self, data_to_filter):
        """Filters all input channels with all impulse response channels.
        
        Parameters
        ----------
        data_to_filter : ndarray of shape (data_dims, num_samples)
            The data to be filtered
        
        Returns
        -------
        filtered : ndarray of shape (data_dims, dim1, dim2, num_samples)
            The filtered data
        """
        num_samples = data_to_filter.shape[-1]

        buffered_input = np.concatenate((self.buffer, data_to_filter), axis=-1)
        filtered = np.zeros((self.ir_dim1, self.ir_dim2, self.data_dims, num_samples))

        for i in range(num_samples):
            filtered[:, :, :,i] = np.sum(np.expand_dims(self.ir,2) * \
                    np.expand_dims(np.expand_dims(np.fliplr(buffered_input[:,i:self.ir_len+i]),0),0),axis=-1)

        self.buffer[:, :] = buffered_input[:, -self.ir_len + 1:]
        return filtered



class FilterNosum:
    """Filters a signal with a MIMO filter without summing over the input dimension.
    
    Is essentially equivalent to FilterBroadcast with ir_dims1=1, 
    although this can be more convenient to use in some cases.

    Is not JIT compiled, and is therefore generally slower than FilterBroadcast
    and FilterSum.

    Parameters
    ----------
    ir : ndarray of shape (num_in, num_out, ir_len)
        The impulse response of the filter
    """

    def __init__(self, ir=None, ir_len=None, num_in=None, num_out=None):
        if ir is not None:
            self.ir = ir
            self.num_in = ir.shape[0]
            self.num_out = ir.shape[1]
            self.ir_len = ir.shape[2]
        elif (ir_len is not None) and (num_in is not None) and (num_out is not None):
            self.ir = np.zeros((num_in, num_out, ir_len))
            self.ir_len = ir_len
            self.num_in = num_in
            self.num_out = num_out
        else:
            raise Exception("Not enough constructor arguments")
        self.buffer = np.zeros((self.num_in, self.ir_len - 1))

    def process(self, data_to_filter):
        """Filter the data with the impulse response.

        Parameters
        ----------
        data_to_filter : ndarray of shape (num_in, num_samples)
            The data to be filtered
        
        Returns
        -------
        filtered : ndarray of shape (num_in, num_out, num_samples)
            The filtered data   
        """
        num_samples = data_to_filter.shape[-1]
        buffered_input = np.concatenate((self.buffer, data_to_filter), axis=-1)

        filtered = np.zeros((self.num_in, self.num_out, num_samples))
        if num_samples > 0:
            for in_idx, out_idx in it.product(range(self.num_in), range(self.num_out)):
                filtered[in_idx, out_idx, :] = spsig.convolve(
                    self.ir[in_idx, out_idx, :], buffered_input[in_idx, :], "valid"
                )

            self.buffer[:, :] = buffered_input[:, buffered_input.shape[-1] - self.ir_len + 1 :]
        return filtered


class FilterMD_slow_fallback:
    """Equivalent to FilterBroadcast, but can handle arbitrary number of dimensions.

    Is not JIT compiled, and is therefore generally slower than FilterBroadcast

    Parameters
    ----------
    data_dims : tuple of ints
        The number of channels of the input signal
    ir : ndarray of shape (ir_dim1, ir_dim2, ..., ir_dimN, ir_len)
        The impulse response of the filter    
    """
    def __init__(self, data_dims, ir=None, ir_len=None, ir_dims=None):
        if ir is not None:
            self.ir = ir
            self.ir_len = ir.shape[-1]
            self.ir_dims = ir.shape[0:-1]
        elif (ir_len is not None) and (ir_dims is not None):
            self.ir = np.zeros(ir_dims + (ir_len,))
            self.ir_len = ir_len
            self.ir_dims = ir_dims
        else:
            raise Exception("Not enough constructor arguments")

        if isinstance(data_dims, int):
            self.data_dims = (data_dims,)
        else:
            self.data_dims = data_dims

        self.output_dims = self.ir_dims + self.data_dims
        self.num_data_dims = len(self.data_dims)
        self.num_ir_dims = len(self.ir_dims)
        self.buffer = np.zeros(self.data_dims + (self.ir_len - 1,))

    def process(self, data_to_filter):
        """Filters all input channels with all impulse response channels.

        Parameters
        ----------
        data_to_filter : ndarray of shape (b0, b1, ..., bn, num_samples)
            The data to be filtered

        Returns
        -------
        filtered : ndarray of shape (a0, a1, ..., an, b0, b1, ..., bn, num_samples)
            The filtered data
        """
        num_samples = data_to_filter.shape[-1]

        buffered_input = np.concatenate((self.buffer, data_to_filter), axis=-1)
        filtered = np.zeros(self.ir_dims + self.data_dims + (num_samples,))

        for idxs in it.product(*[range(d) for d in self.output_dims]):
            filtered[idxs + (slice(None),)] = spsig.convolve(
                self.ir[idxs[0 : self.num_ir_dims] + (slice(None),)],
                buffered_input[idxs[-self.num_data_dims :] + (slice(None),)],
                "valid",
            )

        self.buffer[..., :] = buffered_input[..., -self.ir_len + 1 :]
        return filtered


















class FilterBroadcastFreq:
    """ir is the time domain impulse response, with shape (a0, a1,..., an, irLen)
    tf is frequency domain transfer function, with shape (2*irLen, a0, a1,..., an),

    input of filter function is shape (b0, b1, ..., bn, irLen)
    output of filter function is shape (a0,...,an,b0,...,bn,irLen)

    dataDims must be provided, which is a tuple like (b0, b1, ..., bn)
    filtDim is (a0, a1,..., an) and must then be complemented with irLen or numFreq
    If you give to many arguments, it will propritize tf -> ir -> numFreq -> irLen"""

    def __init__(
        self, data_dims, tf=None, ir=None, filt_dim=None, ir_len=None, num_freq=None
    ):
        # assert(tf or ir or (numIn and numOut and irLen) is not None)
        if tf is not None:
            assert tf.shape[0] % 2 == 0
            self.tf = tf
        elif ir is not None:
            self.tf = ft.fft(
                np.concatenate((ir, np.zeros_like(ir)), axis=-1)
            )
        elif filt_dim is not None:
            if num_freq is not None:
                self.tf = np.zeros((num_freq, *filt_dim), dtype=complex)
            elif ir_len is not None:
                self.tf = np.zeros((2 * ir_len, *filt_dim), dtype=complex)
        else:
            raise ValueError("Arguments missing for valid initialization")

        if isinstance(data_dims, int):
            self.data_dims = (data_dims,)
        else:
            self.data_dims = data_dims

        self.ir_len = self.tf.shape[0] // 2
        self.num_freq = self.tf.shape[0]
        self.filt_dim = self.tf.shape[1:]

        self.buffer = np.zeros((*self.data_dims, self.ir_len))

    def process(self, samples_to_process):
        assert samples_to_process.shape == (*self.data_dims, self.ir_len)
        output_samples = ft.convolve_euclidian_ft(
            self.tf, np.concatenate((self.buffer, samples_to_process), axis=-1)
        )

        self.buffer[...] = samples_to_process
        return output_samples

    def process_freq(self, freqs_to_process):
        """Can be used if the fft of the input signal is already available.
        Assumes padding is already applied correctly."""
        assert freqs_to_process.shape == (self.num_freq, *self.data_dims)
        output_samples = ft.convolve_euclidian_ff(self.tf, freqs_to_process)

        self.buffer[...] = ft.ifft(freqs_to_process)[
            ..., self.ir_len :
        ]
        return output_samples

class FilterSumFreq:
    """ir is the time domain impulse response, with shape (numIn, numOut, irLen)
    tf is frequency domain transfer function, with shape (2*irLen, numOut, numIn),
    it is also possible to only provide the dimensions of the filter, numIn and numOut,
    together with either number of frequencies or ir length, where 2*irLen==numFreq
    dataDims is extra dimensions of the data to broadcast over. Input should then be
    with shape (*dataDims, numIn, numSamples), output will be (*dataDims, numOut, numSamples)

    If you give to many arguments, it will propritize tf -> ir -> numFreq -> irLen"""

    def __init__(
        self,
        tf=None,
        ir=None,
        num_in=None,
        num_out=None,
        ir_len=None,
        num_freq=None,
        data_dims=None,
    ):
        if tf is not None:
            assert all(arg is None for arg in [ir, num_in, num_out, ir_len, num_freq])
            assert tf.ndim == 3
            assert tf.shape[0] % 2 == 0
            self.tf = tf
        elif ir is not None:
            assert all(arg is None for arg in [tf, num_in, num_out])
            if ir_len is not None:
                assert num_freq is None
                tot_len = ir_len*2
            elif num_freq is not None:
                assert ir_len is None
                tot_len = num_freq
            else:
                tot_len = 2*ir.shape[-1]

            ir_padded = np.concatenate((ir, np.zeros(ir.shape[:-1]+(tot_len-ir.shape[-1],))), axis=-1)
            self.tf = ft.fft(ir_padded)
            # np.transpose(
            #     np.fft.fft(,
            #     (2, 1, 0),
            # )

            # self.tf = np.transpose(
            #     np.fft.fft(np.concatenate((ir, np.zeros_like(ir)), axis=-1), axis=-1),
            #     (2, 1, 0),
            # )
        elif num_in is not None and num_out is not None:
            assert all(arg is None for arg in [tf, ir])
            if num_freq is not None:
                self.tf = np.zeros((num_freq, num_out, num_in), dtype=complex)
            elif ir_len is not None:
                self.tf = np.zeros((2 * ir_len, num_out, num_in), dtype=complex)
        else:
            raise ValueError("Arguments missing for valid initialization")

        self.ir_len = self.tf.shape[0] // 2
        self.num_freq = self.tf.shape[0]
        self.num_out = self.tf.shape[1]
        self.num_in = self.tf.shape[2]

        if data_dims is not None:
            if isinstance(data_dims, int):
                self.data_dims = (data_dims,)
            else:
                self.data_dims = data_dims
            self.len_data_dims = len(self.data_dims)
            self.buffer = np.zeros((*self.data_dims, self.num_in, self.ir_len))
        else:
            self.buffer = np.zeros((self.num_in, self.ir_len))
            self.len_data_dims = 0

    def process(self, samples_to_process):
        assert samples_to_process.shape == self.buffer.shape
        sig_buffered = np.concatenate((self.buffer, samples_to_process), axis=-1)

        freqs_to_process = ft.fft(sig_buffered)
        freqs_to_process = np.expand_dims(freqs_to_process, -1)

        tf_new_shape = self.tf.shape[0:1] + (1,) * self.len_data_dims + self.tf.shape[1:]

        temp = self.tf.reshape(tf_new_shape) @ freqs_to_process
        temp = np.squeeze(temp, axis=-1)
        output_samples = ft.ifft(temp)

        self.buffer[...] = samples_to_process
        return np.real(output_samples[..., self.ir_len :])

    def process_freq(self, freqs_to_process):
        """Can be used if the fft of the input signal is already available.
        Assumes padding is already applied correctly."""
        assert freqs_to_process.shape == (self.num_freq, self.num_in, 1)
        tf_new_shape = self.tf.shape[0:1] + (1,) * self.len_data_dims + self.tf.shape[1:]

        temp = self.tf.reshape(tf_new_shape) @ freqs_to_process
        temp = np.squeeze(temp, axis=-1)
        output_samples = ft.ifft(temp)

        freqs_to_process = np.squeeze(freqs_to_process, axis=-1)
        self.buffer[...] = ft.ifft(freqs_to_process)[..., self.ir_len :]
        return np.real(output_samples[..., self.ir_len :])

    def process_without_sum(self, samples_to_process):
        assert samples_to_process.shape == self.buffer.shape

        freqs_to_process = ft.fft(np.concatenate((self.buffer, samples_to_process), axis=-1))
        freqs_to_process = np.expand_dims(freqs_to_process, -1)

        tf_new_shape = self.tf.shape[0:1] + (1,) * self.len_data_dims + self.tf.shape[1:]

        temp = np.swapaxes(self.tf.reshape(tf_new_shape),-1,-2) * freqs_to_process
        temp = np.squeeze(temp, axis=-1)
        output_samples = ft.ifft(temp)

        self.buffer[...] = samples_to_process
        return np.real(output_samples[..., self.ir_len :])

    def process_euclidian(self, samples_to_process):
        """Will filter every channel of the input with every channel of the filter
            outputs shape (filt0)"""
        #assert dataDims is not None
        raise NotImplementedError

    def set_ir(self, ir_new):
        assert ir_new.shape == (self.num_in, self.num_out, self.ir_len)
        self.tf = np.transpose(
                np.fft.fft(np.concatenate((ir_new, np.zeros_like(ir_new)), axis=-1), axis=-1),
                (2, 1, 0),
            )










class MovingAverage:
    def __init__(self, forget_factor, dim, dtype=np.float64):
        self.state = np.zeros(dim, dtype=dtype)
        self.forget_factor = forget_factor
        self.forget_factor_inv = 1 - forget_factor

        self.initialized = False
        self.init_counter = 0
        if self.forget_factor == 1:
            self.num_init = np.inf
        else:
            self.num_init = int(np.ceil(1 / self.forget_factor_inv))

    def update(self, new_data_point, count_as_updates=1):
        """
        
        count_as_updates can be used if the datapoint is already average
        outside of this class. So if new_data_point is the average of N data 
        points, count_as_updates should be set to N.
        """
        assert new_data_point.shape == self.state.shape
        if self.initialized:
            if count_as_updates > 1:
                raise NotImplementedError

            self.state[...] = ne.evaluate("state*ff + new_data_point*ff_inv", 
                                local_dict={"state":self.state, "new_data_point":new_data_point,
                                            "ff":self.forget_factor, "ff_inv":self.forget_factor_inv})
            #self.state *= self.forget_factor
            #self.state += new_data_point * self.forget_factor_inv
        else:
            self.state[...] = ne.evaluate("state*(i/(i+j)) + new_data_point*(j/(i+j))", 
                                local_dict={'state': self.state, "new_data_point":new_data_point, 
                                            'i': self.init_counter, "j" : count_as_updates})

            #self.state *= (self.init_counter / (self.init_counter + 1))
            #self.state += new_data_point / (self.init_counter + 1)
            self.init_counter += count_as_updates
            if self.init_counter >= self.num_init:
                self.initialized = True
                if self.init_count > self.num_init:
                    print("Initialization happened not exactly at self.num_init")

    def reset(self):
        self.initialized = False
        self.num_init = np.ceil(1 / self.forget_factor_inv)
        self.init_counter = 0




class IIRFilter:
    """
    num_coeffs and denom_coeffs should be a list of ndarrays,
        containing the parameters of the rational transfer function
        If only one channel is desired, the arguments can just be a ndarray
    """
    def __init__(self, num_coeffs, denom_coeffs):

        if not isinstance(num_coeffs, (list, tuple)):
            assert not isinstance(denom_coeffs, (list, tuple))
            num_coeffs = [num_coeffs]
            denom_coeffs = [denom_coeffs]
        assert isinstance(denom_coeffs, (list, tuple))
        self.num_coeffs = num_coeffs
        self.denom_coeffs = denom_coeffs

        self.num_channels = len(self.num_coeffs)
        assert len(self.num_coeffs) == len(self.denom_coeffs)

        self.order = [max(len(nc), len(dc)) for nc, dc in zip(self.num_coeffs, self.denom_coeffs)]
        self.filter_state = [spsig.lfiltic(nc, dc, np.zeros((len(dc)-1))) 
                                        for nc, dc in zip(self.num_coeffs, self.denom_coeffs)]

    def process(self, data_to_filter):
        assert data_to_filter.ndim == 2
        num_channels = data_to_filter.shape[0]
        filtered_sig = np.zeros_like(data_to_filter)
        for ch_idx in range(num_channels):
            filtered_sig[ch_idx,:], self.filter_state[ch_idx] = spsig.lfilter(self.num_coeffs[ch_idx], self.denom_coeffs[ch_idx], data_to_filter[ch_idx,:], axis=-1, zi=self.filter_state[ch_idx])
        return filtered_sig







class WOLA:
    def __init__(self, num_in : int, num_out : int, block_size : int, overlap : int):
        self.num_in = num_in
        self.num_out = num_out
        self.block_size = block_size
        self.overlap = overlap
        self.hop = block_size - overlap
        assert self.hop > 0

        if self.block_size % 2 == 0:
            self.num_freqs = 1 + self.block_size // 2
        else:
            raise NotImplementedError

        self.win = get_window_wola(self.block_size, self.overlap)
        #self.win = np.sqrt(1/2)*np.ones(self.block_size)
        self.buf_in = np.zeros((self.num_in, self.overlap), dtype=float)
        self.buf_out = np.zeros((self.num_in, self.num_out, self.overlap), dtype=float)
        self.spectrum = np.zeros((self.num_in, self.num_out, self.num_freqs), dtype=complex)

    def analysis(self, sig):
        """Performs WOLA analysis and saved the contents to self.spectrum 
        
        Parameters
        ----------
        sig : ndarray (self.num_in, self.hop)
            the new samples from the signal that should be analyzed
        """
        assert sig.ndim == 2
        assert sig.shape[-1] == self.hop
        if sig.shape[0] == 1:
            sig = np.broadcast_to(sig, (self.num_in, self.hop))

        sig_to_analyze = np.concatenate((self.buf_in, sig), axis=-1)
        self.buf_in[...] = sig_to_analyze[:,-self.overlap:]

        self.spectrum[...] = wola_analysis(sig_to_analyze, self.win)[:,None,:]
        

    def synthesis(self):
        """Performs WOLA synthesis, sums with previous blocks and returns 
            the self.hop number of valid samples 
        
        Parameters
        ----------

        Return
        ------


        """
        sig = wola_synthesis(self.spectrum, self.buf_out, self.win, self.overlap)

        #sig[:,:self.overlap] += self.buf_out
        self.buf_out[...] = sig[...,-self.overlap:]

        #complete_sig = sig[:,:self.hop] += sig_last_block
        return sig[...,:self.hop]



def get_window_wola(block_size : int, overlap : int):
    """
    block_size is number of samples
    overlap is number of samples
    """
    win = spsig.windows.hann(block_size, sym=False)
    assert spsig.check_COLA(win, block_size, overlap)
    return np.sqrt(win)

def wola_analysis(sig, window):
    """Generate WOLA spectrum from time domain signal

    Parameters
    ----------
    sig : ndarray (num_channels, num_samples)
    window : ndarray (num_samples)

    Returns
    -------
    spectrum : ndarray (num_channels, num_samples//2 + 1)
    """
    num_samples = sig.shape[-1]
    assert sig.ndim == 2
    assert window.ndim == 1
    assert window.shape[0] == num_samples
   
    sig_windowed = sig * window[None,:]
    spectrum = ft.rfft(sig_windowed).T
    #spectrum = np.fft.rfft(sig * window[None,:], axis=-1)
    return spectrum

def wola_synthesis(spectrum, sig_last_block, window, overlap):
    """Generate time domain signal from WOLA spectrum
        
    Keep in mind that only the first block_size-overlap are correct
    the last overlap samples should be saved until last block to 
    be overlapped with the next block

    Parameters
    ----------
    spectrum : ndarray (..., num_channels, num_samples//2 + 1)
        the spectrum associated with the positive frequencies (output from rfft), 
        which will be num_samples // 2 + 1 frequencies. 
    sig_last_block : ndarray (..., num_channels, overlap)
    window : ndarray (num_samples)
    overlap : int 

    Returns
    -------
    sig : ndarray (..., num_channels, num_samples)
    """
    assert spectrum.ndim >= 2
    assert sig_last_block.ndim >= 2
    assert window.ndim == 1

    block_size = window.shape[0]
    num_channels = spectrum.shape[-2]

    assert spectrum.shape[:-2] == sig_last_block.shape[:-2]
    assert spectrum.shape[-1] == block_size // 2 + 1
    assert sig_last_block.shape[-2:] == (num_channels, overlap)
    
    if overlap != block_size // 2:
        raise NotImplementedError
    if not utils.is_power_of_2(block_size):
        raise NotImplementedError

    win_broadcast_dims = spectrum.ndim - 1
    window = window.reshape(win_broadcast_dims*(1,) + (-1,))

    sig = window * np.real_if_close(ft.irfft(np.moveaxis(spectrum, -1, 0)))
    
    sig[...,:overlap] += sig_last_block
    return sig