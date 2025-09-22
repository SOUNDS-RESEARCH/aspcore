import hypothesis as hyp
import hypothesis.strategies as st


import numpy as np
import aspcore.fouriertransform as ft
import aspcore.fouriertransform.dft as dft_module
import aspcore.filter as fc


@hyp.settings(deadline=None)
@hyp.given(num_ch1 = st.integers(min_value=1, max_value=3),
            num_ch2 = st.integers(min_value=1, max_value=3), 
            num_ch3 = st.integers(min_value=1, max_value=3), 
            fft_len = st.integers(min_value=16, max_value=256))
def test_fft_ifft_returns_original_signal(num_ch1, num_ch2, num_ch3, fft_len):
    rng = np.random.default_rng()
    signal = rng.normal(size=(num_ch1, num_ch2, num_ch3, fft_len))
    fft_signal = ft.fft(signal)
    ifft_signal = ft.ifft(fft_signal)
    assert np.allclose(signal, ifft_signal)

@hyp.settings(deadline=None)
@hyp.given(num_ch1 = st.integers(min_value=1, max_value=3),
            num_ch2 = st.integers(min_value=1, max_value=3), 
            num_ch3 = st.integers(min_value=1, max_value=3), 
            half_fft_len = st.integers(min_value=16, max_value=256))
def test_rfft_irfft_returns_original_signal(num_ch1, num_ch2, num_ch3, half_fft_len):
    rng = np.random.default_rng()
    fft_len = 2*half_fft_len
    signal = rng.normal(size=(num_ch1, num_ch2, num_ch3, fft_len))
    fft_signal = ft.rfft(signal)
    ifft_signal = ft.irfft(fft_signal)
    assert np.allclose(signal, ifft_signal)

@hyp.settings(deadline=None)
@hyp.given(num_ch1 = st.integers(min_value=1, max_value=3),
            num_ch2 = st.integers(min_value=1, max_value=3), 
            num_ch3 = st.integers(min_value=1, max_value=3), 
            sig_len = st.integers(min_value=16, max_value=256))
def test_rfft_irfft_with_padding_returns_original_signal(num_ch1, num_ch2, num_ch3, sig_len):
    rng = np.random.default_rng()
    fft_len = 2*sig_len
    signal = rng.normal(size=(num_ch1, num_ch2, num_ch3, sig_len))
    fft_signal = ft.rfft(signal, n = fft_len)
    ifft_signal = ft.irfft(fft_signal)[...,:sig_len]
    assert np.allclose(signal, ifft_signal)



@hyp.settings(deadline=None)
@hyp.given(num_ch1 = st.integers(min_value=1, max_value=3),
            num_ch2 = st.integers(min_value=1, max_value=3), 
            num_ch3 = st.integers(min_value=1, max_value=3), 
            fft_len = st.integers(min_value=16, max_value=256))
def test_rfft_equals_first_half_of_fft(num_ch1, num_ch2, num_ch3, fft_len):
    rng = np.random.default_rng()
    signal = rng.normal(size=(num_ch1, num_ch2, num_ch3, fft_len))
    fft_signal = ft.fft(signal)
    rfft_signal = ft.rfft(signal)
    assert np.allclose(fft_signal[:rfft_signal.shape[0],...], rfft_signal)



@hyp.settings(deadline=None)
@hyp.given(num_ch1 = st.integers(min_value=1, max_value=3),
            sig_len = st.integers(min_value=16, max_value=256))
def test_convolution_with_rfft_gives_real_valued_output(num_ch1, sig_len):
    rng = np.random.default_rng()
    fft_len = 2*sig_len
    signal = rng.normal(size=(num_ch1, sig_len))
    fft_signal = ft.rfft(signal, n = fft_len)
    freq_filter = rng.normal(size = fft_signal.shape) + 1j * rng.normal(size = fft_signal.shape)
    freq_filter[0,:] = np.real(freq_filter[0,:])
    freq_filter[-1,:] = np.real(freq_filter[-1,:])

    filtered_sig = fft_signal * freq_filter
    ifft_signal = ft.irfft(filtered_sig)[...,:sig_len]
    assert np.allclose(0, np.imag(ifft_signal))


@hyp.settings(deadline=None)
@hyp.given(num_ch1 = st.integers(min_value=1, max_value=3),
            num_ch2 = st.integers(min_value=1, max_value=3), 
            num_ch3 = st.integers(min_value=1, max_value=3), 
            fft_len = st.integers(min_value=16, max_value=256))
def test_multiplying_by_dft_vector_gives_same_result_as_fft(num_ch1, num_ch2, num_ch3, fft_len):
    rng = np.random.default_rng()
    signal = rng.normal(size=(num_ch1, num_ch2, num_ch3, fft_len))
    fft_signal = ft.fft(signal)
    for n in range(fft_len):
        manual_dft = np.sum(signal * ft.dft_vector(n, fft_len)[None,None,None,:], axis=-1)
        assert np.allclose(fft_signal[n,...], manual_dft)
    
@hyp.settings(deadline=None)
@hyp.given(num_ch1 = st.integers(min_value=1, max_value=3),
            num_ch2 = st.integers(min_value=1, max_value=3), 
            num_ch3 = st.integers(min_value=1, max_value=3), 
            fft_len = st.integers(min_value=16, max_value=256))
def test_multiplying_by_idft_vector_gives_same_result_as_ifft(num_ch1, num_ch2, num_ch3, fft_len):
    rng = np.random.default_rng()
    freq_signal = rng.normal(size=(fft_len, num_ch1, num_ch2, num_ch3))
    signal = ft.ifft(freq_signal)
    for n in range(fft_len):
        manual_idft = np.sum(freq_signal * ft.idft_vector(n, fft_len)[:,None,None,None], axis=0)
        assert np.allclose(signal[...,n], manual_idft)





@hyp.settings(deadline=None)
@hyp.given(samplerate = st.integers(min_value=1, max_value=128),
           fft_len = st.integers(min_value=1, max_value=128))
def test_get_real_freqs_is_equivalent_to_np_rfftfreq(samplerate, fft_len):
    #ng = np.random.default_rng()
    freqs = ft.get_real_freqs(fft_len, samplerate)
    np_freqs = np.fft.rfftfreq(fft_len, 1/samplerate)
    assert np.allclose(freqs, np_freqs)

@hyp.settings(deadline=None)
@hyp.given(
    st.integers(min_value=1, max_value=32),
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=1, max_value=5),
)
def test_freq_time_domain_convolution_is_equal(ir_len, num_in, num_out, num_blocks):
    rng = np.random.default_rng()

    signal = rng.normal(0, 1, size = (num_in, num_blocks * ir_len))
    ir = rng.normal(0, 1, size = (num_in, num_out, ir_len))

    filt_td = fc.create_filter(ir)
    filt_fd = np.moveaxis(ft.fft(np.concatenate((ir, np.zeros_like(ir)), axis=-1)), 1,2)
    sig_init = rng.normal(0, 1, size=(num_in, ir_len))
    filt_td.process(sig_init)
    input_fd = np.concatenate((sig_init, signal), axis=-1)

    out_td = np.zeros((num_out, ir_len * num_blocks))
    out_fd = np.zeros((num_out, ir_len * num_blocks))
    for i in range(num_blocks):
        out_td[:, i * ir_len : (i + 1) * ir_len] = filt_td.process(
            signal[:, i * ir_len : (i + 1) * ir_len]
        )
        out_fd[:, i * ir_len : (i + 1) * ir_len] = ft.convolve_sum(
            filt_fd, input_fd[:, i * ir_len : (i + 2) * ir_len]
        )

    assert np.allclose(out_fd, out_td)


def test_rdft_mat_is_equivalent_to_rfft():
    rng = np.random.default_rng()
    num_to_remove_low = rng.integers(0, 10)
    num_to_remove_high = rng.integers(0, 10)
    dft_len = rng.integers(50, 100)

    signal = rng.normal(size=(1, dft_len))
    fft_signal = ft.rfft(signal, removed_freqs=(num_to_remove_low, num_to_remove_high))
    rdft_mat = ft.rdft_mat(dft_len, removed_freqs=(num_to_remove_low, num_to_remove_high))
    fft_signal_mat = (rdft_mat @ signal.T)
    assert np.allclose(fft_signal, fft_signal_mat)

def test_rdft_mat_is_equivalent_to_rfft_removed_only_low():
    rng = np.random.default_rng()
    num_to_remove = rng.integers(0, 10)
    dft_len = rng.integers(50, 100)

    signal = rng.normal(size=(1, dft_len))
    fft_signal = ft.rfft(signal, removed_freqs=num_to_remove)
    rdft_mat = ft.rdft_mat(dft_len, removed_freqs=num_to_remove)
    fft_signal_mat = (rdft_mat @ signal.T)
    assert np.allclose(fft_signal, fft_signal_mat)

def test_irdft_mat_and_real_part_operator_is_equivalent_to_irfft():
    rng = np.random.default_rng()
    num_to_remove = rng.integers(0, 10)
    dft_len = rng.integers(50, 100)
    num_freqs = dft_len // 2 + 1 - num_to_remove

    signal = rng.normal(size=(num_freqs, 1)) + 1j * rng.normal(size=(num_freqs, 1))

    B = ft.irdft_mat(dft_len, num_freqs_removed_low=num_to_remove)
    irfft_signal = ft.irfft(signal, removed_freqs=num_to_remove)
    irfft_signal_mat = np.real(B @ signal).T
    assert np.allclose(irfft_signal, irfft_signal_mat)



def test_time_domain_inner_product_is_equivalent_to_freq_domain_real_inner_product():
    rng = np.random.default_rng()
    vec_len = rng.integers(1, 100)

    signal1 = rng.normal(size=(vec_len,))
    signal2 = rng.normal(size=(vec_len,))
    inner_td = np.sum(signal1 * signal2)

    sig1_f = ft.rfft(signal1)
    sig2_f = ft.rfft(signal2)
    C = ft.rdft_weighting(sig1_f.shape[0], vec_len)
    inner_fd = np.sum(np.real(sig1_f * np.conj(sig2_f) * C))

    assert np.allclose(inner_td, inner_fd)

def test_time_domain_inner_product_is_equivalent_to_freq_domain_real_inner_product_backwards_transform():
    rng = np.random.default_rng()
    vec_len = 2*rng.integers(1, 100)
    num_real_freqs = vec_len // 2 + 1

    sig1_f = rng.normal(size=(num_real_freqs,)).astype(complex)
    sig1_f[1:-1] += 1j * rng.normal(size=(num_real_freqs-2,))
    sig2_f = rng.normal(size=(num_real_freqs,)).astype(complex)
    sig2_f[1:-1] += 1j * rng.normal(size=(num_real_freqs-2,))

    C = ft.rdft_weighting(sig1_f.shape[0], vec_len)
    inner_fd = np.sum(np.real(sig1_f * np.conj(sig2_f) * C))

    signal1 = ft.irfft(sig1_f)
    signal2 = ft.irfft(sig2_f)
    inner_td = np.sum(signal1 * signal2)

    assert np.allclose(inner_td, inner_fd)

def test_time_domain_inner_product_is_equivalent_to_freq_domain_real_inner_product_backwards_transform_with_removed_frequencies():
    rng = np.random.default_rng()
    dft_len = rng.integers(50, 100)

    num_real_freqs = dft_len // 2 + 1
    removed =  (rng.integers(0, 10), rng.integers(0, 10))
    dft_vec_len = num_real_freqs - np.sum(removed)

    sig1_f = rng.normal(size=(dft_vec_len,)) + 1j * rng.normal(size=(dft_vec_len,))
    sig2_f = rng.normal(size=(dft_vec_len,)) + 1j * rng.normal(size=(dft_vec_len,))

    C = ft.rdft_weighting(dft_vec_len, dft_len, removed_freqs=removed)
    inner_fd = np.sum(np.real(sig1_f * np.conj(sig2_f) * C))

    signal1 = ft.irfft(sig1_f, n=dft_len, removed_freqs=removed)
    signal2 = ft.irfft(sig2_f, n=dft_len, removed_freqs=removed)
    inner_td = np.sum(signal1 * signal2)

    assert np.allclose(inner_td, inner_fd)



def test_calculation_of_real_vec_len_to_dft_len():
    rng = np.random.default_rng()
    dft_len = rng.integers(50, 100)
    even = dft_len % 2 == 0
    removed_freqs = (rng.integers(0, 10), rng.integers(0, 10))
    sig = rng.normal(size=(dft_len,))
    sig_f = ft.rfft(sig, removed_freqs=removed_freqs)

    real_vec = ft.dft_domain_to_real_vec(sig_f, even=even, scale = True, removed_freqs=removed_freqs)
    dft_len_calc = dft_module._real_vec_len_to_dft_len(real_vec.shape[-1], even=even, removed_freqs=removed_freqs)

    assert dft_len == dft_len_calc


def test_dft_domain_to_real_vec_is_invertible():
    rng = np.random.default_rng()
    td_len = rng.integers(50, 100)
    even = td_len % 2 == 0
    remove_freqs_low = rng.integers(0, 10)
    remove_freqs_high = rng.integers(0, 10)
    removed_freqs = (remove_freqs_low, remove_freqs_high)

    signal1 = rng.normal(size=(td_len,))

    test_sig = ft.dft_domain_to_real_vec(ft.rfft(signal1, removed_freqs=removed_freqs), even=even, scale = True, removed_freqs=removed_freqs)
    real_len = test_sig.shape[-1]

    signal = np.ones(real_len) #rng.normal(size=(real_len,))
    sig_dft = ft.real_vec_to_dft_domain(signal, even = even, scale=True, removed_freqs=removed_freqs)
    signal_inverted = ft.dft_domain_to_real_vec(sig_dft, even=even, scale=True, removed_freqs=removed_freqs)

    assert np.allclose(signal, np.squeeze(signal_inverted))

def test_dft_domain_inner_product_is_equivalent_to_real_inner_product():
    rng = np.random.default_rng()
    vec_len = rng.integers(50, 100)
    even = vec_len % 2 == 0
    remove_freqs = (rng.integers(0, 10), rng.integers(0, 10))

    signal1 = rng.normal(size=(vec_len,))
    signal2 = rng.normal(size=(vec_len,))
    #inner_td = np.sum(signal1 * signal2)

    sig1_f = ft.rfft(signal1, removed_freqs=remove_freqs)
    sig2_f = ft.rfft(signal2, removed_freqs=remove_freqs)
    C = ft.rdft_weighting(sig1_f.shape[0], vec_len, removed_freqs=remove_freqs)
    inner_fd = np.sum(np.real(sig1_f * np.conj(sig2_f) * C))

    sig1 = ft.dft_domain_to_real_vec(sig1_f, even = even, scale=True, removed_freqs=remove_freqs)
    sig2 = ft.dft_domain_to_real_vec(sig2_f, even = even, scale=True, removed_freqs=remove_freqs)
    inner_real = np.sum(sig1 * sig2)

    assert np.allclose(inner_real, inner_fd)


def test_dft_domain_inner_product_is_equivalent_to_real_inner_product_backwards_transform():
    rng = np.random.default_rng()
    td_len = rng.integers(50, 100)
    even = td_len % 2 == 0
    removed_freqs = (rng.integers(0, 10), rng.integers(0, 10))

    signal1 = rng.normal(size=(td_len,))
    test_sig = ft.dft_domain_to_real_vec(ft.rfft(signal1, removed_freqs=removed_freqs), even=even, scale=True, removed_freqs=removed_freqs)
    real_len = test_sig.shape[-1]

    signal_real1 = rng.normal(size=(real_len,))
    signal_real2 = rng.normal(size=(real_len,))

    signal_dft1 = ft.real_vec_to_dft_domain(signal_real1, even = even, scale=True, removed_freqs=removed_freqs)
    signal_dft2 = ft.real_vec_to_dft_domain(signal_real2, even = even, scale=True, removed_freqs=removed_freqs)

    #sig2_f = ft.rfft(signal2, num_freqs_removed_low=remove_freqs)
    C = ft.rdft_weighting(signal_dft1.shape[0], td_len, removed_freqs=removed_freqs)
    inner_fd = np.sum(np.real(signal_dft1 * np.conj(signal_dft2) * C[:,None]))

    inner_real = np.sum(signal_real1 * signal_real2)

    assert np.allclose(inner_real, inner_fd)