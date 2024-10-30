import numpy as np
import hypothesis as hyp
import hypothesis.strategies as st
import aspcore.filter as fc
import aspcore.utilities as util

def test_hardcoded_filtersum():
    ir = np.vstack((np.sin(np.arange(5)), np.cos(np.arange(5))))
    filt1 = fc.create_filter(ir=ir[:, None, :])

    inSig = np.array([[10, 9, 8, 7, 6, 5], [4, 5, 4, 5, 4, 5]])
    out = filt1.process(inSig)

    hardcodedOut = [
        [4.0, 15.57591907, 21.70313731, 17.44714985, 4.33911864, 3.58393245]
    ]
    assert np.allclose(out, hardcodedOut)


@hyp.settings(deadline=None)
@hyp.given(
    ir_len = st.integers(min_value=1, max_value=8),
    num_samples = st.integers(min_value=1, max_value=32),
)
def test_filtersum_ending_zeros_does_not_affect_output(ir_len, num_samples):
    ir2 = np.zeros((1, 1, ir_len))
    ir2[0, 0, 0] = 1
    filt = fc.create_filter(ir2)

    in_sig = np.random.rand(1, num_samples)
    out = filt.process(in_sig)
    assert np.allclose(in_sig, out)



@hyp.settings(deadline=None)
@hyp.given(
    st.integers(min_value=1, max_value=32),
    st.integers(min_value=1, max_value=4),
    st.integers(min_value=1, max_value=4),
    st.integers(min_value=1, max_value=5),
)
def test_freq_time_filter_sum_equal_results(ir_len, num_in, num_out, num_blocks):
    ir = np.random.standard_normal((num_in, num_out, ir_len))
    filt_td = fc.create_filter(ir=ir)
    filt_fd = fc.FilterSumFreq(ir=ir)

    out_td = np.zeros((num_out, num_blocks * ir_len))
    out_fd = np.zeros((num_out, num_blocks * ir_len))

    signal = np.random.standard_normal((num_in, num_blocks * ir_len))
    for i in range(num_blocks):
        out_fd[:, i * ir_len : (i + 1) * ir_len] = filt_fd.process(
            signal[:, i * ir_len : (i + 1) * ir_len]
        )
        out_td[:, i * ir_len : (i + 1) * ir_len] = filt_td.process(
            signal[:, i * ir_len : (i + 1) * ir_len]
        )
    assert np.allclose(out_td, out_fd)

@hyp.settings(deadline=None)
@hyp.given(
    st.integers(min_value=1, max_value=16),
    st.integers(min_value=1, max_value=3),
    st.tuples(st.integers(min_value=1, max_value=3), st.integers(min_value=1, max_value=3)),
    st.integers(min_value=1, max_value=3),
)
def test_freq_time_md_filter_equal_results(ir_len, data_dim, filt_dim, num_blocks):
    ir = np.random.standard_normal((*filt_dim, ir_len))
    filt_td = fc.create_filter(ir, broadcast_dim=data_dim, sum_over_input=False)
    filt_fd = fc.FilterBroadcastFreq(data_dim, ir=ir)

    out_td = np.zeros((*filt_dim, data_dim, num_blocks * ir_len))
    out_fd = np.zeros((*filt_dim, data_dim, num_blocks * ir_len))

    signal = np.random.standard_normal((data_dim, num_blocks * ir_len))
    for i in range(num_blocks):
        out_fd[..., i * ir_len : (i + 1) * ir_len] = filt_fd.process(
            signal[..., i * ir_len : (i + 1) * ir_len]
        )
        out_td[..., i * ir_len : (i + 1) * ir_len] = filt_td.process(
            signal[..., i * ir_len : (i + 1) * ir_len]
        )
    assert np.allclose(out_td, out_fd)





@hyp.settings(deadline=None)
@hyp.given(num_ch = st.integers(min_value=1, max_value=5),
            num_out = st.integers(min_value=1, max_value=5), 
            nfft_exp = st.integers(min_value=3, max_value=7))
def test_wola_perfect_reconstruction(num_ch, num_out, nfft_exp):
    block_size = 2 ** nfft_exp
    overlap = block_size // 2
    num_blocks = 10
    #block_size = 32
    num_samples = num_blocks * block_size

    #src = sources.WhiteNoiseSource(num_ch,1)
    rng = np.random.default_rng()
    
    wola = fc.WOLA(num_ch, num_out, block_size, overlap)

    #sig = src.get_samples(num_samples)
    sig = rng.normal(0, 1, size=(num_ch, num_samples))
    sig_out = np.zeros((num_ch, num_out, num_samples))

    for i in util.block_process_idxs(num_samples, wola.hop, 0):
        wola.analysis(sig[:,i:i+wola.hop])
        sig_out[:,:,i:i+wola.hop] = wola.synthesis()

    sig_in = sig[:,overlap:-block_size]
    sig_out = sig_out[:,:,overlap:-block_size]

    for i in range(num_out):
        assert np.allclose(sig_in[:,:-wola.hop], sig_out[:,i,wola.hop:])
