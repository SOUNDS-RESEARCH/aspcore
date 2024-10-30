import numpy as np
import scipy.signal as spsig

import aspcore.filterdesign as fd

import hypothesis as hyp
import hypothesis.strategies as st

@hyp.settings(deadline=None)
@hyp.given(half_ir_len = st.integers(min_value=16, max_value=256))
def test_fir_from_frequency_response_always_gives_real_valued_filter(half_ir_len):
    rng = np.random.default_rng(123456)
    samplerate = 100
    def freq_resp(f):
        return rng.uniform(-1, 1, size=f.shape) + 1j * rng.uniform(-1, 1, size=f.shape)

    ir_len = half_ir_len * 2 + 1
    ir = fd.fir_from_frequency_function(freq_resp, ir_len, samplerate)

    # fig, ax = plt.subplots(1,1)
    # ax.plot(np.real(ir))
    # ax.plot(np.imag(ir))
    # plt.show()

    assert np.allclose(0, np.imag(ir)), "The impulse response should be real valued"

@hyp.settings(deadline=None)
@hyp.given(half_ir_len = st.integers(min_value=16, max_value=256))
def test_fir_from_frequency_response_gives_correct_group_delay(half_ir_len):
    ir_len = 2 * half_ir_len + 1
    samplerate = 100
    def freq_resp(f):
       return np.ones_like(f)

    ir = fd.fir_from_frequency_function(freq_resp, ir_len, samplerate)

    # fig, ax = plt.subplots(1,1)
    # ax.plot(ir)
    
    w, gd = spsig.group_delay((ir, 1), 512)
    # fig, ax = plt.subplots(1,1)
    # ax.plot(w, gd)
    # plt.show()

    assert np.allclose(np.median(gd), half_ir_len)

