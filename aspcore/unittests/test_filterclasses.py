import numpy as np
import hypothesis as hyp
import hypothesis.strategies as st
import aspcore.filterclasses as fc

# @pytest.fixture
# def setupconstants():
#     pos = preset.getPositionsCylinder3d()
#     sr = int(1000 + np.random.rand() * 8000)
#     noiseFreq = int(100 + np.random.rand() * 800)
#     return pos, sr, noiseFreq


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
    ir_len = st.integers(min_value=1, max_value=8),
    num_in = st.integers(min_value=1, max_value=4),
    num_out = st.integers(min_value=1, max_value=4),
    num_samples = st.integers(min_value=1, max_value=32),
)
def test_filtersum_dynamic_with_static_ir_equals_normal_filtersum(ir_len, num_in, num_out, num_samples):
    rng = np.random.default_rng()
    ir = rng.normal(0, 1, size=(num_in,num_out,ir_len))

    filt_dyn = fc.FilterSumDynamic(ir)
    filt = fc.create_filter(ir)

    out = np.zeros((num_out, num_samples))
    out_dyn = np.zeros((num_out, num_samples))

    sig = rng.normal(0, 1, size=(num_in, num_samples))
    out_dyn = filt_dyn.process(sig)
    out = filt.process(sig)
    # for i in range(num_samples):
    #     out[:,i:i+1] = filt.process(sig[:,i:i+1])
    #     out_dyn[:,i:i+1] = filt_dyn.process(sig[:,i:i+1])
    assert np.allclose(out, out_dyn)


@hyp.settings(deadline=None)
@hyp.given(
    num_in = st.integers(min_value=1, max_value=4),
    num_out = st.integers(min_value=1, max_value=4),
    num_samples = st.integers(min_value=1, max_value=32),
)
def test_filtersum_dynamic_no_output_with_increasing_delay(num_in, num_out, num_samples):
    ir_len = num_samples + 1
    rng = np.random.default_rng()
    
    ir = np.zeros((num_in,num_out, ir_len))
    filt_dyn = fc.FilterSumDynamic(ir)

    out = np.zeros((num_out, num_samples))

    sig = rng.normal(0, 1, size=(num_in, num_samples))
    for i in range(num_samples):
        ir = np.zeros((num_in, num_out, ir_len))
        ir[:,:,i+1:] = 1
        filt_dyn.update_ir(ir)
        out[:,i:i+1] = filt_dyn.process(sig[:,i:i+1])

    assert np.allclose(np.sum(np.abs(out)), 0)




@hyp.settings(deadline=None)
@hyp.given(
    ir_len = st.integers(min_value=1, max_value=8),
    num_in = st.integers(min_value=1, max_value=4),
    num_out = st.integers(min_value=1, max_value=4),
    num_samples = st.integers(min_value=2, max_value=32),
)
def test_filtersum_dynamic_piecewise_static_ir_equals_normal_filtersum(ir_len, num_in, num_out, num_samples):
    rng = np.random.default_rng()
    ir1 = rng.normal(0, 1, size=(num_in,num_out,ir_len))
    ir2 = rng.normal(0, 1, size=(num_in,num_out,ir_len))
    change_idx = num_samples // 2

    filt_dyn = fc.FilterSumDynamic(ir1)
    filt1 = fc.create_filter(ir1)
    filt2 = fc.create_filter(ir2)

    out = np.zeros((num_out, num_samples))
    out_dyn = np.zeros((num_out, num_samples))

    sig = rng.normal(0, 1, size=(num_in, num_samples))
    out[:,:change_idx] = filt1.process(sig[:,:change_idx])
    filt2.process(sig[:,:change_idx])
    out[:,change_idx:] = filt2.process(sig[:,change_idx:])

    for i in range(num_samples):
        out_dyn[:,i:i+1] = filt_dyn.process(sig[:,i:i+1])
        if i == change_idx-1:
            filt_dyn.update_ir(ir2)

    assert np.allclose(out, out_dyn)



@hyp.settings(deadline=None)
@hyp.given(
    ir_len = st.integers(min_value=1, max_value=8),
    num_in = st.integers(min_value=1, max_value=4),
    num_out = st.integers(min_value=1, max_value=4),
    num_samples = st.integers(min_value=2, max_value=32),
)
def test_filtersum_dynamic_zero_ir_equals_static_filter_with_shorter_input_signal(ir_len, num_in, num_out, num_samples):


    assert False # not implemented




@hyp.settings(deadline=None)
@hyp.given(
    ir_len = st.integers(min_value=1, max_value=8),
    num_in = st.integers(min_value=1, max_value=4),
    num_out = st.integers(min_value=1, max_value=4),
    num_samples = st.integers(min_value=2, max_value=32),
)
def test_filtersum_dynamic_equals_direct_implementation_of_definition(ir_len, num_in, num_out, num_samples):

    for n in range(num_samples):
        out[:,n] = h[i,n] * in_sig[:,n-i]
    assert False # not implemented


@hyp.settings(deadline=None)
@hyp.given(
    st.integers(min_value=1, max_value=256),
    st.integers(min_value=1, max_value=8),
    st.integers(min_value=1, max_value=8),
    st.integers(min_value=1, max_value=10),
)
def test_freq_time_filter_sum_equal_results(irLen, numIn, numOut, numBlocks):
    ir = np.random.standard_normal((numIn, numOut, irLen))
    # ir = np.zeros((numIn, numOut, irLen))
    # ir[:,:,0] = 1
    tdFilt = fc.create_filter(ir=ir)
    fdFilt = fc.FilterSumFreq(ir=ir)

    tdOut = np.zeros((numOut, numBlocks * irLen))
    fdOut = np.zeros((numOut, numBlocks * irLen))

    signal = np.random.standard_normal((numIn, numBlocks * irLen))
    for i in range(numBlocks):
        fdOut[:, i * irLen : (i + 1) * irLen] = fdFilt.process(
            signal[:, i * irLen : (i + 1) * irLen]
        )
        tdOut[:, i * irLen : (i + 1) * irLen] = tdFilt.process(
            signal[:, i * irLen : (i + 1) * irLen]
        )
    assert np.allclose(tdOut, fdOut)

@hyp.settings(deadline=None)
@hyp.given(
    st.integers(min_value=1, max_value=16),
    st.integers(min_value=1, max_value=3),
    st.tuples(st.integers(min_value=1, max_value=3), st.integers(min_value=1, max_value=3)),
    st.integers(min_value=1, max_value=3),
)
def test_freq_time_md_filter_equal_results(irLen, dataDim, filtDim, numBlocks):
    ir = np.random.standard_normal((*filtDim, irLen))
    tdFilt = fc.create_filter(ir, broadcast_dim=dataDim, sum_over_input=False)
    fdFilt = fc.FilterBroadcastFreq(dataDim, ir=ir)

    tdOut = np.zeros((*filtDim, dataDim, numBlocks * irLen))
    fdOut = np.zeros((*filtDim, dataDim, numBlocks * irLen))

    signal = np.random.standard_normal((dataDim, numBlocks * irLen))
    for i in range(numBlocks):
        fdOut[..., i * irLen : (i + 1) * irLen] = fdFilt.process(
            signal[..., i * irLen : (i + 1) * irLen]
        )
        tdOut[..., i * irLen : (i + 1) * irLen] = tdFilt.process(
            signal[..., i * irLen : (i + 1) * irLen]
        )
    assert np.allclose(tdOut, fdOut)


# def test_same_output():
#     numIn = 10
#     numOut = 10
#     irLen = 4096
#     sigLen = 4096
#     ir = np.random.standard_normal((numIn, numOut, irLen))

#     sig = np.random.standard_normal((numIn, sigLen))
#     newFilt = fcn.FilterSum(ir = ir)
#     oldFilt = FilterSum(ir = ir)

#     s = time.time()
#     newOut = newFilt.process(sig)
#     print("new algo: ", time.time()-s)
#     s = time.time()
#     oldOut = oldFilt.process(sig)
#     print("old algo: ", time.time()-s)
#     #assert np.allclose(newOut, oldOut)
#     assert False

# def test_incremental_filtering():
#     numIn = 5
#     numOut = 5
#     irLen = 1024
#     sigLen = 1024
#     ir = np.random.standard_normal((numIn, numOut, irLen))
#     sig = np.random.standard_normal((numIn, sigLen))

#     incrFilt = fcn.FilterSum(ir = ir)
#     fullFilt = fcn.FilterSum(ir = ir)

#     incrementalOut = np.zeros((numOut, sigLen))
#     fullOut = np.zeros((numOut, sigLen))

#     s = time.time()
#     for i in range(sigLen):
#         incrementalOut[:,i:i+1] = incrFilt.process(sig[:,i:i+1])
#     print("Incremental algo time: ", time.time()-s)

#     s = time.time()
#     fullOut[:,:] = fullFilt.process(sig)
#     print("Full algo time: ", time.time()-s)

#     assert np.allclose(fullOut, incrementalOut)
#     #assert False


# def test_filt_time():
#     numIn = 5
#     numOut = 5
#     irLen = 1024
#     sigLen = 1024
#     ir = np.random.standard_normal((numIn, numOut, irLen))
#     sig = np.random.standard_normal((numIn, sigLen))

#     newFilt = fcn.FilterSum(ir = ir)
#     oldFilt = FilterSum(ir = ir)

#     newOut = np.zeros((numOut, sigLen))
#     oldOut = np.zeros((numOut, sigLen))

#     s = time.time()
#     for i in range(sigLen):
#         oldOut[:,i:i+1] = oldFilt.process(sig[:,i:i+1])
#     print("old algo time: ", time.time()-s)

#     s = time.time()
#     for i in range(sigLen):
#         newOut[:,i:i+1] = newFilt.process(sig[:,i:i+1])
#     print("new algo time: ", time.time()-s)
#     #assert np.allclose(oldOut, newOut)
#     assert False
