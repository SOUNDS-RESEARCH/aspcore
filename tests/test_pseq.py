import aspcore.pseq as pseq

def test_standard_pseq_is_actually_a_pseq():
    seq_len = 256
    signal = pseq.create_pseq(seq_len)
    info = pseq.verify_pseq(signal, plot=False)
    assert info["Perfect periodic autocorrelation"]

    seq_len = 257
    signal = pseq.create_pseq(seq_len)
    info = pseq.verify_pseq(signal, plot=False)
    assert info["Perfect periodic autocorrelation"]


def test_random_phase_pseq_is_actually_a_pseq():
    seq_len = 256
    signal = pseq.create_pseq_random_phase(seq_len)
    info = pseq.verify_pseq(signal, plot=False)
    assert info["Perfect periodic autocorrelation"]

    seq_len = 257
    signal = pseq.create_pseq_random_phase(seq_len)
    info = pseq.verify_pseq(signal, plot=False)
    assert info["Perfect periodic autocorrelation"]