import czt as czt_package

import aspcore.fouriertransform as ft
import aspcore.fouriertransform.chirp_z_transform as czt

import numpy as np
import scipy.signal as spsig
import matplotlib.pyplot as plt



def test_czt_followed_by_inverse_czt_returns_original_signal():
    rng = np.random.default_rng()
    num_channels = 3
    num_samples = 100
    signal = rng.normal(0, 1, size=(num_channels, num_samples))

    w = rng.normal(0, 1) + 1j * rng.normal(0, 1)
    w = w / np.abs(w)
    a = rng.normal(0, 1) + 1j * rng.normal(0, 1)
    a = a / np.abs(a)

    freq_signal = ft.czt(signal, num_samples, w = w, a = a)
    time_signal = ft.iczt(freq_signal, num_samples, w = w, a = a)

    mean_error = np.mean(np.abs(signal - np.real(time_signal)))
    assert mean_error < 1e-10

def test_czt_equals_scipy_czt_implementation_when_time_convention_is_compensated():
    rng = np.random.default_rng()
    num_channels = 3
    num_samples = 100
    signal = rng.normal(0, 1, size=(num_channels, num_samples))
    
    freq_signal = ft.czt(signal, num_samples)
    freq_signal_2 = np.moveaxis(spsig.czt(signal, num_samples, axis=-1), -1, 0)

    assert np.allclose(freq_signal, np.conj(freq_signal_2))

def test_iczt_on_unit_circle_can_be_computed_using_conjugates():
    rng = np.random.default_rng()
    num_channels = 3
    num_samples = 100
    signal = rng.normal(0, 1, size=(num_channels, num_samples))

    a = rng.normal(0, 1) + 1j * rng.normal(0, 1)
    a = a / np.abs(a)

    freq_signal = ft.czt(signal, num_samples, a = a)

    modified_freq_signal = np.moveaxis(np.conj(freq_signal), 0, -1)
    czt_output = ft.czt(modified_freq_signal, num_samples, a = a)
    time_signal = np.moveaxis(np.conj(czt_output), 0, -1) / num_samples

    mean_error = np.mean(np.abs(signal - np.real(time_signal)))
    assert mean_error < 1e-10














def test_czt_unit_circle_equals_czt_with_unit_parameters():
    rng = np.random.default_rng()
    num_channels = 3
    num_samples = 100
    signal = rng.normal(0, 1, size=(num_channels, num_samples))

    a_angle = rng.uniform(0, 1)
    w_angle = rng.uniform(0, 1)
    a = np.exp(2j * np.pi * a_angle)
    w = np.exp(2j * np.pi * w_angle)

    signal_czt = ft.czt(signal, num_samples, w, a)
    signal_zoom_dft = ft.czt_unit_circle(signal, num_samples, w_angle, a_angle)

    mean_error = np.mean(np.abs(signal_czt - signal_zoom_dft))
    assert mean_error < 1e-10

def test_czt_unit_circle_followed_by_iczt_unit_circle_equals_original_signal():
    for i in range(100):
        rng = np.random.default_rng()
        num_channels = 3
        num_samples = 100
        signal = rng.normal(0, 1, size=(num_channels, num_samples))

        a_angle = rng.uniform(0, 1)
        w_angle = rng.uniform(0, 1)
        
        signal_freq = ft.czt_unit_circle(signal, num_samples, w_angle, a_angle)
        signal_time = ft.iczt_unit_circle(signal_freq, num_samples, w_angle, a_angle)

        mean_error = np.mean(np.abs(signal_time - signal))
        assert mean_error < 1e-10


def test_zoom_dft_followed_by_zoom_idft_equals_original_signal():
    rng = np.random.default_rng()
    N = M = 1000
    num_channels = 3
    signal = rng.normal(0, 1, size=(num_channels, N))

    samplerate = 1000
    freq_limits = (20, 1000)

    freq_signal = ft.zoom_dft(signal, M, freq_limits, samplerate)
    time_signal = ft.zoom_idft(freq_signal, M, freq_limits, samplerate)

    mean_error = np.mean(np.abs(signal - time_signal))
    plt.plot(np.fft.rfft(signal[0,:]))
    plt.plot(np.fft.rfft(time_signal[0,:]))
    plt.show()
    assert mean_error < 1e-10

def test_czt_unit_circle_matrix_equals_czt_matrix_with_unit_parameters():
    rng = np.random.default_rng()
    num_samples = 100
    M = 100
    a_angle = rng.uniform(0, 1) # 0.2 #
    w_angle = rng.uniform(0, 1) # 0.1 + 1 / num_samples #
    a = np.exp(2j * np.pi * a_angle)
    w = np.exp(2j * np.pi * w_angle)

    czt_matrix = czt._czt_matrix(num_samples, M, w, a)
    zoom_dft_matrix = czt._czt_unit_circle_matrix(num_samples, M, w_angle, a_angle)

    mean_error = np.mean(np.abs(czt_matrix - zoom_dft_matrix))
    assert mean_error < 1e-10





def test_czt_matrix_is_invertible():
    rng = np.random.default_rng()
    czt_len = 100

    fig, axes = plt.subplots(3, 1, figsize=(15, 7))
    for i in range(10000):
        a_angle = rng.uniform(0, 1) # 0.2 #
        w_angle = np.max((rng.uniform(0, 1), 1e-1))

        zoom_dft_matrix = czt._czt_unit_circle_matrix(czt_len, czt_len, w_angle, a_angle)
        eigvals = np.linalg.eigvalsh(zoom_dft_matrix)
        axes[0].plot(eigvals)
        axes[0].set_title("Eigenvalues")

        sorted_abs_eigvals = np.sort(np.abs(eigvals))
        axes[1].plot(sorted_abs_eigvals)
        axes[1].set_title("Abs eigenvalues")

        axes[2].plot(10 * np.log10(sorted_abs_eigvals))
        axes[2].set_title("Abs eigenvalues (dB)")
    plt.show()
    # add some reasonable check here, that will not be too random.


def test_show_czt_unit_circle_matrix_condition_numbers():
    rng = np.random.default_rng()
    num_samples = 100
    M = 100

    w_angles = np.logspace(-3, 0, 3000)
    a_angle = 0
    conds = []
    for w_angle in w_angles:
    #a_angle = rng.uniform(0, 1) # 0.2 #
    #w_angle = rng.uniform(0, 1) # 0.1 + 1 / num_samples #
        a = np.exp(2j * np.pi * a_angle)
        w = np.exp(2j * np.pi * w_angle)

        zoom_dft_matrix = czt._czt_unit_circle_matrix(num_samples, M, w_angle, a_angle)
        conds.append(np.linalg.cond(zoom_dft_matrix))

    fig, ax = plt.subplots()
    ax.plot(w_angles, np.log10(conds), )
    ax.set_xlabel("w_angle")
    ax.set_ylabel("log10(cond)")
    ax.set_xscale('log')
    plt.show()
    #mean_error = np.mean(np.abs(czt_matrix - zoom_dft_matrix))
    #assert mean_error < 1e-10


def test_czt_unit_circle_matrix_is_invertible():
    pass


def test_zoom_dft_matrix_is_well_conditioned_for_reasonable_sound_field_parameters():
    rng = np.random.default_rng()
    N = M = 100
    num_channels = 3
    signal = rng.normal(0, 1, size=(num_channels, N))

    samplerate = 1000
    conds = []

    limit_offsets = np.linspace(0, 50, 100)
    for l in limit_offsets:
        freq_limits = (l, samplerate-l)
        
        a_angle = freq_limits[0] / samplerate
        w_angle = (freq_limits[1] - freq_limits[0]) / (samplerate * M)
        #ft._zoom_dft_matrix(signal, M, freq_limits, samplerate)
        mat = czt._czt_unit_circle_matrix(N, M, w_angle, a_angle)
        conds = np.append(conds, np.linalg.cond(mat))
    #print(f"Condition number: {np.linalg.cond(mat)}")
    plt.plot(limit_offsets, np.log10(conds))
    plt.show()
    freq_signal = ft.zoom_dft(signal, M, freq_limits, samplerate)
    time_signal = ft.zoom_idft(freq_signal, M, freq_limits, samplerate)

    mean_error = np.mean(np.abs(signal - time_signal))
    assert mean_error < 1e-10





def test_zoom_dft_equals_scipy_zoom_fft_after_time_convention_compensation():
    rng = np.random.default_rng()
    N = M = 100
    num_channels = 1
    signal = rng.normal(0, 1, size=(num_channels, N))

    samplerate = 1000
    freq_limits = (0, 500)

    freq_signal = ft.zoom_dft(signal, M, freq_limits, samplerate)
    freq_signal_2 = spsig.zoom_fft(signal, freq_limits, axis=-1, fs=samplerate).T

    mean_error = np.mean(np.abs(freq_signal - np.conj(freq_signal_2)))
    assert mean_error < 1e-10


def test_scipy_zoom_dft_with_full_range_equals_fft():
    rng = np.random.default_rng()
    N = M = 100
    num_channels = 1
    signal = rng.normal(0, 1, size=(num_channels, N))

    freq_signal = spsig.zoom_fft(signal, (0,2), axis=-1)
    freq_signal_2 = np.fft.fft(signal, n = M, axis=-1)

    mean_error = np.mean(np.abs(freq_signal - freq_signal_2))
    assert mean_error < 1e-10


def test_czt_from_installed_package_followed_by_iczt_equals_original_signal():
    rng = np.random.default_rng()
    num_samples = 100
    signal = rng.normal(0, 1, size=(num_samples))

    w = rng.normal(0, 1) + 1j * rng.normal(0, 1)
    w = w / np.abs(w)
    a = rng.normal(0, 1) + 1j * rng.normal(0, 1)
    a = a / np.abs(a)

    freq_signal = czt_package.czt(signal, num_samples, w = w, a = a)
    time_signal = czt_package.iczt(freq_signal, num_samples, w = w, a = a)



def test_time2freq_from_installed_package_followed_by_freq2time_equals_original_signal():
    rng = np.random.default_rng()
    num_samples = 128
    signal = rng.normal(0, 1, size=(num_samples))

    samplerate = 1000
    times = np.arange(num_samples) / samplerate
    freqs = np.linspace(0, 2000, num_samples)
    freqs_actual, freq_signal = czt_package.time2freq(times, signal, freqs)
    times_actual, time_signal = czt_package.freq2time(freqs, freq_signal, times)

    print(np.allclose(times, times_actual))
    print(np.allclose(freqs, freqs_actual))

    mean_error = np.mean(np.abs(signal - time_signal))
    assert mean_error < 1e-10

def test_czt_from_installed_package_followed_by_iczt_equals_original_signal():
    rng = np.random.default_rng()
    M = 1024
    signal = rng.normal(0, 1, size=(M))

    #w = #rng.normal(0, 1) + 1j * rng.normal(0, 1)
    w = np.exp(2j * np.pi / (1.1 * M))
    w = w / np.abs(w)
    a = rng.normal(0, 1) + 1j * rng.normal(0, 1)
    a = a / np.abs(a)

    freq_signal = czt_package.czt(signal, M, w, a)
    time_signal = czt_package.iczt(freq_signal, M, w, a, simple=False)

    mean_error = np.mean(np.abs(signal - time_signal[None,:]))

    freq_signal = czt.czt(signal[None,:], M, w, a)
    time_signal = czt.iczt(freq_signal, M, w, a)

    mean_error2 = np.mean(np.abs(signal - time_signal))
    print(mean_error, mean_error2)
    assert mean_error < 1e-10
    assert mean_error2 < 1e-10