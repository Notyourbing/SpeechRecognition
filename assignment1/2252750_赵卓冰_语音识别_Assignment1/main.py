import numpy as np
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import librosa


# Function to apply pre-emphasis to the signal
def pre_emphasis(signal, alpha=0.97):
    # boosts the high-frequency components of the signal
    emphasized_signal = np.append(signal[0], signal[1:] - alpha * signal[:-1])
    return emphasized_signal


# Function to apply Hamming window to the frames
def windowing(frames, frame_length):
    hamming_window = np.hamming(frame_length)
    windowed_frames = frames * hamming_window
    return windowed_frames


# Function to compute Short-Time Fourier Transform(STFT)
def compute_stft(windowed_frames, NFFT=2048):
    magnitude_frames = np.abs(np.fft.rfft(windowed_frames, NFFT))
    return magnitude_frames


# Function to calculate Mel-filter bank
def mel_filter_bank(num_filters, NFFT, sample_rate):
    # Mel scale reflects the perception of frequency by human ear which has a finer perception of the low frequencies.
    # Mel scale of lowest frequency (i.e. 0 Hz)
    low_freq_mel = 0
    # Mel scale of highest frequency (i.e. nyquist frequency, half of sample rate for any signal)
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    # Generates points on the mel frequency scale
    # Plus 2 because the lowest_freq_mel point and highest_freq_mel point need to be included.
    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filters + 2)
    # Convert Mel scale back to Hz
    hz_points = 700 * (10**(mel_points / 2595) - 1)
    # Convert Hz to Fourier transform spectrum bin
    bin_points = np.floor((NFFT + 1) * hz_points / sample_rate).astype(int)

    # Create a filter bank matrix
    filter_bank = np.zeros((num_filters, int(NFFT / 2 + 1)))

    # Generate triangle filter
    for i in range(1, num_filters + 1):
        filter_bank[i - 1, bin_points[i - 1]:bin_points[i]] = \
            (np.arange(bin_points[i - 1], bin_points[i]) - bin_points[i - 1]) / (bin_points[i] - bin_points[i - 1])
        filter_bank[i - 1, bin_points[i]:bin_points[i + 1]] = \
            (bin_points[i + 1] - np.arange(bin_points[i], bin_points[i + 1])) / (bin_points[i + 1] - bin_points[i])
    return filter_bank


def mfcc(signal, sample_rate, frame_length, frame_step, num_filters=40, NFFT=2048, num_ceps=12):
    # Step 1: pre-emphasis
    emphasized_signal = pre_emphasis(signal)

    # Plot emphasized signal
    #plot_signal_wave(emphasized_signal, sample_rate, "Emphasized Signal wave")

    # Step 2: Framing
    signal_length = len(emphasized_signal)
    # Ceil to promise the possible remaining part can be a frame (padding by zero)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    # Signal length after padding by zero
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros(pad_signal_length - signal_length)
    # Signal after padding by zero
    pad_signal = np.append(emphasized_signal, z)
    # Generate frame indices matrix
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    # Extract sample points from signal. frames is a matrix, in which every row is a frame.
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    plot_frames(frames, frame_length, frame_step, sample_rate, "Framed Signal")

    # Step 3: Windowing
    windowed_frames = windowing(frames, frame_length)
    plot_frames(windowed_frames, frame_length, frame_step, sample_rate, "Windowed Framed Signal")

    # Step 4: Short-time Fourier Transform (STFT)
    magnitude_frames = compute_stft(windowed_frames, NFFT)
    plot_stft_spectrogram(magnitude_frames, sample_rate, NFFT, frame_step, "STFT Spectrogram")

    # Step 5: Applying Mel-filter bank
    filter_bank = mel_filter_bank(num_filters, NFFT, sample_rate)
    filter_bank_energy = np.dot(magnitude_frames, filter_bank.T)
    plot_mel_filter_bank(filter_bank, sample_rate, NFFT, title="Mel Filter Bank")

    # Step 6: Taking logarithm of filter bank energies
    # Replace 0 with a small value
    filter_bank_energy = np.where(filter_bank_energy == 0, np.finfo(float).eps, filter_bank_energy)
    log_energy =np.log(filter_bank_energy)
    plot_log_energy(log_energy, sample_rate, frame_step, title="Log Energy of Mel Filter Bank")

    # Step 7: Apply Discrete Cosine Transform (DCT) to getM MFCC (12 coefficients)
    mfcc_coefficients = dct(log_energy, type=2, axis=1, norm='ortho')[:, :num_ceps]

    # Step 8: Calculate energy (add energy to MFCCs)
    energy = np.sum(windowed_frames ** 2, axis=1)  # Energy for each frame
    mfcc_with_energy = np.hstack([mfcc_coefficients, energy[:, np.newaxis]])  # Add energy as 13th feature

    # Step 9: Dynamic feature extraction (Delta and Delta-Delta)
    delta_mfcc = np.gradient(mfcc_with_energy, axis=0)
    delta_delta_mfcc = np.gradient(delta_mfcc, axis=0)

    # Step 10: Feature transformation (concatenate MFCC, delta, and delta-delta)
    combined_feature = np.hstack([mfcc_with_energy, delta_mfcc, delta_delta_mfcc])

    return combined_feature.T


def plot_signal_wave(signal, sr, title):
    time_axis = np.linspace(0, len(signal) / sr, num=len(signal))

    # plot audio signal wave
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, signal)
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
    return


def plot_frames(frames, frame_length, frame_step, sample_rate, title):
    num_frames = frames.shape[0]  # Total number of frames
    total_samples = (num_frames - 1) * frame_step + frame_length  # Total number of samples after framing
    time_axis = np.linspace(0, total_samples / sample_rate, total_samples)  # Create time axis for entire signal

    plt.figure(figsize=(12, 6))

    for i in range(num_frames):
        frame_start = i * frame_step
        frame_end = frame_start + frame_length
        frame_time = time_axis[frame_start:frame_end]  # Time axis for the current frame

        plt.plot(frame_time, frames[i], label=f'Frame {i+1}')

    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend(loc="upper right", fontsize='small', ncol=2)
    plt.show()


def plot_stft_spectrogram(stft_magnitude, sr, NFFT, frame_step, title):
    num_frames = stft_magnitude.shape[0]
    num_frequency_bins = stft_magnitude.shape[1]

    # Time axis (number of frames)
    time_axis = np.arange(0, num_frames * frame_step, frame_step) / sr  # Convert to seconds

    # Frequency axis
    freq_axis = np.linspace(0, sr / 2, num_frequency_bins)

    # Plot the magnitude spectrogram
    plt.figure(figsize=(12, 6))
    plt.imshow(stft_magnitude.T, aspect='auto', origin='lower', extent=[time_axis.min(), time_axis.max(), freq_axis.min(), freq_axis.max()], cmap='jet')
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Magnitude")
    plt.show()


def plot_mel_filter_bank(filter_bank, sample_rate, NFFT, title="Mel Filter Bank"):
    num_filters = filter_bank.shape[0]
    num_fft_bins = filter_bank.shape[1]

    # Frequency axis (convert from FFT bin to frequency in Hz)
    freq_axis = np.linspace(0, sample_rate / 2, num_fft_bins)

    plt.figure(figsize=(12, 6))

    # Plot each filter in the filter bank
    for i in range(num_filters):
        plt.plot(freq_axis, filter_bank[i], label=f'Filter {i+1}')

    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()


def plot_log_energy(log_energy, sample_rate, frame_step, title="Log Energy of Mel Filter Bank"):
    num_frames = log_energy.shape[0]
    num_filters = log_energy.shape[1]

    # Time axis (number of frames)
    time_axis = np.arange(0, num_frames * frame_step, frame_step) / sample_rate  # Convert to seconds

    # Filter axis (Filter index for each filter in Mel filter bank)
    filter_axis = np.arange(num_filters)

    # Plot the log energy spectrogram
    plt.figure(figsize=(12, 6))
    plt.imshow(log_energy.T, aspect='auto', origin='lower', extent=[time_axis.min(), time_axis.max(), filter_axis.min(), filter_axis.max()], cmap='jet')
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Mel Filter Index")
    plt.colorbar(label="Log Energy")
    plt.show()


def plot_mfcc_features(features, title):
    # Display the result
    plt.imshow(features.T, aspect='auto', origin='lower', cmap='jet')
    plt.title(title)
    plt.xlabel("Frame Index")
    plt.ylabel("Coefficient Index")
    plt.colorbar()
    plt.show()



if __name__ == '__main__':
    file_path = 'input.wav'
    # The record's sample rate(sr) is 44100Hz
    signal, sr = librosa.load(file_path, sr=None)
    #plot_signal_wave(signal, sr, "Original Signal")

    # Set frame length and frame step (convert time to num of sample)
    frame_length = int(0.025 * sr)  # 25ms
    frame_step = int(0.01 * sr)  # 10ms

    # Extract 39-dimensional feature vectors
    features = mfcc(signal, sr, frame_length, frame_step)

    plot_mfcc_features(features.T, "MFCC")
