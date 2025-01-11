import numpy as np
import librosa
from main import plot_mfcc_features


def generate_mfcc_features(file_path, n_mfcc=13, n_fft=2048, hop_length=int(0.01 * 44100)):
    # 加载音频文件
    signal, sr = librosa.load(file_path, sr=None)

    # 确保音频加载正确
    if signal is None or len(signal) == 0:
        raise ValueError("Audio signal is empty or couldn't be loaded correctly.")

    # 计算MFCC (前13维)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    # 计算一阶差分（delta特征）
    delta_mfcc = librosa.feature.delta(mfccs)

    # 计算二阶差分（delta-delta特征）
    delta_delta_mfcc = librosa.feature.delta(mfccs, order=2)

    # 合并所有特征到一个39维特征向量
    combined_features = np.vstack([mfccs, delta_mfcc, delta_delta_mfcc])

    return combined_features


# Example usage:
file_path = 'input.wav'  # 指定音频文件路径
mfcc_features = generate_mfcc_features(file_path)

print("MFCC features shape:", mfcc_features.shape)

if __name__ == '__main__':
    file_path = 'input.wav'  # 音频文件路径
    mfcc_librosa = generate_mfcc_features(file_path)
    plot_mfcc_features(mfcc_features.T, "Librosa MFCC")