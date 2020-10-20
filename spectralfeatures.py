import numpy as np
import matplotlib.pyplot as plt
import librosa

SR = 16000
N_FFT = 512

def readAudioFile(filename):
    data, samplerate = librosa.load(filename, sr=SR)
    return data

def mfcc(filename):
    y = readAudioFile(filename)
    return librosa.feature.mfcc(y=y,
                                sr=SR)

def melSpectrogram(filename):
    y = readAudioFile(filename)
    return librosa.feature.melspectrogram(y=y,
                                          sr=SR,
                                          n_fft=N_FFT,
                                          hop_length=N_FFT//2,
                                          center=False)

def spectrogram(filename):
    y = readAudioFile(filename)
    return np.abs(librosa.stft(y=y,
                               n_fft=N_FFT,
                               win_length=N_FFT,
                               hop_length = N_FFT//2,
                               window='hamming',
                               center=False))

def rms(filename):
    y = readAudioFile(filename)
    return librosa.feature.rms(y=y,
                               frame_length=N_FFT)[0]

if __name__ == '__main__':
    fname = 'data/LA/ASVspoof2019_LA_train/flac/LA_T_1000137.flac'
    rms_feats = rms(fname)
    print(rms_feats.shape)
    times = librosa.times_like(rms_feats)
    plt.figure(0)
    plt.plot(times, rms_feats)
    mfcc_feats = mfcc(fname)
    plt.figure(1)
    plt.imshow(mfcc_feats)
    print(mfcc_feats.shape)
    melspec = melSpectrogram(fname)
    print(melspec.shape)
    plt.figure(2)
    plt.imshow(melspec)
    plt.show()
