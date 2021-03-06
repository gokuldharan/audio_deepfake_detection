import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy.fft
import dataloader
MATLAB_API = True
if MATLAB_API:
    import matlab.engine

SR_DEFAULT = 16000
N_FFT = 512
N_SAMPLES = 10000
engine = None

def readAudioFile(filename, sr=SR_DEFAULT):
    data, samplerate = librosa.load(filename, sr)
    data = data[:N_SAMPLES]
    return data

def cqcc(filename, sr=SR_DEFAULT):
    global engine
    if engine is None:
        engine = matlab.engine.start_matlab()
    y = readAudioFile(filename, sr)
    x_mat = matlab.double(y.tolist())
    fs_mat = matlab.double([sr])
    return np.asarray(engine.extractCQCC(x_mat, fs_mat)).T

def mfcc(filename, sr=SR_DEFAULT):
    y = readAudioFile(filename, sr)
    return librosa.feature.mfcc(y=y,
                                sr=sr).T

def melSpectrogram(filename, sr=SR_DEFAULT):
    y = readAudioFile(filename, sr)
    return librosa.feature.melspectrogram(y=y,
                                          sr=sr,
                                          n_fft=N_FFT,
                                          hop_length=N_FFT//2,
                                          center=False).T

def spectrogram(filename, sr=SR_DEFAULT):
    y = readAudioFile(filename, sr)
    return np.abs(librosa.stft(y=y,
                               n_fft=N_FFT,
                               win_length=N_FFT,
                               hop_length = N_FFT//2,
                               window='hamming',
                               center=False)).T

def rms(filename, sr=SR_DEFAULT):
    y = readAudioFile(filename, sr)
    return librosa.feature.rms(y=y,
                               frame_length=N_FFT)[0]

def dct(filename, sr=SR_DEFAULT):
    y = readAudioFile(filename, sr)
    return scipy.fft.dct(y)


featureExtractors = {"mfcc":mfcc,
                     "melSpectrogram":melSpectrogram,
                     "spectrogram":spectrogram,
                     "rms":rms,
                     "dct":dct,
                     "cqcc":cqcc}


def batchTransform(filenames, featureExtractor, sr=SR_DEFAULT):
    transforms = []
    for file in filenames:
        S = featureExtractors[featureExtractor](file, sr)
        if S.ndim == 1:
            S = np.expand_dims(S, axis=1)
        transforms.append(S)
    return transforms

if __name__ == '__main__':
    # fname = 'data/LA/ASVspoof2019_LA_train/flac/LA_T_1000137.flac'
    # readAudioFile(fname, SR_DEFAULT)
    # rms_feats = rms(fname)
    # print(rms_feats.shape)
    # times = librosa.times_like(rms_feats)
    # plt.figure(0)
    # plt.plot(times, rms_feats)
    # mfcc_feats = mfcc(fname)
    # plt.figure(1)
    # plt.imshow(mfcc_feats)
    # print(mfcc_feats.shape)
    # melspec = melSpectrogram(fname)
    # print(melspec.shape)
    # plt.figure(2)
    # plt.imshow(melspec)
    # plt.show()
    X_train, Y_train, X_dev, Y_dev, X_eval, Y_eval = dataloader.load_data()
    min_len = np.inf
    max_len = 0
    for x in X_train:
        data = readAudioFile(x)
        if data.shape[0] < min_len:
            min_len = data.shape[0]
        if data.shape[0] > max_len:
            max_len = data.shape[0]

    for x in X_dev:
        data = readAudioFile(x)
        if data.shape[0] < min_len:
            min_len = data.shape[0]
        if data.shape[0] > max_len:
            max_len = data.shape[0]

    for x in X_eval:
        data = readAudioFile(x)
        if data.shape[0] < min_len:
            min_len = data.shape[0]
        if data.shape[0] > max_len:
            max_len = data.shape[0]

    print('Min len: ', min_len)
    print('Max len: ', max_len)