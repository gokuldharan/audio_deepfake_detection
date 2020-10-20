import numpy as np
import librosa
import dataloader
import sklearn.mixture
from joblib import dump, load
from spectralfeatures import batchTransform

SR = 16000
N_FFT = 512
N_FILES = 60



def train(GenuineFiles, SpoofFiles,featureExtractor):
    print("Starting Fourier Transform")
    X_trainGenuine = np.vstack(batchTransform(GenuineFiles, featureExtractor, sr=SR))
    X_trainSpoof = np.vstack(batchTransform(SpoofFiles, featureExtractor, sr=SR))

    print("Training Genuine GMM")
    genuineGMM = sklearn.mixture.GaussianMixture(n_components=N_FFT, verbose=True, verbose_interval=1)
    genuineGMM.fit(X_trainGenuine)
    dump(genuineGMM, 'models/GMM/genuineGMM_n' + str(N_FILES) +'.joblib',compress=3)

    print("Training Spoof GMM")
    spoofGMM = sklearn.mixture.GaussianMixture(n_components=N_FFT, verbose=True, verbose_interval=1)
    spoofGMM.fit(X_trainSpoof)
    dump(spoofGMM, 'models/GMM/spoofGMM_n' + str(N_FILES) +'.joblib',compress=3)

def evaluate(XFiles, featureExtractor):
    genuineGMM = load('models/GMM/genuineGMM_n' + str(N_FILES) +'.joblib')
    spoofGMM = load('models/GMM/spoofGMM_n' + str(N_FILES) +'.joblib')

    genuineScores = np.zeros(len(XFiles))
    spoofScores = np.zeros(len(XFiles))
    Y = np.zeros(len(XFiles))

    for i in range(len(XFiles)):
        #The first two arrays are just for debugging/analysis
        genuineScores[i] = genuineGMM.score(batchTransform([XFiles[i]], featureExtractor, sr=SR)[0])
        spoofScores[i] = spoofGMM.score(batchTransform([XFiles[i]], featureExtractor, sr=SR)[0])
        Y[i] = 1*(spoofScores[i] > genuineScores[i])

    return Y



def main():
    X_trainfilenames, Y_train, X_devfilenames, Y_dev, X_evalfilenames, Y_eval = dataloader.load_data()
    featureExtractor = "spectrogram"

    X_trainGenuineFiles = X_trainfilenames[Y_train==0][0:N_FILES]
    Y_trainGenuine = Y_train[Y_train==0][0:N_FILES]

    X_trainSpoofFiles = X_trainfilenames[Y_train==1][0:N_FILES]
    Y_trainSpoof = Y_train[Y_train==1][0:N_FILES]

    assert(Y_trainSpoof.all() and not Y_trainGenuine.any())

    train(X_trainGenuineFiles, X_trainSpoofFiles,featureExtractor)
    Y_pred = evaluate(np.concatenate((X_trainGenuineFiles, X_trainSpoofFiles)),featureExtractor)
    acc = np.mean(Y_pred == np.concatenate((Y_trainGenuine, Y_trainSpoof)))
    print("Training Set Accuracy: " + str(acc))

if __name__ == "__main__":
    main()