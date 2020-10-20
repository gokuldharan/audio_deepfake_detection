import numpy as np
import librosa
import dataloader
import sklearn.mixture
from joblib import dump, load
from spectralfeatures import batchTransform
import os.path

SR = 8000
N_FILES = 60


genuineGMMPath = 'models/GMM/genuineGMM_sr' + str(SR) + '_n' + str(N_FILES) +'.joblib'
spoofGMMPath = 'models/GMM/spoofGMM_sr' + str(SR) + '_n' + str(N_FILES) +'.joblib'


def train(GenuineFiles, SpoofFiles,featureExtractor):
    if os.path.exists(genuineGMMPath) and os.path.exists(spoofGMMPath):
        print("Already trained with these params, skipping!")
        return

    print("Starting Feature Extraction")
    X_trainGenuine = np.vstack(batchTransform(GenuineFiles, featureExtractor, sr=SR))
    X_trainSpoof = np.vstack(batchTransform(SpoofFiles, featureExtractor, sr=SR))

    print("Training Genuine GMM")
    genuineGMM = sklearn.mixture.GaussianMixture(n_components=N_FFT, verbose=True, verbose_interval=1)
    genuineGMM.fit(X_trainGenuine)
    dump(genuineGMM, genuineGMMPath,compress=3)

    print("Training Spoof GMM")
    spoofGMM = sklearn.mixture.GaussianMixture(n_components=N_FFT, verbose=True, verbose_interval=1)
    spoofGMM.fit(X_trainSpoof)
    dump(spoofGMM, spoofGMMPath,compress=3)


def evaluate(XFiles, featureExtractor):
    print("Loading Models")
    genuineGMM = load(genuineGMMPath)
    spoofGMM = load(spoofGMMPath)

    genuineScores = np.zeros(len(XFiles))
    spoofScores = np.zeros(len(XFiles))
    Y = np.zeros(len(XFiles))

    print("Evaluating Predictions")
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
    accuracy_dev,f1_dev,precision_dev,recall_dev = utils.accuracies(Y_dev, Y_pred)
    print("dev accuracy:", accuracy_dev)
    print("dev f1:", f1_dev)
    print("dev precision:", precision_dev)
    print("dev recall:", recall_dev)

    acc = np.mean(Y_pred == np.concatenate((Y_trainGenuine, Y_trainSpoof)))
    print("Training Set Accuracy: " + str(acc))

if __name__ == "__main__":
    main()