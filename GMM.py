import numpy as np
import librosa
import dataloader
import sklearn.mixture
from joblib import dump, load
from spectralfeatures import batchTransform
import os.path
import utils
import time

SR = 16000
#max 2580
N_FILES = 2580
N_CMP = 512

featureExtractor = "spectrogram"

#MAKE SURE THIS PATH ("models/GMM/") EXISTS!!! It's hacky I know
genuineGMMPath = 'models/GMM/genuineGMM_' + featureExtractor + '_sr' + str(SR) + '_n' + str(N_FILES) + '_NCMP' + str(N_CMP) + '.joblib'
spoofGMMPath = 'models/GMM/spoofGMM_' + featureExtractor + '_sr' + str(SR) + '_n' + str(N_FILES)  + '_NCMP' + str(N_CMP) + '.joblib'
resultsPath = 'models/GMM/' + featureExtractor + '_sr' + str(SR) + '_n' + str(N_FILES)  + '_NCMP' + str(N_CMP) + '.txt'

def train(GenuineFiles, SpoofFiles):
    if os.path.exists(genuineGMMPath) and os.path.exists(spoofGMMPath):
         print("Already trained with these params, skipping!")
         return

    print("Starting Feature Extraction")
    X_trainGenuine = np.vstack(batchTransform(GenuineFiles, featureExtractor, sr=SR))
    X_trainSpoof = np.vstack(batchTransform(SpoofFiles, featureExtractor, sr=SR))

    #SWITCHING TO DIAGONAL COVARIANCES FOR RUNTIME
    print("Training Genuine GMM")
    genuineGMM = sklearn.mixture.GaussianMixture(n_components=N_CMP, verbose=True, verbose_interval=1, covariance_type='diag')
    genuineGMM.fit(X_trainGenuine)
    dump(genuineGMM, genuineGMMPath,compress=3)

    print("Training Spoof GMM")
    spoofGMM = sklearn.mixture.GaussianMixture(n_components=N_CMP, verbose=True, verbose_interval=1, covariance_type='diag')#, reg_covar=1e-5
    spoofGMM.fit(X_trainSpoof)
    dump(spoofGMM, spoofGMMPath,compress=3)


def evaluate(XFiles):
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

def evaluateAndSaveResults(X_train, Y_train, X_dev, Y_dev, X_eval, Y_eval, X_eval2, Y_eval2):

    if os.path.exists(resultsPath):
        print("Already saved results with these params, skipping evaluation and loading from file!")
    else:
        file = open(resultsPath, "a+")

        # print("Evaluating on Training Set")
        # Y_trnpred = evaluate(X_train)
        # accuracy_trn,f1_trn,precision_trn,recall_trn = utils.accuracies(Y_train, Y_trnpred)
        # file.write("trn accuracy:" + str(accuracy_trn) + "\n")
        # file.write("trn f1:"+ str( f1_trn) + "\n")
        # file.write("trn precision:"+ str( precision_trn) + "\n")
        # file.write("trn recall:"+ str( recall_trn) + "\n")

        print("Evaluating on Development Set")
        Y_devpred = evaluate(X_dev)
        accuracy_dev,f1_dev,precision_dev,recall_dev = utils.accuracies(Y_dev, Y_devpred)
        file.write("dev accuracy:"+ str( accuracy_dev) + "\n")
        file.write("dev f1:"+ str( f1_dev) + "\n")
        file.write("dev precision:"+ str( precision_dev) + "\n")
        file.write("dev recall:"+ str( recall_dev) + "\n")

        print("Evaluating on Evaluation Set")
        Y_evalpred = evaluate(X_eval)
        accuracy_eval,f1_eval,precision_eval,recall_eval = utils.accuracies(Y_eval, Y_evalpred)
        file.write("eval accuracy:"+ str( accuracy_eval) + "\n")
        file.write("eval f1:"+ str( f1_eval) + "\n")
        file.write("eval precision:"+ str( precision_eval) + "\n")
        file.write("eval recall:"+ str( recall_eval) + "\n")

        print("Evaluating on Evaluation Set 2 - Unknown Protocols")
        Y_eval2pred = evaluate(X_eval2)
        accuracy_eval2,f1_eval2,precision_eval2,recall_eval2 = utils.accuracies(Y_eval2, Y_eval2pred)
        file.write("eval2 accuracy:"+ str( accuracy_eval2) + "\n")
        file.write("eval2 f1:"+ str( f1_eval2) + "\n")
        file.write("eval2 precision:"+ str( precision_eval2) + "\n")
        file.write("eval2 recall:"+ str( recall_eval2) + "\n")

        file.close()

    file = open(resultsPath, "r")
    for line in file.readlines():
        print(line)


def main():
    X_trainfilenames, Y_train, X_devfilenames, Y_dev, X_evalfilenames, Y_eval = dataloader.load_data()
    X_eval2filenames, Y_eval2 = dataloader.load_other_eval_data()


    X_trainGenuineFiles = X_trainfilenames[Y_train==0][0:N_FILES]
    Y_trainGenuine = Y_train[Y_train==0][0:N_FILES]

    X_trainSpoofFiles = X_trainfilenames[Y_train==1][0:N_FILES]
    Y_trainSpoof = Y_train[Y_train==1][0:N_FILES]

    assert(Y_trainSpoof.all() and not Y_trainGenuine.any())

    train(X_trainGenuineFiles, X_trainSpoofFiles)
    evaluateAndSaveResults(np.concatenate((X_trainGenuineFiles, X_trainSpoofFiles)),
                            np.concatenate((Y_trainGenuine, Y_trainSpoof)),
                            X_devfilenames,
                            Y_dev,
                            X_evalfilenames,
                            Y_eval,
                            X_eval2filenames,
                            Y_eval2)


if __name__ == "__main__":
    main()