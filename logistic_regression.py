import numpy as np
import librosa
import dataloader
import sklearn.mixture
import argparse
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import utils

SF = 16000
N_FFT = 512
N_FILES = 30


def fourierTransform(filenames):
    fourierTransforms = []
    #import pdb
    #pdb.set_trace()
    for file in filenames:
        y, _ = librosa.load(file, sr=SF)
        
        # NOTE TO SELF
        S = np.abs(librosa.stft(y, n_fft=N_FFT, win_length=N_FFT, hop_length = N_FFT//2, window='hamming', center=False)).T
        S = S[:10, :] # (100,), clipping the audio for logistic regression, clipping didnt help
        
        
        S = S.flatten() #
        fourierTransforms.append(S)
    return fourierTransforms

def logistic_regression(X_train, Y_train, X_dev, Y_dev):
    #import pdb
    #pdb.set_trace()
    clf = LogisticRegression(random_state=0).fit(X_train, Y_train)
    #dump(clf, 'models/LR/LR' + str(N_FILES) +'.joblib',compress=3)
    pred_train = clf.predict(X_train)
    accuracy_train,f1_train,precision_train,recall_train = utils.accuracies(Y_train, pred_train)
    print("train accuracy:", accuracy_train)
    print("train f1:", f1_train)
    print("train precision:", precision_train)
    
    pred_dev = clf.predict(X_dev)
    accuracy_dev,f1_dev,precision_dev,recall_dev = utils.accuracies(Y_dev, pred_dev)
    print("dev accuracy:", accuracy_dev)
    print("dev f1:", f1_dev)
    print("dev precision:", precision_dev)
    print("dev recall:", recall_dev)
    
def main():
    print("Inside Main")
    parser = argparse.ArgumentParser()
        ## Required parameters
    parser.add_argument("--feature",
                        default=None,
                        type=str,
                        required=True,
                        help="fft, mfcc, lfcc, dct")
    args = parser.parse_args()
    feature = args.feature
    X_trainfilenames, Y_train, X_devfilenames, Y_dev, X_evalfilenames, Y_eval = dataloader.load_data()
    
    X_trainfilenames = X_trainfilenames[0:N_FILES]
    Y_train = Y_train[0:N_FILES]
    
    
    if(feature == "fft"):
       #import pdb
       #pdb.set_trace()
       features_x_train = fourierTransform(X_trainfilenames)
       X_train = np.stack(features_x_train) # (30, 10, 257) -> (30,), Gokul why do you use vstack - converting it to array instead
       X_dev = np.stack(fourierTransform(X_devfilenames))
    else if(feature == "dct"):
    
   
    logistic_regression(X_train, Y_train, X_dev, Y_dev)
    
if __name__ == "__main__":
    main()
    