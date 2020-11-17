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
import spectralfeatures
from sklearn.metrics import roc_auc_score

SF = 16000
N_FFT = 512
#N_FILES = 30
N_FILES = 5160

def fourierTransform(filenames):
    fourierTransforms = []
    #import pdb
    #pdb.set_trace()
    for file in filenames:
        #y, _ = librosa.load(file, sr=SF)
        
        # NOTE TO SELF
        #S = np.abs(librosa.stft(y, n_fft=N_FFT, win_length=N_FFT, hop_length = N_FFT//2, window='hamming', center=False)).T
        #S = S[:10, :] # (100,), clipping the audio for logistic regression, clipping didnt help
        S = spectralfeatures.spectrogram(file, sr=SF)
        S = S.flatten() #
        fourierTransforms.append(S)
    return np.array(fourierTransforms)

def cqcc(filenames):
    cqcc_list = []
    #import pdb
    #pdb.set_trace()
    for file in filenames:
        y, _ = librosa.load(file, sr=SF)
        
        # NOTE TO SELF
        #S = spectralfeatures.mfcc(file, sr=SF)
        #S = S[:10, :] # (100,), clipping the audio for logistic regression, clipping didnt help
        S = spectralfeatures.cqcc(file, sr=SF)
        S = S.flatten() #
        cqcc_list.append(S)
    return np.array(cqcc_list)

def mfcc(filenames):
    mfcc_list = []
    #import pdb
    #pdb.set_trace()
    for file in filenames:
        y, _ = librosa.load(file, sr=SF)
        
        # NOTE TO SELF
        #S = spectralfeatures.mfcc(file, sr=SF)
        #S = S[:10, :] # (100,), clipping the audio for logistic regression, clipping didnt help
        S = spectralfeatures.mfcc(file, sr=SF)
        S = S.flatten() #
        S = S.flatten() #
        mfcc_list.append(S)
    return np.array(mfcc_list)

def mel(filenames):
    mfcc_list = []
    #import pdb
    #pdb.set_trace()
    for file in filenames:
        y, _ = librosa.load(file, sr=SF)
        
        # NOTE TO SELF
        S = spectralfeatures.melSpectrogram(file, sr=SF)
        S = S[:10, :]
        S = S.flatten() #
        mfcc_list.append(S)
    return np.array(mfcc_list)

def rms(filenames):
    mfcc_list = []
    #import pdb
    #pdb.set_trace()
    for file in filenames:
        y, _ = librosa.load(file, sr=SF)
        
        # NOTE TO SELF
        S = spectralfeatures.rms(file, sr=SF).T
        S = S[:20] # (100,), clipping the audio for logistic regression, clipping didnt help
        S = S.flatten() #
        mfcc_list.append(S)
    return bp.array(mfcc_list)
    
def logistic_regression(feature, X_train, Y_train, X_dev, Y_dev,X_eval,Y_eval,X_eval_other,Y_eval_other):
    #import pdb
    #pdb.set_trace()
    clf = LogisticRegression(random_state=0,max_iter=1000).fit(X_train, Y_train)
    dump(clf, 'models/LR/LR' + str(N_FILES) + feature+'.joblib',compress=3)
    pred_train = clf.predict(X_train)
    accuracy_train,f1_train,precision_train,recall_train = utils.accuracies(Y_train, pred_train)
    print("train accuracy:", accuracy_train)
    print("train f1:", f1_train)
    print("train precision:", precision_train)
    
    #import pdb
    #pdb.set_trace()
    pred_dev = clf.predict(X_dev)
    accuracy_dev,f1_dev,precision_dev,recall_dev = utils.accuracies(Y_dev, pred_dev)
    print("dev accuracy:", accuracy_dev)
    print("dev f1:", f1_dev)
    print("dev precision:", precision_dev)
    print("dev recall:", recall_dev)
    
    pred_eval = clf.predict(X_eval)
    accuracy_eval,f1_eval,precision_eval,recall_eval = utils.accuracies(Y_eval, pred_eval)
    print("eval accuracy:", accuracy_eval)
    print("eval f1:", f1_eval)
    print("eval precision:", precision_eval)
    print("eval recall:", recall_eval)
    
    pred_eval_other = clf.predict(X_eval_other)
    accuracy_eval_other,f1_eval_other,precision_eval_other,recall_eval_other = utils.accuracies(Y_eval_other, pred_eval_other)
    print("eval_other accuracy:", accuracy_eval_other)
    print("eval_other f1:", f1_eval_other)
    print("eval_other precision:", precision_eval_other)
    print("eval_other recall:", recall_eval_other)
    
    
def evaluate(modelPath):
    print("Loading Models")
    model = load(modelPath)
    
    pred_eval = clf.predict(X_eval)
    accuracy_eval,f1_eval,precision_eval,recall_eval = utils.accuracies(Y_eval, pred_eval)
    print("eval accuracy:", accuracy_eval)
    print("eval f1:", f1_eval)
    print("eval precision:", precision_eval)
    print("eval recall:", recall_eval)
    
    
def evaluateAndSaveResults(X_train, Y_train, X_dev, Y_dev, X_eval, Y_eval, X_eval2, Y_eval2):

        spectogramPath = "C:/Users/19498/OneDrive/Documents/CS229/audio_deepfake_detection/models/LR/LR5000fft.joblib"
        file = open(spectogramPath, "a+")
        print("Evaluating LR spectogram on Development Set")
        Y_eval_pred = evaluate(X_eval,spectogramPath)

        mel-spectogramPath = "C:/Users/19498/OneDrive/Documents/CS229/audio_deepfake_detection/models/LR/LR5160mel.joblib"
        file = open(mel-spectogramPath, "a+")
        print("Evaluating LR mel-spectogram on Development Set")
        Y_eval_pred = evaluate(X_eval,mel-spectogramPath)
        
        mfccPath = "C:/Users/19498/OneDrive/Documents/CS229/audio_deepfake_detection/models/LR/LR5160mfcc.joblib"
        file = open(mfccPath, "a+")
        print("Evaluating LR mfcc on Development Set")
        Y_eval_pred = evaluate(X_eval,mfccPath)

        rmsPath = "C:/Users/19498/OneDrive/Documents/CS229/audio_deepfake_detection/models/LR/LR5160rms.joblib"
        file = open(rmsPath, "a+")
        print("Evaluating LR RMS on Development Set")
        Y_eval_pred = evaluate(X_eval,rmsPath)

def main():
    parser = argparse.ArgumentParser()
        ## Required parameters
    parser.add_argument("--feature",
                        default=None,
                        type=str,
                        required=True,
                        help="fft, mfcc, lfcc, dct")
    args = parser.parse_args()
    feature = args.feature
    print("Feature", feature)
    X_trainfilenames, Y_train, X_devfilenames, Y_dev, X_evalfilenames, Y_eval = dataloader.load_data()
    X_evalfilenames_other, Y_eval_other = dataloader.load_other_eval_data()
    
    #X_trainfilenames = X_trainfilenames[0:N_FILES]
    #Y_train = Y_train[0:N_FILES]
 
    if(feature == "fft"):
       #import pdb
       #pdb.set_trace()
       features_x_train = fourierTransform(X_trainfilenames)
       X_train = np.stack(features_x_train) # (30, 10, 257) -> (30,), Gokul why do you use vstack - converting it to array instead
       print("X_train finished")
       X_dev = np.stack(fourierTransform(X_devfilenames))
       print("X_dev finished")
       X_eval = np.stack(fourierTransform(X_evalfilenames))
       print("X_eval finished")
       X_eval = np.stack(fourierTransform(X_evalfilenames_other))
       print("X eval other finished")
    elif(feature == "dct"):
       features_x_train = dct_func(X_trainfilenames)
       X_train = np.stack(features_x_train) # (30, 10, 257) -> (30,), Gokul why do you use vstack - converting it to array instead
       print("X_train finished")
       X_dev = np.stack(dct_func(X_devfilenames))
       print("X_dev finished")
       X_eval = np.stack(dct_func(X_evalfilenames))
       print("X_eval finished")
       X_eval_other = np.stack(dct_func(X_evalfilenames_other))
       print("X eval other finished")
    elif(feature == "mfcc"): 
       features_x_train = mfcc(X_trainfilenames)
       print("mfcc features shape",features_x_train.shape)
       X_train = np.stack(features_x_train) # (30, 10, 257) -> (30,), Gokul why do you use vstack - converting it to array instead
       print("X_train finished")
       print("X_train features shape",X_train.shape)
       X_dev = np.stack(mfcc(X_devfilenames))
       print("X_dev finished")
       print("X_dev features shape",X_dev.shape)
       X_eval = np.stack(mfcc(X_evalfilenames))
       print("X_eval finished")
       X_eval_other = np.stack(mfcc(X_evalfilenames_other))
       print("X eval other finished")
    elif(feature == "mel"):
       features_x_train = mel(X_trainfilenames)
       X_train = np.stack(features_x_train) # (30, 10, 257) -> (30,), Gokul why do you use vstack - converting it to array instead
       print("X_train finished")
       X_dev = np.stack(mel(X_devfilenames))
       print("X_dev finished")
       X_eval = np.stack(mel(X_evalfilenames))
       print("X_eval finished")
       X_eval_other = np.stack(mel(X_evalfilenames_other))
       print("X eval other finished")
    elif(feature == "rms"):
       features_x_train =rms(X_trainfilenames)
       X_train = np.stack(features_x_train) # (30, 10, 257) -> (30,), Gokul why do you use vstack - converting it to array instead
       X_dev = np.stack(rms(X_devfilenames))
       X_eval = np.stack(rms(X_evalfilenames))
       X_eval_other = np.stack(rms(X_evalfilenames_other))
       print("X eval other finished")
    elif(feature == "cqcc"):
       features_x_train =cqcc(X_trainfilenames)
       X_train = np.stack(features_x_train) # (30, 10, 257) -> (30,), Gokul why do you use vstack - converting it to array instead
       X_dev = np.stack(cqcc(X_devfilenames))
       X_eval = np.stack(cqcc(X_evalfilenames))
       X_eval_other = np.stack(cqcc(X_evalfilenames_other))
       print("X eval other finished")
    
   
    logistic_regression(feature,X_train, Y_train, X_dev, Y_dev,X_eval,Y_eval,X_eval_other,Y_eval_other)
    
if __name__ == "__main__":
    print("Inside Main")
    main()
    