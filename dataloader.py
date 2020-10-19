import os
import numpy as np
import pandas as pd
import librosa

def balance_data():
    test_train_ratio = 0.1
    data_basepath = os.path.join(os.getcwd(), 'data', 'LA')
    data_subset_basepath = os.path.join(os.getcwd(), 'data_subset')

    if not os.path.exists(data_subset_basepath):
        os.mkdir(data_subset_basepath)

    n_dev = len(os.listdir(os.path.join(data_basepath, 'ASVspoof2019_LA_dev/flac')))
    n_eval = len(os.listdir(os.path.join(data_basepath, 'ASVspoof2019_LA_eval/flac')))
    n_train = len(os.listdir(os.path.join(data_basepath, 'ASVspoof2019_LA_train/flac')))

    train_df = pd.read_csv('data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt',
                           sep=" ",
                           header=None,
                           names=['speaker_id', 'filename', '-', 'spoof_type', 'speech_type'])

    n_per_class = 0
    out_df = None
    spoof_types = pd.unique(train_df['spoof_type'])
    n_spoof_types = len(spoof_types) - 1
    for spoof_type in spoof_types:
        sub_df = train_df.loc[train_df['spoof_type'] == spoof_type]
        if spoof_type == '-': # if bonafide
            n_per_class = sub_df.shape[0]
            train_out_df = sub_df[['filename', 'spoof_type']]
        else:
            sub_df = sub_df[:int(n_per_class/n_spoof_types)]
            train_out_df = pd.concat((train_out_df, sub_df[['filename', 'spoof_type']]))

    print(train_out_df)
    train_out_df.to_csv(os.path.join(data_subset_basepath, 'train_balanced.txt'), index=False, index_label=False)


    dev_df = pd.read_csv('data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt',
                           sep=" ",
                           header=None,
                           names=['speaker_id', 'filename', '-', 'spoof_type', 'speech_type'])
    for spoof_type in spoof_types:
        sub_df = dev_df.loc[dev_df['spoof_type'] == spoof_type]
        if spoof_type == '-': # if bonafide
            sub_df = sub_df[['filename', 'spoof_type']]
            dev_out_df = sub_df[:int(n_per_class*test_train_ratio)]
        else:
            sub_df = sub_df[:int((n_per_class/n_spoof_types)*test_train_ratio)]
            dev_out_df = pd.concat((dev_out_df, sub_df[['filename', 'spoof_type']]))

    print(dev_out_df)

    dev_out_df.to_csv(os.path.join(data_subset_basepath, 'dev_balanced.txt'), index=False, index_label=False)

    eval_df = pd.read_csv('data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt',
                           sep=" ",
                           header=None,
                           names=['speaker_id', 'filename', '-', 'spoof_type', 'speech_type'])
    for spoof_type in spoof_types:
        sub_df = eval_df.loc[eval_df['spoof_type'] == spoof_type]
        if spoof_type == '-': # if bonafide
            sub_df = sub_df[['filename', 'spoof_type']]
            eval_out_df = sub_df[:int(n_per_class*test_train_ratio)]
        else:
            sub_df = sub_df[:int((n_per_class/n_spoof_types)*test_train_ratio)]
            eval_out_df = pd.concat((eval_out_df, sub_df[['filename', 'spoof_type']]))

    print(eval_out_df)
    eval_out_df.to_csv(os.path.join(data_subset_basepath, 'eval_balanced.txt'), index=False, index_label=False)

def load_data():
    data_basepath = 'data_subset'

    df_train = pd.read_csv(os.path.join(data_basepath, 'train_balanced.txt'))
    df_dev = pd.read_csv(os.path.join(data_basepath, 'dev_balanced.txt'))
    df_eval = pd.read_csv(os.path.join(data_basepath, 'eval_balanced.txt'))

    X_train = df_train['filename'].to_numpy()
    X_train = 'data/LA/ASVspoof2019_LA_train/flac/' + X_train + '.flac'
    Y_train = np.array(df_train['spoof_type'] != '-').astype('int') # 0 if bonafide, 1 if spoof

    X_dev = df_dev['filename'].to_numpy()
    X_dev= 'data/LA/ASVspoof2019_LA_dev/flac/' + X_dev + '.flac'
    Y_dev = np.array(df_dev['spoof_type'] != '-').astype('int') # 0 if bonafide, 1 if spoof

    X_eval = df_eval['filename'].to_numpy()
    X_eval = 'data/LA/ASVspoof2019_LA_eval/flac/' + X_eval + '.flac'
    Y_eval = np.array(df_train['spoof_type'] != '-').astype('int') # 0 if bonafide, 1 if spoof

    return X_train, Y_train, X_dev, Y_dev, X_eval, Y_eval

if __name__ == '__main__':
    print(load_data())
