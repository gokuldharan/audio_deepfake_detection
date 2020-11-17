import tensorflow as tf
import numpy as np
from dataloader import *
from spectralfeatures import *
from sklearn import metrics
import matplotlib.pyplot as plt

# Builds Keras model
def build_model(input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(8, (3,3), padding='same', activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPool2D((2,2)))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Conv2D(16, (3,3), padding='same', activation='relu'))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    return model

# Trains Keras model
def train_model(featureExtractor):
    X_train, y_train, X_dev, y_dev, X_eval, y_eval = load_data()

    # Transforming into feature space
    X_train = np.stack(batchTransform(X_train, featureExtractor))
    X_dev = np.stack(batchTransform(X_dev, featureExtractor))
    X_eval = np.stack(batchTransform(X_eval, featureExtractor))
    X_train = X_train[:, :, :, np.newaxis] # add channels axis (1)
    X_dev = X_dev[:, :, :, np.newaxis] # add channels axis (1)
    X_eval = X_eval[:, :, :, np.newaxis] # add channels axis (1)

    model = build_model(X_train.shape[1:])
    model.summary()

    logdir = 'test_logdir/train'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    batch_size = 1
    epochs = 10
    learning_rate = 1e-3
    loss = 'binary_crossentropy'
    metrics = ['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size)
    ds_dev = tf.data.Dataset.from_tensor_slices((X_dev, y_dev)).shuffle(len(X_dev)).batch(batch_size)
    ds_eval = tf.data.Dataset.from_tensor_slices((X_eval, y_eval)).shuffle(len(X_eval)).batch(batch_size)


    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.fit(ds_train, epochs=epochs, validation_data=ds_dev, callbacks=[tensorboard_callback])
    results = model.evaluate(ds_dev)

    loss = results[0]
    accuracy = results[1]
    recall = results[2]
    precision = results[3]
    f1 = 2*(precision*recall)/(precision+recall)

    print('Accuracy:', accuracy)
    print('Recall:', recall)
    print('Precision:', precision)
    print('F1:', f1)

    model.save('models/CNN/cqcc_batch1_epochs10_lr1e-3_dropout')

def evaluate_model(model_name):
    batch_size = 1
    featureExtractor = 'melSpectrogram'

    model = tf.keras.models.load_model(model_name)
    logdir = 'test_logdir/train'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    X_train, y_train, X_dev, y_dev, X_eval, y_eval = load_data()
    X_eval = np.stack(batchTransform(X_eval, featureExtractor))
    X_eval = X_eval[:, :, :, np.newaxis] # add channels axis (1)

    X_other_eval, y_other_eval = load_other_eval_data()
    X_other_eval = np.stack(batchTransform(X_other_eval, featureExtractor))
    X_other_eval = X_other_eval[:, :, :, np.newaxis] # add channels axis (1)

    ds_eval = tf.data.Dataset.from_tensor_slices((X_eval, y_eval)).shuffle(len(X_eval)).batch(batch_size)
    ds_other_eval = tf.data.Dataset.from_tensor_slices((X_other_eval, y_other_eval)).shuffle(len(X_other_eval)).batch(batch_size)

    results = model.evaluate(ds_eval)

    loss = results[0]
    accuracy = results[1]
    recall = results[2]
    precision = results[3]
    f1 = 2*(precision*recall)/(precision+recall)

    print('Same spoof method eval:')
    print('Accuracy:', accuracy)
    print('Recall:', recall)
    print('Precision:', precision)
    print('F1:', f1)

    results = model.evaluate(ds_other_eval, callbacks=[tensorboard_callback])

    loss = results[0]
    accuracy = results[1]
    recall = results[2]
    precision = results[3]
    f1 = 2*(precision*recall)/(precision+recall)

    print('Other spoof method eval:')
    print('Accuracy:', accuracy)
    print('Recall:', recall)
    print('Precision:', precision)
    print('F1:', f1)


def plot_roc():
    features = ['spectrogram', 'melSpectrogram', 'mfcc', 'cqcc']
    batch_size = 1
    X_train, y_train, X_dev, y_dev, X_eval_orig, y_eval = load_data()
    X_other_eval_orig, y_other_eval = load_other_eval_data()
    for feature in features:
        model = tf.keras.models.load_model('models/CNN/'+feature+'_batch1_epochs10_lr1e-3_dropout')

        X_eval = np.stack(batchTransform(X_eval_orig, feature))
        X_eval = X_eval[:, :, :, np.newaxis] # add channels axis (1)

        y_pred = model.predict(X_eval)

        fpr, tpr, thresholds = metrics.roc_curve(y_eval, y_pred)
        roc_auc = metrics.auc(fpr, tpr)
        print('AUC for '+feature+':',roc_auc)

        plt.plot(fpr, tpr)

    legend_labels = ['Spectrogram (area = 0.94)', 'Mel-Spectrogram (area = 0.76)', 'MFCC (area = 0.96)', 'CQCC (area = 0.5)']
    plt.legend(legend_labels)
    plt.plot([0, 1], [0, 1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for CNN')

    plt.show()

# AUC for spectrogram: 0.9357936221781544
# AUC for melSpectrogram: 0.7458072932315767
# AUC for mfcc: 0.9580053282054364
# AUC for cqcc: 0.5



if __name__ == '__main__':
    # train_model('cqcc')
    # evaluate_model('models/CNN/melSpectrogram_batch1_epochs10_lr1e-3_dropout')
    plot_roc()
