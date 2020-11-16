import tensorflow as tf
import numpy as np
from dataloader import *
from spectralfeatures import *

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

    batch_size = 4
    epochs = 10
    learning_rate = 1e-3
    loss = 'binary_crossentropy'
    metrics = ['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size)
    ds_dev = tf.data.Dataset.from_tensor_slices((X_dev, y_dev)).shuffle(len(X_dev)).batch(batch_size)
    ds_eval = tf.data.Dataset.from_tensor_slices((X_eval, y_eval)).shuffle(len(X_eval)).batch(batch_size)


    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.fit(ds_train, epochs=epochs, validation_data=ds_dev)
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

    model.save('models/CNN/melSpectrogram_batch4_epochs10_lr1e-3_dropout')

def evaluate_model(model_name):
    batch_size = 4
    featureExtractor = 'spectrogram'

    model = tf.keras.models.load_model(model_name)
    X_other_eval, y_other_eval = load_other_eval_data()
    X_other_eval = np.stack(batchTransform(X_other_eval, featureExtractor))
    X_other_eval = X_other_eval[:, :, :, np.newaxis] # add channels axis (1)

    ds_other_eval = tf.data.Dataset.from_tensor_slices((X_other_eval, y_other_eval)).shuffle(len(X_other_eval)).batch(batch_size)
    results = model.evaluate(ds_other_eval)

    loss = results[0]
    accuracy = results[1]
    recall = results[2]
    precision = results[3]
    f1 = 2*(precision*recall)/(precision+recall)

    print('Accuracy:', accuracy)
    print('Recall:', recall)
    print('Precision:', precision)
    print('F1:', f1)

if __name__ == '__main__':
    # train_model('melSpectrogram')
    evaluate_model('models/CNN/spectrogram_batch4_epochs10_lr1e-3_dropout')
