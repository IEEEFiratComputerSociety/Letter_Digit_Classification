import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import HelperFunctions

emnist_train = pd.read_csv('data/emnist-balanced-train.csv')
emnist_test = pd.read_csv('data/emnist-balanced-test.csv')

print(f'Shape of emnist_train: {emnist_train.shape}\nShape of emnist_test: {emnist_test.shape}')


# %%



emnist_train_x, emnist_train_y = HelperFunctions.preprocess(emnist_train)
emnist_test_x, emnist_test_y = HelperFunctions.preprocess(emnist_test)
del emnist_train
del emnist_test

print(f'{emnist_train_x.shape}\n{emnist_train_y.shape}\n{emnist_test_x.shape}\n{emnist_test_y.shape}')


# %%


def cnn2d():
    train_X = []
    test_X = []

    for i in range(emnist_train_x.shape[0]):
        a = np.resize(emnist_train_x[i], (28, 28))
        a = np.transpose(a)
        train_X.append(a)

    for i in range(emnist_test_x.shape[0]):
        a = np.resize(emnist_test_x[i], (28, 28))
        a = np.transpose(a)
        test_X.append(a)

    x_train = np.array(train_X)
    x_test = np.array(test_X)

    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    input_shape = (28, 28, 1)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(47, activation="softmax"),
        ]
    )
    model.summary()

    batch_size = 128
    epochs = 10

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, emnist_train_y, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    return model


def cnn1d_2layer():
    """

    :return:
    """
    pass


def cnn1d_3layer():
    input_shape = 784
    model = keras.Sequential([
        layers.Conv1D(64, 8, padding='same', input_shape=(input_shape, 1)),
        layers.MaxPool1D(2),
        layers.BatchNormalization(),
        layers.Conv1D(128, 7, padding='same'),
        layers.MaxPool1D(2),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Conv1D(128, 6, padding='same'),
        layers.MaxPool1D(2),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(47, activation="softmax"),
    ])
    BATCH_SIZE = 128
    EPOCH = 20
    train = np.expand_dims(emnist_train_x, axis=2)
    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
    model.fit(train, emnist_train_y, batch_size=BATCH_SIZE, epochs=EPOCH, validation_split=0.2)
    return model


# %%
if __name__ == "__main__":
    pass
    # import Visualization
    # import random
    # for i in range(10):
    #     array = emnist_train_x[random.randint(0, 10000)]
    #     Visualization.show_image(array)

    # print(chr(map_to_letter(39)))
    # cnn2d()
    # cnn1d_3layer()
