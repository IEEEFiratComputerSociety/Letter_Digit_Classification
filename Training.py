import cv2
import keras.utils
import pandas as pd
import numpy as np
import Visualization
import matplotlib.pyplot as plt
import tensorflow.keras
from PIL import Image

import Visualization
import main

maps = pd.read_csv('data/emnist-balanced-mapping.txt', delimiter=' ', header=None, index_col=0)
emnist_train = pd.read_csv('data/emnist-balanced-train.csv')
emnist_test = pd.read_csv('data/emnist-balanced-test.csv')

print(f'Shape of emnist_train: {emnist_train.shape}\nShape of emnist_train: {emnist_test.shape}')


# %%
emnist_train_x, emnist_train_y = main.preprocess(emnist_train)
emnist_test_x, emnist_test_y = main.preprocess(emnist_test)
del emnist_train
del emnist_test

print(f'{emnist_train_x.shape}\n{emnist_train_y.shape}\n{emnist_test_x.shape}\n{emnist_test_y.shape}')


# %%
def map_to_letter(number, map_file=maps):
    return map_file.loc[number, 1]


def convert_1d_to_2d():
    pass


def cnn2d():
    train_X = []
    test_X = []
    
    for i in range(emnist_train_x.shape[0]):
        a = np.resize(emnist_train_x[i],(28,28))
        a = np.transpose(a)
        train_X.append(a)

    for i in range(emnist_test_x.shape[0]):
        a = np.resize(emnist_test_x[i],(28,28))
        a = np.transpose(a)
        test_X.append(a)
        
        
    x_train = np.array(train_X)
    x_test = np.array(test_X)

    x_train = x_train.reshape((x_train.shape[0],28,28,1))
    x_test = x_test.reshape((x_test.shape[0],28,28,1))
    
    input_shape = (28,28,1)
    
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(64,kernel_size=(3,3),padding="same" ,activation="relu"),
            layers.MaxPooling2D(pool_size=(2,2)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Conv2D(128,kernel_size=(3,3),padding="same" ,activation="relu"),
            layers.MaxPooling2D(pool_size=(2,2)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Conv2D(256,kernel_size=(3,3),padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2,2)),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(128,activation="relu"),
            layers.Dense(64,activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(47,activation="softmax"),
        ]
    )
    model.summary()
    
    batch_size = 128
    epochs = 10

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, emnist_train_y, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    
    open("harf_siniflandirma_emnist_model.json","w").write(model.to_json())# modeli kaydetmek için kullanılır
    model.save_weights("harf_sinif_emnist.h5")

def cnn1d_2layer():
    """

    :return:
    """
    pass


def cnn1d_3layer():
    """

    :return:
    """
    pass

# %%
# array = emnist_test_x[999]
# Visualization.show_image(array, save=True)
# print(chr(map_to_letter(39)))
