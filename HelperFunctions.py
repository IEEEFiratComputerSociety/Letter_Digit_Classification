import pandas as pd
import os
from tensorflow.keras.utils import to_categorical


def map_to_letter(number):
    maps = pd.read_csv('data/emnist-balanced-mapping.txt', delimiter=' ', header=None, index_col=0)
    return chr(maps.loc[number, 1])


def save_model(model, path):
    pass


def load_model(file):
    pass
    # return model


def image_preprocess(file):
    pass
    # return processed_image

def emnist_preprocess(file):
    # girdi olarak bir DataFrame alır

    # bağımlı ve bağımsız değişkenleri ayırıyoruz
    X = file.iloc[:, 1:].values
    Y = file.iloc[:, :1].values

    # to_categorical ile her sınıf için ayrı sınıflandırma sütunu oluşturuyoruz
    Y = to_categorical(Y, num_classes=47)

    # model için değerleri normalize ediyoruz
    X = X / 255.0

    return X, Y


def convert_1d_to_2d():
    pass


def be_sure_file_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print(f'{path} | the folder was created')
    else:
        print(f'{path} | the folder path exist')


def be_sure_2d(image_array):
    second_dimension = image_array.ndim
    if second_dimension < 2:
        return image_array.reshape([28, 28])
    else:
        return image_array
