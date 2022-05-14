import os

import pandas as pd
from tensorflow.keras.utils import to_categorical


def map_to_letter(number):
    maps = pd.read_csv('data/emnist-balanced-mapping.txt', delimiter=' ', header=None, index_col=0)
    return chr(maps.loc[number, 1])


def save_model(model, path, name):
    result=path+"/"+name+".h5"
    model.save(result)
    print("Saving...")


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


def draw_rectangle(image, prediction, box):
    """
    This function uses for draw rectangle around prediction

    :param image: the image which is used for localization
    :param prediction: It is localization area of classification output.
    :type prediction: char
    :param box: The localization area position on image. It will be a tuple (x, y, w, h).
    :type box: tuple
    :return:
    """
    # TODO: verilen gorsel(image) uzerine box'da(x, y, w, h) bulunan kare cizilecek ve karenin sol ust kosesine
    #  prediction da bulunan o karenin char degeri yazilacak.
    #  https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
    pass


def find_counters(image):
    """
    This function uses to find possible letter and digit areas.

    :param image:
    :return: list of (x, y, w, h)
    """
    # TODO: Verilen gorsele sirasiyla gray, blur, thrash, findCounters islemleri uygulanacak. Bulunan
    #  counterlarin (x, y, w, h) degerleri bulunup liste olarak dondurulecek
    #  https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
    pass


