import os
import cv2 as cv
import pandas as pd
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import to_categorical


def map_to_letter(number):
    maps = pd.read_csv('data/emnist-balanced-mapping.txt', delimiter=' ', header=None, index_col=0)
    return chr(maps.loc[number, 1])


def save_model(model, path, name):
    h5 = path + "/" + name + ".h5"
    json = path + "/" + name + ".json"
    open(json, "w").write(model.to_json())
    model.save_weights(h5)
    print("Saving...")


def load_model(json_path, h5_path):
    model = model_from_json(open(json_path, "r").read())
    model.load_weights(h5_path)
    return model


def image_preprocess(image):
    resized_image = cv.resize(image, (28, 28))
    gray = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
    _, image_threshold = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)
    image_threshold = image_threshold.astype(float)
    image_threshold /= 255
    preprocessed_image = image_threshold.reshape((1, 28, 28, 1))
    return preprocessed_image


def emnist_preprocess(file):
    x = file.iloc[:, 1:].values
    y = file.iloc[:, :1].values
    y = to_categorical(y, num_classes=47)
    x = x / 255.0
    return x, y


def convert_1d_to_2d(data):
    data2d = []
    for i in range(data.shape[0]):
        temp = np.resize(data[i], (28, 28))
        temp = np.transpose(temp)
        data2d.append(temp)
        
    data2d = np.array(data2d)
    data2d = data2d.reshape((data2d.shape[0], 28, 28, 1))
    return data2d


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
    x, y, w, h = box
    cv.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
    cv.putText(image, prediction, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1.55, (0, 255, 255), 2)


def find_counters(image):
    """
        This function uses to find possible letter and digit areas.

        :param image:
        :return: list of (x, y, w, h)
    """
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img, (7, 7), 0)
    ret, thresh_img = cv.threshold(img_blur, thresh=125, maxval=255, type=cv.THRESH_BINARY_INV)
    contours, hierarch = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    boxes = []
    for i in range(len(contours)):
        boxes.append(cv.boundingRect(contours[i]))
    return boxes
