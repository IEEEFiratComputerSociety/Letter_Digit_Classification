import os

import cv2 as cv
import pandas as pd
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import to_categorical


def map_to_letter(number):
    maps = pd.read_csv('data/emnist-balanced-mapping.txt', delimiter=' ', header=None, index_col=0)
    return chr(maps.loc[number, 1])


def save_model(model, path, name):
    h5 = path + "/" + name + ".h5"
    json = path + "/" + name + ".json"
    open(json, "w").write(model.to_json())  # modeli kaydetmek için kullanılır
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
    x,y,w,h=[box]
    img = cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 1)
    #cv2.putText(image, 'p', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    cv2.putText(image,'predict :{}'.format(prediction),(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),2)
    

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
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img_blur = cv.blur(img, ksize=(5, 5))
    ret, thresh_img = cv.threshold(img_blur, thresh=125, maxval=255, type=cv.THRESH_BINARY_INV)
    contours, hierarch = cv.findContours(thresh_img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    boxes = []
    for i in range(len(contours)):
        boxes.append(cv.boundingRect(contours[i]))
    return boxes
