import os

import cv2
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def show_image(image_array, save=False, save_path='data/csv_to_image', name_of_image='show_image', image_type='.png'):
    checked_array = be_sure_2d(image_array)
    checked_array = checked_array*255
    checked_array = checked_array.astype(int)
    checked_array = np.transpose(checked_array, (1, 0))
    plt.subplot()
    plt.imshow(checked_array, cmap='gray')
    plt.show()

    if save:
        be_sure_file_exist(save_path)
        data = Image.fromarray(checked_array)
        path = os.path.join(save_path, name_of_image + image_type)
        data = ImageOps.grayscale(data)
        data.save(path)


def model_output_plot():
    pass


def prediction_bar_plot(prediction_array, save=False, save_path='data/prediction_bar_plot'):

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

