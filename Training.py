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
    """

    :return:
    """
    pass


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
