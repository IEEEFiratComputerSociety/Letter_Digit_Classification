import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def show_image(image_array, save=False, save_path='data/csv_to_image'):

    pass


def prediction_bar_plot(prediction_array, save=False, save_path='data/prediction_bar_plot'):

    pass


def be_sure_file_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print(f'{path} | the folder was created')
    else:
        print(f'{path} | the folder path exist')

