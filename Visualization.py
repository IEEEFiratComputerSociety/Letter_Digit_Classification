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



def prediction_bar_plot(save=False, save_path='data/prediction_bar_plot'):

    
   
    
    liste=[48,49,50,51,52,53,54,55,56,57,65,66,67,68,69,70,71,72
    ,73,74,75,76,77,78,79,80,81,82,83,84,
    85,86,87,88,89,90,97,98,100,101,102,103,104,110,
    113,114,
    116] # direkt data frame olarak ascıı tabloya donuşturulemedi burayı konuşalım tekrar.
    array=[]
    for i in liste:
        print(chr(i)) #listedeki değerlerin ASCII karşılıkları verildi.
        array.append(chr(i))
    
    plt.title("Letter Detection") #başlık ekledik
    x=(array)
    y=(len(array))
    plt.bar(x,y,color="purple",alpha=0.7,label="line",width=0.7)
    plt.xlabel("Letters and Numbers")
    plt.ylabel("Prediction Result")
    
    plt.show()




prediction_bar_plot()


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

