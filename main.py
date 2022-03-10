import numpy as np
import cv2 as cv

image_path = ''


def preprocess(image):
    # Resmi BGR renk tonlamasından siyah beyaza çeviriyoruz
    image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    
    # Resmi sınıflandırma algoritmamızın girdi boyutu olan (28x28) boyutlarına getiriyoruz
    image = cv.resize(image,(28,28))
    
    # Sınıflandırma algoritmamız girdi olarak bir vertör aldığı için resmi vektör haline getiriyoruz
    image = image.reshape(784)
    
    return image
