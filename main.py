import numpy as np
import cv2 as cv

image_path = ''


def preprocess(file):
    # girdi olarak bir DataFrame alır
    
    # bağımlı ve bağımsız değişkenleri ayırıyoruz
    X = file.iloc[:,1:].values
    Y = file.iloc[:,:1].values
    
    # to_categorical ile her sınıf için ayrı sınıflandırma sütunu oluşturuyoruz
    Y = to_categorical(Y,num_classes=47)
    
    # model için değerleri normalize ediyoruz
    X = X / 255.0
    
    return X,Y
