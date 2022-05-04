import argparse

import cv2 as cv
import numpy as np
from tensorflow.keras.models import model_from_json

import HelperFunctions

image_path = '20220429_205137.jpg'
model_path = ''

parser = argparse.ArgumentParser(description='Single letter or digit classier')
parser.add_argument('-p', '--image_path')
args = parser.parse_args()
image_path = args.image_path

if __name__ == "__main__":
    model = HelperFunctions.load_model(model_path)
    image = cv.imread(image_path)
    model_input_image = HelperFunctions.image_preproccess(image)
    prediction = model.predict(model_input_image)
    print(f'The Prediction is {HelperFunctions.map_to_letter(np.argmax(prediction))}')


