import numpy as np
from tensorflow.keras.models import model_from_json
import HelperFunctions
import cv2 as cv

image_path = '20220429_205137.jpg'
model_path = ''


if __name__ == "__main__":
    # model = load_model(model_path)
    image = cv.imread(image_path)
    # model_input_image = image_preproccess(image)


