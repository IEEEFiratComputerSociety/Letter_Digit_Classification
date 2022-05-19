import argparse
import sys

import cv2 as cv
import numpy as np

import HelperFunctions

model_weights = "models/2dmodel.h5"
model_json = "models/2dmodel.json"
model = HelperFunctions.load_model(model_json, model_weights)

parser = argparse.ArgumentParser(description='Letter or digit classier')
parser.add_argument('-s', '--s_image_path', default=None)
parser.add_argument('-m', '--m_image_path', default=None)

args = parser.parse_args()

if args.s_image_path:
    image = cv.imread(args.s_image_path)

    model_input_image = HelperFunctions.image_preprocess(image)
    prediction = model.predict(model_input_image)
    predicted_char = HelperFunctions.map_to_letter(np.argmax(prediction))

    cv.putText(image, predicted_char, (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    cv.imshow('image', image)
    print(f'The Prediction is {predicted_char}')

    cv.waitKey(0)

elif args.m_image_path:
    image = cv.imread(args.m_image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    boxes = HelperFunctions.find_counters(image)
    for (x, y, w, h) in boxes:
        if (5 <= w <= 300) and (30 <= h <= 300):
            roi = gray[y:y + h, x:x + w]
            _, thresh = cv.threshold(roi, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
            height, width = thresh.shape
            if width > height:
                dim = (28, int(height * (28 / float(width))))
                thresh = cv.resize(thresh, dim, interpolation=cv.INTER_AREA)
            else:
                dim = (int(width * (28 / float(height))), 28)
                thresh = cv.resize(thresh, dim, interpolation=cv.INTER_AREA)

            padded = cv.copyMakeBorder(thresh, top=10, bottom=10, left=10, right=10, borderType=cv.BORDER_CONSTANT,
                                       value=0)
            padded = cv.resize(padded, (28, 28))
            padded = padded.astype("float32") / 255.0
            padded = padded.reshape((1, 28, 28, 1))
            prediction = model.predict(padded)
            predicted_char = HelperFunctions.map_to_letter(np.argmax(prediction))
            HelperFunctions.draw_rectangle(image, predicted_char, (x, y, w, h))

    image = cv.resize(image, (680, 680))
    cv.imshow('image', image)
    cv.waitKey(0)

else:
    print('"-m" or "-s" should be used.', file=sys.stderr)

cv.destroyAllWindows()
