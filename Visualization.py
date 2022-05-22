import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

import HelperFunctions


def show_image(image_array, save=False, save_path='data/csv_to_image', name_of_image='show_image', image_type='.png'):
    checked_array = HelperFunctions.be_sure_2d(image_array)
    checked_array = checked_array.astype(int)
    checked_array = np.transpose(checked_array, (1, 0))
    plt.subplot()
    plt.imshow(checked_array, cmap='gray')
    plt.show()

    if save:
        HelperFunctions.be_sure_file_exist(save_path)
        data = Image.fromarray(checked_array)
        path = os.path.join(save_path, name_of_image + image_type)
        data = ImageOps.grayscale(data)
        data.save(path)


def conf_matrix(model, x_test, y_true):
    test_X = []

    for i in range(x_test.shape[0]):
        a = np.resize(x_test[i], (28, 28))
        a = np.transpose(a)
        test_X.append(a)

    test_X = np.array(test_X)
    test_X = test_X.reshape((test_X.shape[0], 28, 28, 1))

    # test setinden tahmin et
    x_pred = model.predict(test_X)

    # tahmin edilen veriyi çevir
    x_pred_class = np.argmax(x_pred, axis=1)

    # karışıklık matrisi hesaplama
    confusion_mtx = confusion_matrix(y_true, x_pred_class)
    labels = list(map(HelperFunctions.map_to_letter, list(range(47))))

    # karşıklık matrisi çizdir
    f, ax = plt.subplots(figsize=(30, 30))
    sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, xticklabels = labels, yticklabels = labels, cmap="Greens", linecolor="gray", fmt="d", ax=ax)
    plt.xlabel("tahmin edilen etiket")
    plt.ylabel("Gerçek etiket")
    plt.title("confusion matrix")
    plt.show()



def prediction_bar_plot(prediction_data, save=False, save_path='data/prediction_bar_plot', name_of_plot='prediction_plot', image_type='.png'):
    x = []
    for i in range(47):
        x.append(HelperFunctions.map_to_letter(i))
    plt.title("Letter Detection")  # başlık ekledik
    plt.bar(x, prediction_data, color="skyblue", alpha=0.7, label="line", width=0.7)
    plt.xlabel("Letters and Numbers")
    plt.ylabel("Prediction Result")
    if save:
        HelperFunctions.be_sure_file_exist(save_path)
        path = os.path.join(save_path, name_of_plot + image_type)
        plt.savefig(path)
    plt.show()


