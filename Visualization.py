import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

import HelperFunctions


def show_image(image_array, save=False, save_path='data/csv_to_image', name_of_image='show_image', image_type='.png'):
    checked_array = HelperFunctions.be_sure_2d(image_array)
    checked_array = checked_array * 255
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


# def show_images(image_data, labels, number_of_image=10, save=False, save_path='data/csv_to_image',
#                 name_of_image='show_image', image_type='.png'):
#     temp_data = pd.DataFrame()
#     plt.subplot(48, name_of_image)
#     i, j = 0, 0
#     for group in range(48):
#         condition = labels == i
#         temp_data.append(image_data[condition].sample(n=number_of_image))
#         i += 1 if group
#         plt.subplot(48, name_of_image, )


def conf_matrix(model, x_test, y_true):
    '''
    Eğitilmiş modeli parametre olarak veriyoruz
    '''
    # test_X = []
    # X, Y = preprocess()
    #
    # for i in range(emnist_test_x.shape[0]):
    #     a = np.resize(emnist_test_x[i], (28, 28))
    #     a = np.transpose(a)
    #     test_X.append(a)
    #
    # x_test = np.array(test_X)
    # x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    # test setinden tahmin et
    x_pred = model.predict(x_test)

    # tahmin edilen veriyi çevir
    x_pred_class = np.argmax(x_pred, axis=1)

    # # test verisi çevir
    # y_true = np.argmax(Y, axis=1)

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


def training_plot():
    pass


def prediction_bar_plot(prediction_data, save=False, save_path='data/prediction_bar_plot'):
    """
    liste=[48,49,50,51,52,53,54,55,56,57,65,66,67,68,69,70,71,72
    ,73,74,75,76,77,78,79,80,81,82,83,84,
    85,86,87,88,89,90,97,98,100,101,102,103,104,110,
    113,114,
    116] # direkt data frame olarak ascıı tabloya donuşturulemedi burayı konuşalım tekrar.

    for i in liste:
        print(chr(i)) #listedeki değerlerin ASCII karşılıkları verildi.
        array.append(chr(i))"""
    array = []
    for i in range(47):
        array.append(chr(HelperFunctions.map_to_letter(i)))
    plt.title("Letter Detection")  # başlık ekledik
    x = (array)
    y = (len(array))
    plt.bar(x, prediction_data, color="skyblue", alpha=0.7, label="line", width=0.7)
    plt.xlabel("Letters and Numbers")
    plt.ylabel("Prediction Result")

    plt.show()


