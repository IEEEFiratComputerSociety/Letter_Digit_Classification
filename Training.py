import pandas as pd
import numpy as np
import gzip
import matplotlib.pyplot as plt

maps = pd.read_csv('data/emnist-balanced-mapping.txt', delimiter=' ', header=None, index_col=0)
emnist_train = pd.read_csv('data/emnist-balanced-train.csv')
emnist_test = pd.read_csv('data/emnist-balanced-test.csv')


# %%
def map_to_letter(number, map_file=maps):
    return map_file.loc[number, 1]


print(map_to_letter(25))
