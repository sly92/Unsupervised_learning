#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Some runs with our PCA implementations on different data
"""


from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

from glados.implementations import PCA
from glados.utils import load_sign_language_digits_ds


(x_train, _), (x_test, _) = mnist.load_data()


# PCA on Mnist
pca_data = np.reshape(x_train, (-1, 784)) / 255.0
pca = PCA(pca_data)
pca.fit()
plt.plot(pca.extracted_data[0], pca.extracted_data[1], 'ro')
plt.show()


# PCA on hand Dataset
dhi = load_sign_language_digits_ds('./../data/Sign-Language-Digits-Dataset-master', (28, 28))
dhi_train = np.reshape(dhi.astype('float32'), (-1, 784)) / 255.0
pca2 = PCA(dhi_train)
pca2.fit()
plt.plot(pca2.extracted_data[0], pca2.extracted_data[1], 'ro')
plt.show()
