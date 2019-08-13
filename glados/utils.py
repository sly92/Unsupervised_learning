#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Utilities functions for the GlasDos Package
"""


from glob import glob
from typing import List

from keras import Model
from keras.layers import Layer
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def model_builder(layers: List[Layer]) -> Model:
    """
    Build a Keras models based on a layer architecture
    :param layers: The layers to build the model
    :return: A builded (not compiled) Keras Model
    """
    inputs = layers[0]
    layer = layers[1](inputs)
    for l in layers[2:]:
        layer = l(layer)
    model = Model(inputs, layer)
    return model


def plot_elements(data: np.ndarray, columns=10, rows=10) -> None:
    """
    Generate multiple random element from data
    :param data: The data to use to generate the images
    :return: A numpy array containing all the generated images
    """
    print('PLOTTING')
    # width, height = data.shape[1], data.shape[2]
    plt.figure(figsize=(10, 10))
    for i in range(columns * rows):
        plt.subplot(rows, columns, i + 1)
        plt.imshow(data[i], cmap='gray')
    print('FINISHED SUBPLOTS')
    plt.show()


def load_sign_language_digits_ds(path: str, resize) -> np.ndarray:
    """
    Load the Deaf hand sign language digits Datasets
    :param path: The path to the dataset
    :param resize: The shape to use to reshape the image
    :return: The List of image as a numpy array
    """
    image_list = list()
    for dos in range(10):
        for filename in glob(f'{path}/{dos}/*.JPG'):
            im = Image.open(filename).convert('L')
            im.thumbnail(resize, Image.ANTIALIAS)
            im_vec = np.array(im)
            if im_vec.shape == resize:
                image_list.append(im_vec)
    return np.array(image_list)
