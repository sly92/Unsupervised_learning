#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Some runs with our Kmeans implementations on different data
"""


from keras.datasets import mnist
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# from PIL import Image

from glados.implementations import LloydKmeans
from glados.utils import plot_elements, load_sign_language_digits_ds


(x_train, y_train), (x_test, y_test) = mnist.load_data()
fake_data = np.asarray([[1, 3], [1.5, 2.5], [1, 2], [3, 1], [3, 2]])

# Kmeans run on MNIST
mnist_lloyd_kmeans = LloydKmeans(np.reshape(x_train[0:50], (-1, 784)) / 255.0, y_train, 10)
mnist_lloyd_kmeans.fit()

centroids = []
items = []
clusters = {}
labels = {}
points = []

listColor = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c',
             '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
             '#000075', '#808080', '#ffffff', '#000000']
i = 0


for item in mnist_lloyd_kmeans.mus:
    test=item
    centroids.append(item.representative.coordinate)
    items.append(item.cluster_item)
    i += 1

for j in range(len(centroids)):
    pts = []
    for item in items[j]:
        pts.append(
            [np.linalg.norm(item.coordinate - centroids[0]), np.linalg.norm(item.coordinate - centroids[1]), item.label,
             listColor[int(item.label)]])
    points += pts

columns = ['x', 'y', 'number', 'c']

df = pd.DataFrame(points, columns=columns)

plt.scatter(df['x'], df['y'], color=df['c'])
plt.title("Kmeans for Mnist Dataset")
plt.xlabel("distance from centroid 1")
plt.ylabel("distance from centroid 2")
l = []
for i in range(10):
    red_patch = mpatches.Patch(color=listColor[i], label=i)
    l.append(red_patch)
    plt.legend(handles=l)
plt.show()

"""
mus_imgs = np.asarray([np.reshape(mu.representative.coordinate, (28, 28)) * 255.0 for mu in mnist_lloyd_kmeans.mus.values()])
plot_elements(mus_imgs, 5, 3)
generated_img = LloydKmeans.generate(mnist_lloyd_kmeans.mus['mu1'], mnist_lloyd_kmeans.mus['mu2'], 0.5)
generated_img = np.reshape(generated_img, (28, 28)) * 255.0
Image.fromarray(generated_img).show()
"""


"""
# Kmeans on hand Dataset
dhi = load_sign_language_digits_ds('./../data/Sign-Language-Digits-Dataset-master', (28, 28))
dhi_train = np.reshape(dhi.astype('float32'), (-1, 784)) / 255.0
dhi_lloyd_kmeans = LloydKmeans(dhi_train, 15)
dhi_lloyd_kmeans.fit()
mus_imgs = np.asarray([np.reshape(mu.representative, (28, 28)) * 255.0 for mu in dhi_lloyd_kmeans.mus.values()])
plot_elements(mus_imgs, 5, 3)
generated_img = LloydKmeans.generate(dhi_lloyd_kmeans.mus['mu1'], dhi_lloyd_kmeans.mus['mu2'], 0.5)
generated_img = np.reshape(generated_img, (28, 28)) * 255.0
"""
