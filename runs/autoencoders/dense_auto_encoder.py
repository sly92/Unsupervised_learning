#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Some runs with our CNNAutoEncoder implementations on different data
"""


from keras.datasets import mnist
import numpy as np

from glados.implementations import DenseAutoEncoder
from glados.utils import plot_elements, load_sign_language_digits_ds


(x_train, _), (x_test, _) = mnist.load_data()


# RUN DenseAutoEncoder on Mnist Dataset
xtrain_ae = np.reshape(x_train.astype('float32'), (-1, 784)) / 255.0
xtest_ae = np.reshape(x_test.astype('float32'), (-1, 784)) / 255.0
dense_auto_encoder = DenseAutoEncoder((784,), 784, 2, [256, 128, 64, 32])
dense_auto_encoder.build()
dense_auto_encoder.encoder_decoder.fit(xtrain_ae, xtrain_ae, batch_size=256, epochs=5,
                                       shuffle=True, validation_data=(xtest_ae, xtest_ae))
generated_inputs = np.asarray([[x, y] for x in np.arange(0, 1, 0.1)
                              for y in np.arange(0, 1, 0.1)], dtype=np.float32)
generated_imgs = dense_auto_encoder.decoder.predict(generated_inputs)
plot_elements(np.reshape(generated_imgs, (-1, 28, 28)) * 255.0)


# Run DenseAutoEncoder on deaf hand image dataset
dhi = load_sign_language_digits_ds('./../../data/Sign-Language-Digits-Dataset-master', (28, 28))
dhi_train = np.reshape(dhi.astype('float32'), (-1, 784)) / 255.0
dhi_dense_auto_encoder = DenseAutoEncoder((784,), 784, 2, [128, 64, 32])
dhi_dense_auto_encoder.build()
dhi_dense_auto_encoder.encoder_decoder.fit(dhi_train, dhi_train, batch_size=256,
                                           epochs=1024, shuffle=True, validation_split=0.2)
generated_inputs = np.asarray([[x, y] for x in np.arange(0, 1, 0.1) for y in np.arange(0, 1, 0.1)], dtype=np.float32)
generated_imgs = dhi_dense_auto_encoder.decoder.predict(generated_inputs)
plot_elements(np.reshape(generated_imgs, (-1, 28, 28)) * 255.0)

