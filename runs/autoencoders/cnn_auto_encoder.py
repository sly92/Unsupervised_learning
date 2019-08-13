#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Some runs with our CNNAutoEncoder implementations on different data
"""


from keras.datasets import mnist
import numpy as np

from glados.implementations import CNNAutoEncoder
from glados.utils import plot_elements, load_sign_language_digits_ds


(mnist_train, _), (mnist_test, _) = mnist.load_data()


# RUN CNNAutoEncoder on Mnist Dataset
mnist_train_ae = np.reshape(mnist_train.astype('float32'), (-1, 28, 28, 1)) / 255.0
mnist_test_ae = np.reshape(mnist_test.astype('float32'), (-1, 28, 28, 1)) / 255.0
mnist_cnn_auto_encoder = CNNAutoEncoder((28, 28, 1), None, 2, [128, 64, 32])
mnist_cnn_auto_encoder.build()
mnist_cnn_auto_encoder.encoder_decoder.fit(mnist_train_ae, mnist_train_ae, batch_size=256, epochs=5,
                                           shuffle=True, validation_data=(mnist_test_ae, mnist_test_ae))
generated_inputs = np.asarray([[x, y] for x in np.arange(0, 1, 0.1)
                              for y in np.arange(0, 1, 0.1)], dtype=np.float32)
generated_imgs = mnist_cnn_auto_encoder.decoder.predict(generated_inputs)
plot_elements(np.reshape(generated_imgs, (-1, 28, 28)) * 255.0)


# # Run CNNAutoEncoder on deaf hand image dataset
dhi = load_sign_language_digits_ds('./../../data/Sign-Language-Digits-Dataset-master', (28, 28))
dhi_train = np.reshape(dhi.astype('float32'), (-1, 28, 28, 1)) / 255.0
dhi_cnn_auto_encoder = CNNAutoEncoder((28, 28, 1), None, 2, [128, 64, 32])
dhi_cnn_auto_encoder.build()
dhi_cnn_auto_encoder.encoder_decoder.fit(dhi_train, dhi_train, batch_size=256,
                                         epochs=256, shuffle=True, validation_split=0.2)
generated_inputs = np.asarray([[x, y] for x in np.arange(0, 1, 0.1) for y in np.arange(0, 1, 0.1)], dtype=np.float32)
generated_imgs = dhi_cnn_auto_encoder.decoder.predict(generated_inputs)
plot_elements(np.reshape(generated_imgs, (-1, 28, 28)) * 255.0)
