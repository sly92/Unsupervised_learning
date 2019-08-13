#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Some runs with our GANs implementations on different data
"""

from keras.datasets import mnist
import numpy as np

from glados.implementations import VanillaGAN
from glados.utils import plot_elements


(x_train, _), (x_test, _) = mnist.load_data()


# RUN VanillaGAN on Mnist Dataset
latent_space = 13
batch_size = 32
gan_data = np.reshape(x_train.astype('float32'), (-1, 784)) / 255.0
gan = VanillaGAN(gan_data, latent_space)
gan.build('generator', [128, 256, 784], output_activation='tanh')
gan.build('discriminator', [128, 256, 1], loss='binary_crossentropy', metrics=['binary_accuracy'])
gan.fit(128, batch_size)
generated_inputs = np.asarray([[np.random.normal() for _ in range(latent_space)] for _ in range(batch_size)])
generated_imgs = gan.generator.predict(generated_inputs)
plot_elements(np.reshape(generated_imgs, (-1, 28, 28)) * 255.0, 4, 8)
