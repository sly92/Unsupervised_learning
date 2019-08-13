#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Multiple implementations of the GAN algorithm
"""


from random import randint
from typing import List, Tuple

from keras import Model
from keras.layers import BatchNormalization, Dense, Dropout, Input, Layer
import numpy as np

from glados.utils import model_builder


class VanillaGAN:

    def __init__(self, data: np.ndarray, latent_space: int, generator: Model = None,
                 discriminator: Model = None):
        """
        Initialize a Vanilla GAN object
        :param data: The real data to use in the training process
        :param latent_space: The size of the entry latent space
        :param generator: Keras model generator to use if any
        :param discriminator: Keras model discriminator to use if any
        """
        self.data = data
        self.latent_space = latent_space
        self.generator_layers = None
        self.generator = generator
        self.discriminator_layers = None
        self.discriminator = discriminator

    def _architecture_builder(self, input_shape: Tuple, neurons: List[int],
                              layer_activation: str, output_activation: str) -> List[Layer]:
        """
        Build the keras architecture for the Dense Vanilla GAN
        :return: The list of layers for the dense vanilla gan
        """
        layers = [Input(input_shape)]
        for n in neurons[:-1]:
            layers.append(Dense(n, activation=layer_activation))
            layers.append(BatchNormalization())
            layers.append(Dropout(0.5))
        layers.append(Dense(neurons[-1], activation=output_activation))
        return layers

    def build(self, what: str, layers: List[int], layers_activation='relu',
              output_activation='relu', optimizer='adam', loss='logcosh', metrics=None) -> None:
        """
        Build a generator or discriminator model and store it inside the VanillaGAN object
        :param what: The type of model to generate (generator or discriminator)
        :param layers: The layers to use (size of each)
        :param layers_activation: The activation function to use within the layers
        :param output_activation: The activation function to use in the output layer
        :param optimizer: The optimizer function to use
        :param loss: The loss function to use
        :param metrics: The metrics
        """
        if what == 'generator':
            self.generator_layers = self._architecture_builder((self.latent_space,), layers,
                                                               layers_activation, output_activation)
            self.generator = model_builder(self.generator_layers)
            self.generator.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        if what == 'discriminator':
            self.discriminator_layers = self._architecture_builder(self.data.shape[1:], layers,
                                                                   layers_activation, output_activation)
            self.discriminator = model_builder(self.discriminator_layers)
            self.discriminator.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def _discriminator_learning(self, batch_size=50, iteration=1, split=0.2, **kwargs) -> None:
        """
        Make the discriminator learn
        :param batch_size: The size of the batch to train on
        :param iteration: The number of iteration to do in the learning process of the discriminator
        :param split: The ratio in which to split the data
        :param kwargs: The keywords arguments to pass to the keras Model.fit method
        """
        print('=== Discriminator Learning ===')
        discriminator_data = self._generate_discriminator_batch(batch_size)
        x_train = [dd[0] for dd in discriminator_data]
        y_train = [dd[1] for dd in discriminator_data]
        self.discriminator.fit(np.asarray(x_train), np.asarray(y_train), epochs=iteration, validation_split=split, **kwargs)

    def _generate_discriminator_batch(self, batch_size: int) -> np.ndarray:
        """
        Generate a batch for the GAN discriminator model with fake and real data shuffled
        :param batch_size: The size of the batch to generate
        :return: A batch shuffled with fake and real data
        """
        data_size = self.data.shape[0]
        random_real_data = [[self.data[randint(0, data_size)], 1] for _ in range(int(batch_size/2))]
        fake_input = np.asarray([[np.random.normal() for _ in range(self.latent_space)] for _ in range(int(batch_size/2))])
        fake_data = [[fd, 0] for fd in self.generator.predict(fake_input, int(batch_size/2), verbose=0)]
        discriminator_data = np.concatenate((random_real_data, fake_data), axis=0)
        np.random.shuffle(discriminator_data)
        return discriminator_data

    def _generator_learning(self, batch_size: int, split=0.2, *args, **kwargs) -> None:
        """
        Make the generator learn
        :param batch_size: The size of the batch to generate
        :param split: The validation split to cut data
        :param args: The args to pass to Keras Model.fit method
        :param kwargs: The keyword args to pass to Keras Model.fit method
        """
        print('=== Generator Learning ===')
        x_train = np.asarray([[np.random.normal() for _ in range(self.latent_space)] for _ in range(int(batch_size))])
        y_train = np.ones(batch_size)
        for l in self.discriminator.layers:
            l.trainable = False
        full_gan_layers = self.generator_layers + self.discriminator_layers[1:]
        full_gan = model_builder(full_gan_layers)
        full_gan.compile(optimizer=self.discriminator.optimizer, loss=self.discriminator.loss,
                         metrics=self.discriminator.metrics)
        full_gan.fit(x_train, y_train, epochs=1, validation_split=split, *args, **kwargs)
        for l in self.discriminator.layers:
            l.trainable = True

    def fit(self, iteration: int, batch_size: int, *args, **kwargs) -> None:
        """
        Fit the DenseEncoderDecoder model to the passed data
        :param iteration: The number of learning iteration to do
        :param batch_size: The batch size for the learning process
        :param args: The args to pass to keras model fit method
        :param kwargs: The keyword args to pass to keras model fit method
        """
        for it in range(iteration):
            print(f'!!!!! Iteration {it} !!!!!')
            self._discriminator_learning(batch_size)
            self._generator_learning(batch_size)
            print('\n ##################### \n')
        print('Done')
        # self.model.fit(*args, *kwargs)

    def plot(self, *args, **kwargs) -> None:
        raise NotImplementedError
