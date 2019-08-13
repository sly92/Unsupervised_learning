#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Multiple implementations of the AutoEncoders algorithm
"""


from functools import reduce
from operator import mul
from typing import Callable, List, Tuple, Union

from keras.layers import Conv2D, Dense, Flatten, Input, Layer, MaxPooling2D, UpSampling2D, Reshape

from glados.utils import model_builder


__all__ = ['DenseAutoEncoder', 'CNNAutoEncoder']


class AutoEncoder:

    def __init__(self, input_shape: Tuple, output_shape: int, encoder_size: int, neurons: List[int],
                 optimizer: Union[Callable, str] = 'adam', loss: Union[Callable, str] = 'logcosh',
                 encoder_layer_activation='relu', encoder_output_activation='relu',
                 decoder_layer_activation='relu', decoder_output_activation='relu'):
        """
        Initializer for the AutoEncoder class
        :param input_shape: The input shape of the data
        :param output_shape: The output shape of the data
        :param encoder_size: The size of the encoder layer
        :param neurons: The number of neurons per layers
        :param optimizer: The optimizer function to use
        :param loss: The loss function to use
        :param encoder_layer_activation: The activation function to use for the encoder in the hidden layers
        :param encoder_output_activation: The activation function to use for the encoder in the output layer
        :param decoder_layer_activation: The activation function to use for the decoder in the hidden layers
        :param decoder_output_activation: The activation function to use for the decoder in the output layer
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.encoder_size = encoder_size
        self.neurons = neurons
        self.optimizer = optimizer
        self.loss = loss
        self.encoder_layer_activation = encoder_layer_activation
        self.encoder_output_activation = encoder_output_activation
        self.decoder_layer_activation = decoder_layer_activation
        self.decoder_output_activation = decoder_output_activation
        self.encoder = None
        self.decoder = None
        self.encoder_decoder = None

    def fit(self, *args, **kwargs) -> None:
        """
        Fit the DenseEncoderDecoder model to the passed data
        :param args: The args to pass to keras model fit method
        :param kwargs: The keyword args to pass to keras model fit method
        """
        self.encoder_decoder.fit(*args, *kwargs)

    def _architecture_builder(self, what: str) -> None:
        raise NotImplementedError

    def build(self) -> None:
        raise NotImplementedError

    def plot(self, *args, **kwargs) -> None:
        raise NotImplementedError


class DenseAutoEncoder(AutoEncoder):

    def __init__(self, *args, **kwargs):
        """
        Initializer for the DenseAutoEncoder class
        :param args: The args to pass to the AutoEncoder super class
        :param kwargs: The args to pass to the AutoEncoder super class
        """
        super(DenseAutoEncoder, self).__init__(*args, **kwargs)

    def _architecture_builder(self, what: str) -> List[Layer]:
        """
        Build the keras architecture for the Dense
        :param what: The type of architecture to build (encoder or decoder)
        :return: The list of layers for the dense autoencoder
        """
        def encoder_builder():
            layers = [Input(self.input_shape)]
            layer_activation = self.encoder_layer_activation
            output_activation = self.encoder_output_activation
            for n in self.neurons:
                layers.append(Dense(n, activation=layer_activation))
            layers.append(Dense(self.encoder_size, activation=output_activation))
            return layers

        def decoder_builder():
            layers = [Input((self.encoder_size,))]
            layer_activation = self.decoder_layer_activation
            output_activation = self.decoder_output_activation
            for n in self.neurons[::-1]:
                layers.append(Dense(n, activation=layer_activation))
            layers.append(Dense(self.output_shape, activation=output_activation))
            return layers

        return encoder_builder() if what == 'encoder' else decoder_builder()

    def build(self) -> None:
        """
        Build the three models aka the AutoEncoder, encoder and decoder
        """
        encoder_layers = self._architecture_builder('encoder')
        decoder_layers = self._architecture_builder('decoder')
        encoder_decoder_layers = encoder_layers + decoder_layers[1:]
        encoder = model_builder(encoder_layers)
        decoder = model_builder(decoder_layers)
        encoder_decoder = model_builder(encoder_decoder_layers)
        encoder_decoder.compile(optimizer=self.optimizer, loss=self.loss)
        self. encoder_decoder = encoder_decoder
        self.encoder = encoder
        self.decoder = decoder


class CNNAutoEncoder(AutoEncoder):

    def __init__(self, *args, **kwargs):
        """
        Initializer for the CNNAutoEncoder class
        :param args: The args to pass to the AutoEncoder super class
        :param kwargs: The args to pass to the AutoEncoder super class
        """
        super(CNNAutoEncoder, self).__init__(*args, **kwargs)

    def _architecture_builder(self, what: str, conv: Tuple[int, int],
                              decoder_entry_shape: Tuple = None, padding='same')\
            -> List[Layer]:
        """
        Build the keras architecture for the Dense
        :param what: The type of architecture to build (encoder or decoder)
        :param conv: The conv to apply in the Conv2D layers
        :param decoder_entry_shape: The entry shape to use for the decoder first layers
        :param padding: The padding to use in the conv layers
        :return: The list of layers for the cnn autoencoder
        """
        def encoder_builder():
            layers = [Input(self.input_shape)]
            for n in self.neurons:
                layers.append(Conv2D(n, conv, activation=self.encoder_layer_activation, padding=padding))
                if n != self.neurons[-1]:
                    layers.append(MaxPooling2D((2, 2), padding=padding))
            layers += [Flatten(), Dense(self.encoder_size, activation=self.encoder_output_activation)]
            return layers

        def decoder_builder():
            flat_reshape = int(reduce(mul, decoder_entry_shape, 1))
            layers = [Input((self.encoder_size,)), Dense(flat_reshape, activation=self.decoder_layer_activation),
                      Reshape(decoder_entry_shape)]
            for n in self.neurons[::-1]:
                layers.append(Conv2D(n, conv, activation=self.decoder_layer_activation, padding=padding))
                if n != self.neurons[::-1][-1]:
                    layers.append(UpSampling2D((2, 2)))
            layers.append(Conv2D(1, conv, padding=padding, activation=self.decoder_output_activation))
            return layers

        return encoder_builder() if what == 'encoder' else decoder_builder()

    def build(self, conv=(3, 3)) -> None:
        """
        Build the three models aka the AutoEncoder, encoder and decoder
        :param conv: The conv to apply in the Conv2D layers
        """
        encoder_layers = self._architecture_builder('encoder', conv)
        encoder = model_builder(encoder_layers)
        decoder_entry_shape = encoder.layers[-3].output_shape[1:]
        decoder_layers = self._architecture_builder('decoder', conv, decoder_entry_shape=decoder_entry_shape)
        decoder = model_builder(decoder_layers)
        encoder_decoder_layers = encoder_layers + decoder_layers[1:]
        encoder_decoder = model_builder(encoder_decoder_layers)
        encoder_decoder.compile(optimizer=self.optimizer, loss=self.loss)
        self.encoder_decoder = encoder_decoder
        self.encoder = encoder
        self.decoder = decoder
