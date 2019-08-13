#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Multiple implementations of the PCA algorithm
TODO: Implement a generate fot the PCA
"""


import numpy as np


def _calculate_covariance_matrix(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Determinate the covariance matrix of two vector
    :param v1: The first vector
    :param v2: The second vector
    :return: The resultant covariance matrix
    """
    return (1 / (len(v1) - 1)) * sum(((v1[i] - np.mean(v1)) * (v2[i] - np.mean(v2)) for i in range(len(v1))))


# covariance_matrix = np.vectorize(lambda d: np.asarray([[_calculate_covariance_matrix(x, y) for x in d] for y in d]))


class PCA:

    def __init__(self, data: np.ndarray, components=2):
        """
        PCA class initializer
        :param data: The data to compute the PCA on
        :param components: The number of components to extract
        """
        self.data = data.copy()
        self.components = components
        self.centered_data = None
        self.cov_matrix = None
        self.eigen_vec = {'vanilla': None, 'sorted': None}
        self.extracted_data = None

    def fit(self) -> None:
        """
        Fit the Data to with the PCA algorithm with the parameters given with the PCA constructor
        """
        self.centered_data = self.data - np.mean(self.data.T, axis=1)
        # self.cov_matrix = np.asarray([[_calculate_covariance_matrix(x, y)
        #                              for x in self.centered_data] for y in self.centered_data])
        self.cov_matrix = np.matmul(self.centered_data.T, self.centered_data)
        eigen_val, self.eigen_vec['vanilla'] = np.linalg.eigh(self.cov_matrix)
        order = (-eigen_val).argsort()
        self.eigen_vec['sorted'] = np.transpose(self.eigen_vec['vanilla'])[order]
        self.extracted_data = self.eigen_vec['sorted'][0:self.components].dot(self.centered_data.T)

    def plot(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def generate(self, *args, **kwargs) -> None:
        raise NotImplementedError
