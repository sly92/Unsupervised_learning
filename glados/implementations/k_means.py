#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Multiple implementations of the K-Means algorithm
"""


from operator import itemgetter
from random import randint
from typing import Any, List, NamedTuple

import numpy as np
from tqdm import tqdm

__all__ = ['LloydKmeans', 'generate_element']

class Point(NamedTuple):
    coordinate: any
    label: str

class MU(NamedTuple):
    representative: Point
    cluster_item: List[Point]

class Kmeans:

    def __init__(self, data: np.ndarray, labels: np.ndarray, nb_clusters: int, random_mus=False):
        """
        Kmeans Initializers
        TODO: Allow random initialization of mus
        :param data: The Array to compute the K-Means on
        :param nb_clusters: The number of cluster find
        """
        self.data = data.copy()
        self.labels = labels.copy()
        self.nb_clusters = nb_clusters
        if not random_mus:
            self.mus = {f'mu{nc}': MU(Point(data[randint(0, data.shape[0])],''), list()) for nc in range(nb_clusters)}
        if random_mus:
            self.mus = {f'mu{nc}': MU(Point(np.asarray([np.random.normal() for _ in range(data.shape[1])]),''), list())
                        for nc in range(nb_clusters)}

    @staticmethod
    def generate(mu1: np.ndarray, mu2: np.ndarray, gradient: float) -> np.ndarray:
        """
        Generate a random element
        :param mu1: The first mu to base the generation on
        :param mu2: The second mu to base the generation on
        :param gradient: The gradient between two centroid [0:1]
        :return: The generated element
        """
        return mu1 * gradient + mu2 * (1 - gradient)

    def fit(self, max_iteration=100) -> None:
        raise NotImplementedError

    def plot(self, **kwargs) -> None:
        raise NotImplementedError


class LloydKmeans(Kmeans):

    def __init__(self, data: np.ndarray, labels: np.ndarray, nb_clusters: int, random_mus=False):
        """
        LloydKmeans Initializers
        :param data: The Array to compute the K-Means on
        :param nb_clusters: The number of cluster find
        """
        super(LloydKmeans, self).__init__(data, labels, nb_clusters, random_mus)

    def fit(self, max_iteration=100) -> None:
        """
        Fit the Data to with the Lloyd algorithm with the parameters given with the Kmeans constructor
        :param max_iteration: The maximum number of iteration to run the Kmeans
        """
        iteration = 1
        while "we haven't find each mu":
            print(f'Iteration {iteration}')
            pbar = tqdm(total=100)
            for i in range(len(self.data)):
                mu_distance = ((mu_id, np.linalg.norm(self.data[i] - mu.representative.coordinate)) for mu_id, mu in
                               self.mus.items())
                min_mu_id = sorted(mu_distance, key=itemgetter(1))[0][0]
                self.mus[min_mu_id].cluster_item.append(Point(self.data[i], self.labels[i]))
                pbar.update((1 / len(self.data)) * 100)
            pbar.close()
            new_mus = {mu_id: MU(Point(np.mean([item.coordinate for item in mu.cluster_item], axis=0), ''), list()) for
                       mu_id, mu in self.mus.items()}
            is_finished = all(
                ((self.mus[mu_id].representative.coordinate == new_mus[mu_id].representative.coordinate).all() for mu_id
                 in self.mus.keys()))
            if is_finished or iteration > max_iteration:
                break
            self.mus = new_mus
            iteration += 1
