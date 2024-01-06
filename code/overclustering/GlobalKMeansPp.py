import concurrent.futures
from dataclasses import dataclass
from functools import partial
from typing import Optional

import numpy as np
from numpy import ndarray
from sklearn.cluster import KMeans

from data.results.OverClusteringResult import OverClusteringResult


@dataclass
class Cluster:
    cluster_centers: ndarray = None
    cluster_distance_space: ndarray = None
    labels: ndarray = None
    inertia: float = float('inf')


class GlobalKmeansPp:
    def __init__(self, number_of_clusters=2, n_init=100, max_iter=300, tol=1e-4, verbose=1):
        self.n_clusters = number_of_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def fit(self, X) -> OverClusteringResult:
        current_cluster = self.__initial_fit(X)
        for k in range(2, self.n_clusters + 1):
            self.__print_progress(k)
            previous_cluster = current_cluster
            centroid_candidates = self.__fit_kmeans_pp(X, previous_cluster.cluster_distance_space)
            current_cluster = Cluster()
            for i, xi in enumerate(centroid_candidates):
                kmeans_result = self.__fit_kmeans(X, k, xi, previous_cluster, current_cluster.inertia)
                if kmeans_result is not None:
                    current_cluster = kmeans_result
        return OverClusteringResult(labels=current_cluster.labels,
                                    cluster_centers=current_cluster.cluster_centers)

    def fit_parallel(self, X) -> OverClusteringResult:
        current_cluster = self.__initial_fit(X)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for k in range(2, self.n_clusters + 1):
                self.__print_progress(k)
                previous_cluster = current_cluster
                centroid_candidates = self.__fit_kmeans_pp(X, previous_cluster.cluster_distance_space)
                fit_partial = partial(self.__fit_kmeans_parallel, X, k, previous_cluster)
                results = list(executor.map(fit_partial, centroid_candidates))
                current_cluster = min(results, key=lambda result: result.inertia)
        return OverClusteringResult(labels=current_cluster.labels,
                                    cluster_centers=current_cluster.cluster_centers)

    def __fit_kmeans_pp(self, X, cluster_distance_space) -> ndarray:
        cluster_distance_space = np.power(cluster_distance_space, 2).flatten()
        sum_distance = np.sum(cluster_distance_space)
        selection_prob = cluster_distance_space / sum_distance
        selected_indexes = np.random.choice(X.shape[0], size=self.n_init, p=selection_prob, replace=False)
        kmeans_pp_selected_centroids = X[selected_indexes]
        return kmeans_pp_selected_centroids

    def __initial_fit(self, X) -> Cluster:
        kmeans = KMeans(n_clusters=1, init='random', n_init=1, tol=self.tol).fit(X)
        return Cluster(cluster_centers=kmeans.cluster_centers_,
                       cluster_distance_space=kmeans.transform(X).min(axis=1),
                       inertia=kmeans.inertia_,
                       labels=kmeans.labels_)

    def __fit_kmeans(self, X: ndarray, k: int, xi, cluster: Cluster, inertia: float) -> Optional[Cluster]:
        current_centroids = np.vstack((cluster.cluster_centers, xi))
        kmeans = KMeans(n_clusters=k, init=current_centroids, n_init=1, tol=self.tol)
        kmeans = kmeans.fit(X)
        if kmeans.inertia_ < inertia:
            return Cluster(
                cluster_centers=kmeans.cluster_centers_,
                cluster_distance_space=kmeans.transform(X).min(axis=1),
                labels=kmeans.labels_,
                inertia=kmeans.inertia_)
        else:
            return None

    def __fit_kmeans_parallel(self, X: ndarray, k: int, cluster: Cluster, xi) -> Cluster:
        current_centroids = np.vstack((cluster.cluster_centers, xi))
        kmeans = KMeans(n_clusters=k, init=current_centroids, n_init=1, tol=self.tol)
        kmeans = kmeans.fit(X)
        return Cluster(
            cluster_centers=kmeans.cluster_centers_,
            cluster_distance_space=kmeans.transform(X).min(axis=1),
            labels=kmeans.labels_,
            inertia=kmeans.inertia_)

    def __print_progress(self, k: int):
        if k != self.n_clusters:
            print(f"Solving {k}-means.", end='\r')
        else:
            print(f"Solving last {k}-means.", end='\n')
