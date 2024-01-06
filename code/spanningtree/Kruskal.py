from typing import Any, Optional

import networkx as nx
import numpy as np
from diptest import diptest
from networkx.utils import UnionFind
from numpy import ndarray
from scipy.sparse import lil_matrix
from sklearn.metrics import euclidean_distances

from code.data.enums.Distribution import Distribution
from code.data.enums.Statistical_Test import Statistical_Test
from code.data.options.SpanningTreeOptions import SpanningTreeOptions
from code.data.results.MinimumSpanningTreeResult import MinimumSpanningTreeResult


class Kruskal:
    __sub_predictions: ndarray
    __sub_cluster_centers: ndarray
    __sub_cluster_sizes: ndarray
    __adjacency_matrix: lil_matrix
    __index_map: dict[tuple, int]
    __sub_cluster_sizes: ndarray
    __is_active: ndarray[Any, np.dtype[np.bool_]]
    __active_sub_clusters: ndarray
    __union_find: UnionFind
    __edges: list[tuple[tuple, tuple]]

    def __init__(
            self,
            number_of_clusters: int = 2,
            n_init: int = 100,
            max_iter: int = 300,
            tol: float = 1e-4,
            verbose: int = 1
    ):
        self.n_clusters = number_of_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.adjacency_matrix = None

    def fit(
            self,
            data: ndarray,
            sub_predictions: ndarray,
            sub_cluster_centers: ndarray,
            number_of_sub_clusters: int,
            options: SpanningTreeOptions
    ) -> MinimumSpanningTreeResult:
        self._initialize(number_of_sub_clusters, sub_cluster_centers, sub_predictions,
                         options.min_samples_per_sub_cluster)
        self._initialize_edges(sub_cluster_centers)
        self._reassign_inactive_sub_cluster_samples(data, options.min_samples_per_sub_cluster, sub_cluster_centers)
        self._recalculate_sub_cluster_centers(number_of_sub_clusters, data)

        self._connect_edges(data, options)

        clusters = [cluster for cluster in self.__union_find.to_sets()]
        predictions = np.array(
            [self._connected_components_index(clusters, self.__sub_predictions[i]) for i in range(data.shape[0])])

        return MinimumSpanningTreeResult(predictions, self.__sub_predictions, self.__sub_cluster_centers,
                                         self.__is_active,
                                         self.__adjacency_matrix.toarray(),
                                         clusters)

    def _initialize(
            self,
            number_of_sub_clusters: int,
            sub_cluster_centers: ndarray,
            sub_predictions: ndarray,
            min_samples_per_sub_cluster: int
    ) -> None:
        self.__adjacency_matrix = lil_matrix(np.zeros([number_of_sub_clusters, number_of_sub_clusters], dtype=float))
        self.__index_map = {tuple(sub_cluster_centers[i].tolist()): i for i in range(number_of_sub_clusters)}
        self.__sub_cluster_sizes = np.unique(sub_predictions, return_counts=True)[1]
        self.__is_active = (self.__sub_cluster_sizes >= min_samples_per_sub_cluster)
        self.__active_sub_clusters = np.nonzero(self.__is_active)[0]
        self.__union_find = nx.utils.UnionFind(self.__active_sub_clusters)
        self.__sub_cluster_centers = sub_cluster_centers
        self.__sub_predictions = sub_predictions

    def _initialize_edges(
            self,
            sub_cluster_centers: ndarray
    ) -> None:
        print("Initializing edges...")
        self.__edges = [
            (tuple(sub_cluster_centers[i].tolist()), tuple(sub_cluster_centers[j].tolist()))
            for i in self.__active_sub_clusters
            for j in self.__active_sub_clusters if i < j
        ]
        # Sorting edges by their length in
        print("Sorting edges...")
        self.__edges = sorted(self.__edges,
                              key=lambda edge: (edge[0][0] - edge[1][0]) ** 2 + (edge[0][1] - edge[1][1]) ** 2)

    def _reassign_inactive_sub_cluster_samples(
            self,
            data: ndarray,
            min_samples_per_sub_cluster: int,
            sub_cluster_centers: ndarray
    ) -> None:
        print("Reassigning inactive sub clusters...")
        is_inactive = (self.__sub_cluster_sizes < min_samples_per_sub_cluster)
        inactive_sub_clusters = np.nonzero(is_inactive)[0]
        remaining_samples_indices = np.nonzero(np.isin(self.__sub_predictions, inactive_sub_clusters))[0]
        if remaining_samples_indices.shape[0] > 0:
            self.__sub_predictions[remaining_samples_indices] = self.__active_sub_clusters[
                np.argmin(
                    euclidean_distances(data[remaining_samples_indices],
                                        sub_cluster_centers[self.__active_sub_clusters]),
                    axis=1)]

    def _recalculate_sub_cluster_centers(
            self,
            number_of_sub_clusters: int,
            data: ndarray
    ) -> None:
        print("Recalculating sub cluster centers...")
        for j in range(number_of_sub_clusters):
            if self.__is_active[j]:
                self.__sub_cluster_centers[j, :] = np.average(data[np.where(self.__sub_predictions == j)[0]], axis=0)

    def _reached_target_number_of_clusters(
            self,
            specific_number_of_clusters: int
    ) -> bool:
        return (specific_number_of_clusters is not None
                and len([cluster for cluster in self.__union_find.to_sets()]) <= specific_number_of_clusters)

    def _distribution_calculation(
            self,
            distribution: Distribution,
            data: ndarray,
            i: int,
            j: int,
            alpha: float,
            statistical_method: Statistical_Test
    ) -> None:
        match distribution:
            case Distribution.Default:
                if self._axis_split(data, self.__sub_cluster_centers[i], self.__sub_cluster_centers[j], alpha,
                                    statistical_method):
                    self.__adjacency_matrix[i, j] -= 1
                else:
                    self.__adjacency_matrix[i, j] += 1

    def _connect_edges(
            self,
            data: ndarray,
            options: SpanningTreeOptions
    ) -> None:
        print("Connecting edges...")
        for edge in self.__edges:
            i: int = self.__index_map[edge[0]]
            j: int = self.__index_map[edge[1]]
            if not self._in_same_tree(i, j):
                # Samples of sub clusters i and j
                samples_sub_cluster_i: ndarray = np.where(self.__sub_predictions == i)[0]
                samples_sub_cluster_j: ndarray = np.where(self.__sub_predictions == j)[0]
                # samples_indexes: ndarray = np.concatenate((samples_sub_cluster_i, samples_sub_cluster_j))
                # X, y = data[samples_indexes], sub_predictions[samples_indexes]
                self._sorting_sub_clusters_by_sample_size(
                    samples_sub_cluster_i,
                    samples_sub_cluster_j,
                    i,
                    j,
                    data,
                    options.majority_n_tests,
                    options.distribution,
                    options.statistical_test,
                    options.alpha
                )

                self.__adjacency_matrix[j, i] = self.__adjacency_matrix[i, j]
                if self.__adjacency_matrix[i, j] > 0:
                    self.__union_find.union(i, j)

            # Early termination
            if self._reached_target_number_of_clusters(options.specific_number_of_clusters):
                break

    def _sorting_sub_clusters_by_sample_size(
            self,
            samples_sub_cluster_i: ndarray,
            samples_sub_cluster_j: ndarray,
            i: int,
            j: int,
            data: ndarray,
            majority_n_tests: int, distribution: Distribution,
            statistical_method: Statistical_Test,
            alpha: float
    ) -> None:
        n_i = samples_sub_cluster_i.shape[0]
        n_j = samples_sub_cluster_j.shape[0]
        if n_j < n_i:
            temp = samples_sub_cluster_i
            samples_sub_cluster_i = samples_sub_cluster_j
            samples_sub_cluster_j = temp

            temp = n_i
            n_i = n_j
            n_j = temp

            temp = i
            i = j
            j = temp
        for ii in range(majority_n_tests):
            sub_indexes = np.random.choice(n_j, size=n_i, replace=False)
            samples_indexes = np.concatenate((samples_sub_cluster_i, samples_sub_cluster_j[sub_indexes]))
            x, _ = data[samples_indexes], self.__sub_predictions[samples_indexes]
            self._distribution_calculation(distribution, x, i, j, alpha, statistical_method)

    def _in_same_tree(
            self,
            i: int,
            j: int
    ) -> bool:
        for this_set in self.__union_find.to_sets():
            if i in this_set and j in this_set:
                return True
        return False

    # Uni modality Testing
    def _axis_split(
            self,
            X: ndarray,
            c_i: ndarray,
            c_j: ndarray,
            alpha: float,
            method: Statistical_Test):
        # hyperplane equation wx + b = 0
        w = c_j - c_i
        b = - np.dot(c_i, c_i)
        un_normalized_algebraic_distances_to_hyperplane = np.dot(X, w) + b
        p_val = self._statistical_test(method, un_normalized_algebraic_distances_to_hyperplane)
        condition = p_val < alpha
        return condition

    @staticmethod
    def _statistical_test(
            method: Statistical_Test,
            un_normalized_algebraic_distances_to_hyperplane: ndarray
    ) -> float:
        match method:
            case Statistical_Test.DipTest:
                _, p_val = diptest(un_normalized_algebraic_distances_to_hyperplane, boot_pval=False, n_boot=10 ** 4)
                return p_val

    @staticmethod
    def _connected_components_index(
            cc_list: list,
            node: ndarray
    ) -> Optional[int]:
        for i, cc in enumerate(cc_list):
            if node in cc:
                return i
        return None
