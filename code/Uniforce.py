from typing import Optional

from numpy import ndarray

from data.options.UniforceOptions import UniforceOptions
from data.results.MinimumSpanningTreeResult import MinimumSpanningTreeResult
from data.results.OverClusteringResult import OverClusteringResult
from data.results.UniforceResult import UniforceResult
from overclustering.OverClusteringUtils import over_clustering
from spanningtree.MinimumSpanningTreeUtils import spanning_tree


class Uniforce:

    def __init__(self, options=UniforceOptions):
        self._options = options

    def fit(self, X: ndarray) -> Optional[UniforceResult]:
        print(f"Starting Over Clustering ({self._options.algorithm_over_clustering.name})...")
        over_clustering_result: OverClusteringResult = over_clustering(
            data=X,
            algorithm=self._options.algorithm_over_clustering,
            number_of_clusters=self._options.number_of_clusters)
        print("Finished Over Clustering.")

        print(f"Starting Minimum Spanning Tree ({self._options.algorithm_spanning_tree.name})...")
        spanning_tree_result: MinimumSpanningTreeResult = spanning_tree(
            data=X,
            algorithm=self._options.algorithm_spanning_tree,
            labels=over_clustering_result.labels,
            cluster_centers=over_clustering_result.cluster_centers,
            number_of_sub_clusters=self._options.number_of_clusters,
            options=self._options.spanning_tree_options
        )
        print("Finished Minimum Spanning Tree.")

        return UniforceResult(
            labels=spanning_tree_result.labels,
            sub_labels=spanning_tree_result.sub_labels,
            sub_cluster_centers=spanning_tree_result.sub_cluster_centers,
            adjacency=spanning_tree_result.adjacency,
            is_active=spanning_tree_result.is_active,
            clusters=spanning_tree_result.clusters,
            n_clusters=len(spanning_tree_result.clusters),
            n_sub_clusters=spanning_tree_result.sub_cluster_centers.shape[0]
        )
