from dataclasses import dataclass

from numpy import ndarray


@dataclass
class MinimumSpanningTreeResult:
    labels: ndarray
    sub_labels: ndarray
    sub_cluster_centers: ndarray
    is_active: ndarray
    adjacency: ndarray
    clusters: list
