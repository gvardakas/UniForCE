from dataclasses import dataclass

from numpy import ndarray


@dataclass
class UniforceResult:
    labels: ndarray
    sub_labels: ndarray
    sub_cluster_centers: ndarray
    adjacency: ndarray
    is_active: ndarray
    clusters: list
    n_clusters: int
    n_sub_clusters: int
