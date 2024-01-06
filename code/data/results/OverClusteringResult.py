from dataclasses import dataclass

from numpy import ndarray


@dataclass
class OverClusteringResult:
    labels: ndarray
    cluster_centers: ndarray
