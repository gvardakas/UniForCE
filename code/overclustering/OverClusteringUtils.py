from numpy import ndarray
from sklearn.cluster import KMeans

from code.data.enums.Algorithm_Over_Clustering import Algorithm_Over_Clustering
from code.data.results.OverClusteringResult import OverClusteringResult
from code.overclustering.GlobalKMeansPp import GlobalKmeansPp


def over_clustering(
        data: ndarray,
        algorithm: Algorithm_Over_Clustering,
        number_of_clusters: int
) -> OverClusteringResult:
    match algorithm:
        case Algorithm_Over_Clustering.GlobalKMeansPp:
            return _execute_global_kmeans_pp(data, number_of_clusters)
        case Algorithm_Over_Clustering.GlobalKMeansPpParallel:
            return _execute_global_kmeans_pp_parallel(data, number_of_clusters)
        case Algorithm_Over_Clustering.KmeansPp:
            return _execute_sklearn_kmeans(data, number_of_clusters)


def _execute_global_kmeans_pp(data: ndarray, number_of_clusters: int) -> OverClusteringResult:
    return GlobalKmeansPp(number_of_clusters=number_of_clusters, n_init=10).fit(data)


def _execute_global_kmeans_pp_parallel(data: ndarray, number_of_clusters: int) -> OverClusteringResult:
    return GlobalKmeansPp(number_of_clusters=number_of_clusters, n_init=10).fit_parallel(data)


def _execute_sklearn_kmeans(data: ndarray, number_of_clusters: int) -> OverClusteringResult:
    selected_model = KMeans(n_clusters=number_of_clusters, n_init=10).fit(data)
    return OverClusteringResult(labels=selected_model.labels_, cluster_centers=selected_model.cluster_centers_)
