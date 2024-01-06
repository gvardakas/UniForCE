from numpy import ndarray

from data.enums.Algorithm_Spanning_Tree import Algorithm_Spanning_Tree
from data.options.SpanningTreeOptions import SpanningTreeOptions
from data.results.MinimumSpanningTreeResult import MinimumSpanningTreeResult
from spanningtree.Kruskal import Kruskal


def spanning_tree(data: ndarray, algorithm: Algorithm_Spanning_Tree, labels: ndarray, cluster_centers: ndarray,
                  number_of_sub_clusters: int, options: SpanningTreeOptions) -> MinimumSpanningTreeResult:
    match algorithm:
        case Algorithm_Spanning_Tree.Kruskal:
            return Kruskal().fit(data=data, sub_predictions=labels, sub_cluster_centers=cluster_centers,
                                 number_of_sub_clusters=number_of_sub_clusters, options=options)
