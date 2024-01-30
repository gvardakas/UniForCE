from numpy import ndarray

from code.data.enums.Algorithm_Spanning_Tree import Algorithm_Spanning_Tree
from code.data.options.SpanningTreeOptions import SpanningTreeOptions
from code.data.results.MinimumSpanningTreeResult import MinimumSpanningTreeResult
from code.spanningtree.Kruskal import Kruskal


def spanning_tree(data: ndarray, algorithm: Algorithm_Spanning_Tree, labels: ndarray, cluster_centers: ndarray,
                  number_of_sub_clusters: int, options: SpanningTreeOptions) -> MinimumSpanningTreeResult:
    match algorithm:
        case Algorithm_Spanning_Tree.Kruskal:
            return Kruskal().fit(data=data, sub_predictions=labels, sub_cluster_centers=cluster_centers,
                                 number_of_sub_clusters=number_of_sub_clusters, options=options)
