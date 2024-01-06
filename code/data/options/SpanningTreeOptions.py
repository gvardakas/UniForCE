from dataclasses import dataclass

from ..enums.Distribution import Distribution
from ..enums.Statistical_Test import Statistical_Test


@dataclass
class SpanningTreeOptions:
    min_samples_per_sub_cluster: int = 25
    majority_n_tests: int = 11
    alpha: float = 1e-3
    statistical_test: Statistical_Test = Statistical_Test.DipTest
    distribution: Distribution = Distribution.Default
    specific_number_of_clusters: int = None
