import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from diptest import diptest
from collections import OrderedDict, Counter
import networkx as nx

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances


class Global_Kmeans_pp():
    def __init__(self, n_clusters=2, n_init=100, max_iter=300, tol=1e-4, verbose=1):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.cluster_centers_ = OrderedDict()
        self.cluster_distance_space_ = OrderedDict()
        self.inertia_ = OrderedDict()
        self.labels_ = OrderedDict()
        
    def fit(self, X, y=None):
        kmeans = KMeans(n_clusters=1, init='random', n_init=1, tol=self.tol).fit(X)
        self.n_data = X.shape[0]
        self.cluster_centers_[1] = kmeans.cluster_centers_
        self.cluster_distance_space_[1] = kmeans.transform(X).min(axis=1)
        self.labels_[1] = kmeans.labels_
        self.inertia_[1] = kmeans.inertia_
        
        k = 1
        while True:
            if k < self.n_clusters:
                k += 1

                if 0 < self.verbose: print('Solving {:d}-means'.format(k))
                centroid_candidates, _ = self.__kmeans_pp(X, self.cluster_distance_space_[k-1])

                self.inertia_[k] = float('inf')
                for i, xi in enumerate(centroid_candidates): # TODO parallel
                    current_centroids = np.vstack((self.cluster_centers_[k-1], xi))
                    kmeans = KMeans(n_clusters=k, init=current_centroids, n_init=1, tol=self.tol)
                    kmeans = kmeans.fit(X)

                    if kmeans.inertia_ < self.inertia_[k]:
                        self.cluster_centers_[k] = kmeans.cluster_centers_
                        self.labels_[k] = kmeans.labels_
                        self.inertia_[k] = kmeans.inertia_
                        self.cluster_distance_space_[k] = kmeans.transform(X).min(axis=1)
            else:
                break
        
        return self
      
    def predict(self, X):
        return self.labels_, self.cluster_centers_, self.inertia_
    
    def transform(self, X):
        return self.cluster_distance_space_
    
    def __kmeans_pp(self, X, cluster_distance_space):
        cluster_distance_space = np.power(cluster_distance_space, 2).flatten()
        sum_distance = np.sum(cluster_distance_space)
        selection_prob = cluster_distance_space / sum_distance
        selected_indexes = np.random.choice(self.n_data, size=self.n_init, p=selection_prob, replace=False)
        kmeans_pp_selected_centroids = X[selected_indexes]
        return kmeans_pp_selected_centroids, selected_indexes


# Unimodality Testing
def axis_split(X, c_i, c_j, alpha):
    # hyperplane equation wx + b = 0
    w = c_j - c_i
    b = - np.dot(c_i, c_i)
    unnormalized_algebraic_distances_to_hyperplane = np.dot(X, w) + b
    
    dip, pval = diptest(unnormalized_algebraic_distances_to_hyperplane, boot_pval = False, n_boot=10**4)
    #pval = unimodality_test(unnormalized_algebraic_distances_to_hyperplane, method = 'ACR', n = 2**5)[1][0]
    #print(pval)
    
    condition = pval < alpha
    return condition


# Utilities
class UnionFind:    
    def __init__(self, elements):
        self.size_ = len(elements) # number of trees
        self.data_structure_ = [i for i in range(self.size_)]
        self.sizes_ = [1 for i in range(self.size_)] # sizes of trees
        self.element_ = elements # index to element
        self.index_ = {element: index for index, element in enumerate(elements)} # element to index
    
    def union(self, element_1, element_2):
        root_1_index = self.index_[self.find(element_1)]
        root_2_index = self.index_[self.find(element_2)]
        
        size_1 = self.sizes_[root_1_index]
        size_2 = self.sizes_[root_2_index]
        union_size = size_1 + size_2
        
        # connecting the two trees
        if size_1 <= size_2:
            self.data_structure_[root_2_index] = root_1_index
        else:
            self.data_structure_[root_1_index] = root_2_index
        
        self.sizes_[root_1_index] = self.sizes_[root_2_index] = union_size
        self.size_ -= 1
    
    def find(self, element):
        index = self.index_[element]
        path_indices = []
        while self.data_structure_[index] != index:
            path_indices.append(index)
            index = self.data_structure_[index]
        root_index = index
        
        # shortening the tree
        for index in path_indices:
            self.data_structure_[index] = root_index
        
        return self.element_[root_index]
    
    def to_list(self):
        # turning all trees into stars
        for element in self.element_:
            self.find(element)
        
        list_of_lists = [[] for i in range(len(self.element_))]
        for index, element in enumerate(self.element_):
            list_of_lists[self.data_structure_[index]].append(element)
        return [_list for _list in list_of_lists if len(_list) > 0]

def cc_index(cc_list, node):
    for i, cc in enumerate(cc_list):
        if node in cc:
            return i
    return None


def overclustering(data, options):
    if np.isscalar(options['n_subclusters']):
        if options['algorithm'] == 'globalkmeans++':
            selected_model = Global_Kmeans_pp(n_clusters = options['n_subclusters'], n_init = 10).fit(data)
            return selected_model.labels_[selected_model.n_clusters], selected_model.cluster_centers_[selected_model.n_clusters]
        else:
            model = {
                'kmeans++': KMeans(n_clusters = options['n_subclusters'], n_init = 10),
            }
            selected_model = model[options['algorithm']].fit(data)
            return selected_model.labels_, selected_model.cluster_centers_
    else:
        if options['algorithm'] == 'globalkmeans++':
            selected_model = Global_Kmeans_pp(n_clusters = np.max(options['n_subclusters']), n_init = 10).fit(data)
            return selected_model.labels_, selected_model.cluster_centers_
        else:
            labels = {}
            cluster_centers = {}
            for K in options['n_subclusters']:
                model = {
                    'kmeans++': KMeans(n_clusters = K, n_init = 10)
                }
                selected_model = model[options['algorithm']].fit(data)
                labels[K] = selected_model.labels_
                cluster_centers[K] = selected_model.cluster_centers_
            return labels, cluster_centers


def in_same_tree(uf, i, j):
    for this_set in uf.to_sets():
        if i in this_set and j in this_set:
            return True
    return False


def kruskal(data, subpredictions, subcluster_centers, options):
    n_clusters = options['n_clusters']
    n_subclusters = options['n_subclusters']
    min_size = options['min_size']
    distribution = options['distribution']
    alpha = options['alpha']
    n_tests = options['n_tests']
        
    unimodality_test = {'default': axis_split}
    
    adjacency_matrix = lil_matrix(np.zeros([n_subclusters, n_subclusters], dtype=float))
    index_map = {tuple(subcluster_centers[i].tolist()): i for i in range(n_subclusters)}
    
    subcluster_sizes = np.unique(subpredictions, return_counts=True)[1]
    is_active = (subcluster_sizes >= min_size)
    active_subclusters = np.nonzero(is_active)[0]
    union_find = nx.utils.UnionFind(active_subclusters)
    
    edges = [(tuple(subcluster_centers[i].tolist()), tuple(subcluster_centers[j].tolist())) for i in active_subclusters for j in active_subclusters if i < j]
    
    # Sorting edges by their length in
    edges = sorted(edges, key = lambda edge: (edge[0][0] - edge[1][0])**2 + (edge[0][1] - edge[1][1])**2)
    
    # reassigning inactive subcluster samples
    is_inactive = (subcluster_sizes < min_size)
    inactive_subclusters = np.nonzero(is_inactive)[0]
    remaining_samples_indices = np.nonzero(np.isin(subpredictions, inactive_subclusters))[0]
    if remaining_samples_indices.shape[0] > 0:
        subpredictions[remaining_samples_indices] = active_subclusters[np.argmin(euclidean_distances(data[remaining_samples_indices], subcluster_centers[active_subclusters]), axis = 1)]
    
    # recalculating subcluster centers
    for j in range(n_subclusters):
        if is_active[j]:
            subcluster_centers[j, :] = np.average(data[np.where(subpredictions == j)[0]], axis = 0)
        
    for edge in edges:
        i = index_map[edge[0]]
        j = index_map[edge[1]]
        if not in_same_tree(union_find, i, j):
            """
            subcluster_i = data[np.where(subpredictions==i)[0]]
            subcluster_j = data[np.where(subpredictions==j)[0]]
            if should_link(subcluster_i, subcluster_j, alpha):
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
                union_find.union(i, j)
            """
            # Samples of subclusters i and j
            samples_indexes_i = np.where(subpredictions==i)[0]
            samples_indexes_j = np.where(subpredictions==j)[0]
            samples_indexes = np.concatenate((samples_indexes_i, samples_indexes_j))
            X, y = data[samples_indexes], subpredictions[samples_indexes]
            
            # sorting subclusters by sample size
            n_i = samples_indexes_i.shape[0]
            n_j = samples_indexes_j.shape[0]
            if n_j < n_i:
                temp = samples_indexes_i
                samples_indexes_i = samples_indexes_j
                samples_indexes_j = temp
                
                temp = n_i
                n_i = n_j
                n_j = temp
                
                temp = i
                i = j
                j = temp
            """
            voting = [False for ii in range(n_tests)]
            for ii in range(n_tests):
                sub_indexes = np.random.choice(n_j, size=n_i, replace=False)
                samples_indexes = np.concatenate((samples_indexes_i, samples_indexes_j[sub_indexes]))
                X, y = data[samples_indexes], subpredictions[samples_indexes]
                voting[ii] = not unimodality_test[distribution](X, subcluster_centers[i], subcluster_centers[j], alpha)
            if Counter(voting).most_common(1)[0][0] == True:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
                union_find.union(i, j)
            
            """
            for ii in range(n_tests):
                sub_indexes = np.random.choice(n_j, size=n_i, replace=False)
                samples_indexes = np.concatenate((samples_indexes_i, samples_indexes_j[sub_indexes]))
                X, y = data[samples_indexes], subpredictions[samples_indexes]                
                if unimodality_test[distribution](X, subcluster_centers[i], subcluster_centers[j], alpha):
                    adjacency_matrix[i, j] -= 1
                else:
                    adjacency_matrix[i, j] += 1
            adjacency_matrix[j, i] = adjacency_matrix[i, j]
            if adjacency_matrix[i, j] > 0:
                union_find.union(i, j)
            """"""
        
        # early termination if target number of clusters was reached
        if (n_clusters is not None) and (len([cluster for cluster in union_find.to_sets()]) <= n_clusters):
            break
    
    clusters = [cluster for cluster in union_find.to_sets()]
    predictions = np.array([cc_index(clusters, subpredictions[i]) for i in range(data.shape[0])])
        
    return predictions, subpredictions, subcluster_centers, is_active, adjacency_matrix.toarray(), clusters


def kruskal_plus_plus(data, options):
    n_subclusters = options['n_subclusters']
    alpha = options['alpha']
    max_iter = options['max_iter']
    
    print('Starting at ' + str(n_subclusters) + ' centers.')
    for i in range(max_iter):
        options['n_subclusters'] = n_subclusters
        sublabels, subcluster_centers = overclustering(data, options)
        n_subclusters = subcluster_centers.shape[0]
        options['n_subclusters'] = n_subclusters
        labels, sublabels, subcluster_centers, is_active, adjacency_matrix, clusters = kruskal(data, sublabels, subcluster_centers, options)
        if i == max_iter - 1: break
        increment = 0
        for k in range(n_subclusters):
            #if (not muted[k]) and pca_split(data[np.where(sublabels==k)[0]], None, None, alpha):
            if is_active[k] and should_split(data[np.where(sublabels==k)[0]], alpha):
                increment += 1
        if increment > 0:
            print('Adding ' + str(increment) + ' centers.')
            n_subclusters += increment
        else:
            break
    print('Finished at ' + str(n_subclusters) + ' centers.')
    return labels, sublabels, subcluster_centers, is_active, adjacency_matrix, clusters


class UniForCE:
    default_options_ = {
        'algorithm': 'globalkmeans++', # clustering algorithm for initial overclustering from {'kmeans++', 'globalkmeans++'}
        'n_clusters': None, # number of desired clusters. #TODO This is not yet implemented!
        'n_subclusters': 50, # number of overclustering clusters
        'stat_test': 'diptest', # statistical test for unimodality testing
        'distribution': 'default', # distribution to be tested for unimodality from {'default'}
        'min_size': 25, # minimum subcluster size (required for statistical test validity)
        'n_tests': 11, # number of tests on subsampled clusters
        'alpha': 1e-3, # confidence interval
        'max_iter': 1, # maximum number of overclustering iterations
    }
    
    def __init__(self, options):
        self.options_ = self.default_options_
        for key in options.keys():
            self.options_[key] = options[key]
        
    def fit(self, X, y = None):
        if np.isscalar(self.options_['n_subclusters']):
            self.labels_, self.sublabels_, self.subcluster_centers_, self.is_active_, self.adjacency_, self.clusters_ = kruskal_plus_plus(X, self.options_)
            self.n_clusters_ = len(self.clusters_)
            self.n_subclusters_ = self.subcluster_centers_.shape[0]
        else:
            self.labels_ = {}
            self.sublabels_, self.subcluster_centers_ = overclustering(X, self.options_)
            self.is_active_ = {}
            self.adjacency_ = {}
            self.clusters_ = {}
            self.n_clusters_ = {}
            self.n_subclusters_ = {}
            options = dict(self.options_)
            for K in self.options_['n_subclusters']:
                options['n_subclusters'] = K
                self.labels_[K], self.sublabels_[K], self.subcluster_centers_[K], self.is_active_[K], self.adjacency_[K], self.clusters_[K] = kruskal(X, self.sublabels_[K], self.subcluster_centers_[K], options)
                self.n_clusters_[K] = len(self.clusters_[K])
                self.n_subclusters_[K] = self.subcluster_centers_[K].shape[0]
        return self