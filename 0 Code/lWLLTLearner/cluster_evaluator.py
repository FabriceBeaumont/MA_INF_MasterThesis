from typing import Dict, Tuple, List, Any
import numpy as np
from logging import ERROR as LOG_ERR
from dataclasses import dataclass
from itertools import combinations
import my_consts as c
from my_utils.RuntimeTimer import convert_s_to_h_m_s
# For the computation of the distance matrix.
from scipy.spatial.distance import pdist, squareform
from time import time

# https://ysig.github.io/GraKeL/0.1a8/generated/grakel.cross_validate_Kfold_SVM.html
from grakel.utils import cross_validate_Kfold_SVM

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score
# Mean intra-cluster distance (a) and the mean nearest-cluster distance (b)
# The Silhouette Coefficient for a sample is (b - a) / max(a, b)
# Best 1, Worst -1, Overlapping 0
from sklearn.metrics import silhouette_score as sklearn_silhouette_score 
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html#sklearn.metrics.calinski_harabasz_score
# The higher, the better (dense, well separated clusters)
from sklearn.metrics import calinski_harabasz_score as sklearn_calinski_harabasz_score
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html#sklearn.metrics.davies_bouldin_score
# Average similarity of each cluster with its most similar cluster (ratio of within-cluster distances to between-cluster distances)
# The smaler the better. Minimum zero.
from sklearn.metrics import davies_bouldin_score as sklearn_davies_bouldin_score

from my_utils.decorators import get_logger
LOG = get_logger(filename="Loggers/cluster_evaluations.txt", level=LOG_ERR, create_file=True)

@dataclass
class ClusterStats():
    """
    This class is a container for different data. It can be retrieved with a name/description and this is suitable for plotting.
    """
    _single_names_list = [
        c.C_CL_DIST_INTER_SUM,
        c.C_CL_DIST_INTRA_SUM,
        c.C_CL_SAMPLE_MOVEMENT_ERROR,
        c.C_SCORE_SILHOUETTE, 
        c.C_SCORE_DAVIES_BOULDIN, 
        c.C_SCORE_CALINSKI_HARABASZ, 
        c.C_SVM_ACC, 
        c.C_SVM_STD_DEV
    ]
    _constructed_names_list = [
        c.C_CL_DIST_INTRA_MEAN_Cl, 
        c.C_CL_DIST_INTRA_MAX_Cl, 
        c.C_CL_DIST_INTRA_MIN_Cl, 
        c.C_CL_DIST_INTER_MEAN_Cls, 
        c.C_CL_DIST_INTER_MAX_Cls, 
        c.C_CL_DIST_INTER_MIN_Cls
    ]
    _names_list = _single_names_list + _constructed_names_list    

    def __init__(self, classes_vec: np.array, cluster_classes_set: np.array = None, pairw_diff_cl_ids: List[Tuple[int, int]] = None):
        # A list of all different clusters for every sample.
        self.classes_vec: np.array = classes_vec        
        # The number of samples.
        self.n_samples: int = len(self.classes_vec)
        # A list of all different clusters.
        if cluster_classes_set is None: cluster_classes_set = np.unique(self.classes_vec)
        self.different_classes: np.array = cluster_classes_set
        # A list storing each pair of cluster names once (such that the smaller name comes first).
        if pairw_diff_cl_ids is None: pairw_diff_cl_ids = list(combinations(self.different_classes, 2))
        self.pairwise_different_cl_ids: List[Tuple[int, int]] = pairw_diff_cl_ids

        self._values_dict = self._initialize_values_dict()

    def _initialize_values_dict(self) -> Dict[str, Any]:
        def _construct_keys() -> List[str]:
            keys_list: List[str] = []
            # Add the column names for INTRA cluster distances.
            for cl_name in self.different_classes:
                keys_list.append(f"{c.C_CL_DIST_INTRA_MAX_Cl}_{cl_name}")
                keys_list.append(f"{c.C_CL_DIST_INTRA_MIN_Cl}_{cl_name}")
                keys_list.append(f"{c.C_CL_DIST_INTRA_MEAN_Cl}_{cl_name}")
            # Add the column names for INTER cluster distances.
            for cl_tuple in self.pairwise_different_cl_ids:
                cl_tuple_str = f"{cl_tuple[0]}_{cl_tuple[1]}"
                keys_list.append(f"{c.C_CL_DIST_INTER_MAX_Cls}_" + cl_tuple_str)
                keys_list.append(f"{c.C_CL_DIST_INTER_MIN_Cls}_" + cl_tuple_str)
                keys_list.append(f"{c.C_CL_DIST_INTER_MEAN_Cls}_" + cl_tuple_str)
            # Add the column names for all sklearn scores.
            keys_list += self._single_names_list
            return keys_list

        keys_lst = _construct_keys()
        return dict(zip(keys_lst, [None] * len(keys_lst)))

    def reset_values(self) -> None:
        self._initialize_values_dict()
                  
    def set_value(self, name: str, value: Any, dict_key: Any = None) -> None:
        if dict_key is not None:
            if type(dict_key) is tuple:
                name = f"{name}_{dict_key[0]}_{dict_key[1]}"
            else:
                name = f"{name}_{dict_key}"

        if name in self._values_dict.keys():
            self._values_dict[name] = value
        else:
            LOG.error(f"Unknown data named '{name}' could not be set to value {value}!\nKnown params are:\n\t{self._names_list}")
            
    def get_value(self, name: str) -> None:
        if name in self._values_dict.keys():
            return self._values_dict[name]
        else:
            LOG.error(f"Unknown hyperparameter named '{name}'!\nKnown params: {self._names_list}")

    def get_name_value_list(self) -> List[Tuple[str, Any]]:
        return zip(list(self._values_dict.keys()), list(self._values_dict.values()))

    def get_name_value_tostring(self) -> str:        
        ret_str = ""
        for k, v in self._values_dict.items():
            ret_str += f"{k}:\t{v}\n"
        return ret_str

    def get_all_stat_names(self) -> List[str]:
        return list(self._values_dict.keys())

class ClusterEvaluator():
    """
    This class has the ability to compute and save different scores and statistics of clustered data.
    """
    def __init__(self, classes_vec: np.array, cluster_classes_set: np.array = None, pairw_diff_cl_ids: List[Tuple[int, int]] = None):
        # A list of all different clusters for every sample.
        self.classes_vec: np.array = classes_vec        
        # The number of samples.
        self.n_samples: int = len(self.classes_vec)
        # A list of all different clusters.
        if cluster_classes_set is None: cluster_classes_set = np.unique(self.classes_vec)
        self.different_classes: np.array = cluster_classes_set
        # A list storing each pair of cluster names once (such that the smaller name comes first).
        if pairw_diff_cl_ids is None: pairw_diff_cl_ids = list(combinations(self.different_classes, 2))
        self.pairwise_different_cl_ids: List[Tuple[int, int]] = pairw_diff_cl_ids
        
        self.cluster_stats: ClusterStats = ClusterStats(self.classes_vec, self.different_classes, self.pairwise_different_cl_ids)

        # Store in a dictionary which rows and columns correspond to which classes.
        # This can be used for example to partition the computed distance matrix in clusters.
        self.cluster_rows: Dict[int, List[int]] = {}
        for cl_name in self.different_classes:
            self.cluster_rows[cl_name] = np.where(self.classes_vec == cl_name)[0]
    
    def get_all_names(self) -> List[str]:
        return self.cluster_stats.get_all_stat_names()

    def get_name_value_list(self) -> List[Tuple[str, Any]]:
        return self.cluster_stats.get_name_value_list()

    def reset_cluster_stats(self) -> None:
        self.cluster_stats.reset_values()

    def save_intra_cl_maxminmean(self, D: np.array, dist_mat_ids: np.array) -> float:
        # Sum up the distance between all samples, from same clusters. This value shall decrease with the training.
        same_samples_distance: float = 0.0        

        for cl_name in self.different_classes:
            # Get the rows and columns of this cluster.            
            cl_ids = self.cluster_rows.get(cl_name)            
            # The matrix may be only partially computed. In this case, we need to map the cluster indices to the ones that actually occur in D.
            cl_ids_local = np.in1d(dist_mat_ids, cl_ids, assume_unique=True).nonzero()[0]
            # Extract the distance matrix for all samples in this cluster.
            cl_dist_mat = D[np.ix_(cl_ids_local, cl_ids_local)]
            # Extract the upper triangle (without the diagonal) for this cluster.
            cl_data = cl_dist_mat[np.triu_indices(cl_dist_mat.shape[0], k=1)]
            if len(cl_data) < 1: cl_data = cl_dist_mat
            # Store the statistics for this cluster.
            c_max, c_min, c_mean = round(cl_data.max(), c.N_DECIMALS), round(cl_data.min(), c.N_DECIMALS), round(cl_data.mean(), c.N_DECIMALS)
            self.cluster_stats.set_value(c.C_CL_DIST_INTRA_MAX_Cl,   value=c_max,  dict_key=cl_name)
            self.cluster_stats.set_value(c.C_CL_DIST_INTRA_MIN_Cl,   value=c_min,  dict_key=cl_name)
            self.cluster_stats.set_value(c.C_CL_DIST_INTRA_MEAN_Cl,  value=c_mean, dict_key=cl_name)            
            same_samples_distance += cl_data.sum()
        
        return same_samples_distance

    def save_inter_cl_maxminmean(self, D: np.array, dist_mat_ids: np.array) -> float:
        # Sum up the distance between all samples, from different clusters. This value shall increase with the training.
        different_samples_distance:  float = 0.0

        for cl_tuple in self.pairwise_different_cl_ids:
            # Get the rows (=columns) of these clusters.
            cl0_ids = self.cluster_rows.get(cl_tuple[0])
            cl1_ids = self.cluster_rows.get(cl_tuple[1])
            # The matrix may be only partially computed. In this case, we need to map the cluster indices to the ones that 
            # actually occur in D.
            cl_a_ids_local = np.in1d(dist_mat_ids, cl0_ids, assume_unique=True).nonzero()[0]
            cl_b_ids_local = np.in1d(dist_mat_ids, cl1_ids, assume_unique=True).nonzero()[0]
            # Extract the distance matrix for all samples BETEWEEN these clusters.
            cl_ab_ids_local = np.ix_(cl_a_ids_local, cl_b_ids_local)
            cl_data = D[cl_ab_ids_local]
            # Note that this will already be an upper or lower triangle portion and not include the diagonal.
            # Store the statistics for this cluster.
            c_max, c_min, c_mean = round(cl_data.max(), c.N_DECIMALS), round(cl_data.min(), c.N_DECIMALS), round(cl_data.mean(), c.N_DECIMALS)
            self.cluster_stats.set_value(c.C_CL_DIST_INTER_MAX_Cls,   value=c_max,  dict_key=cl_tuple)
            self.cluster_stats.set_value(c.C_CL_DIST_INTER_MIN_Cls,   value=c_min,  dict_key=cl_tuple)
            self.cluster_stats.set_value(c.C_CL_DIST_INTER_MEAN_Cls,  value=c_mean,  dict_key=cl_tuple)            
            different_samples_distance += cl_data.sum()
        
        return different_samples_distance

    def save_sklearn_scores(self, D: np.array, dist_mat_ids: np.array) -> None:
        classes_fraction = self.classes_vec[dist_mat_ids]
        # SILHOUETTE COEFFICIENT: (b - a) / max(a, b)
        #       - using the mean intra-cluster distance (a) and 
        #       - the mean nearest-cluster distance (b) for each sample. (Mean of the distances between the sample and the nearest cluster that it is not a part of.)        
        # This function returns the mean Silhouette Coefficient over all samples. To obtain the values for each sample, use silhouette_samples.
        # The best value is 1 and the worst value is -1. 
        # Values near 0 indicate overlapping clusters. 
        # Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.
        silhouette_score = round(sklearn_silhouette_score(D, labels=classes_fraction, metric='precomputed'), c.N_DECIMALS)
        self.cluster_stats.set_value(c.C_SCORE_SILHOUETTE, silhouette_score)

        # DAVIES BOULDIN:
        # Average similarity measure of each cluster with its most similar cluster, 
        # where similarity is the ratio of within-cluster distances to between-cluster distances. 
        # Thus, clusters which are farther apart and less dispersed will result in a better (lower) score.
        # The optimal score is zero.
        db_score = round(sklearn_davies_bouldin_score(D, labels=classes_fraction), c.N_DECIMALS)
        self.cluster_stats.set_value(c.C_SCORE_DAVIES_BOULDIN, db_score)

        # CALINSKI AND HARABASZ [Variance Ratio Criterion]:
        # Ratio of the sum of between-cluster dispersion and of within-cluster dispersion.
        # The score is higher when clusters are dense and well separated.
        # Notice that score is generally higher for convex clusters than other concepts of clusters, 
        # such as density based clusters like those obtained through DBSCAN.
        ch_score = round(sklearn_calinski_harabasz_score(D, labels=classes_fraction), c.N_DECIMALS)
        self.cluster_stats.set_value(c.C_SCORE_CALINSKI_HARABASZ, ch_score)    

    def save_SVM_acc(self, distance_matrix: np.array, dist_mat_ids: np.array = None, nr_features: int = None, kernel_lambda: Any = 'auto', svm_n_iter: int = 10, save_kernel_mat_path: str = None, class_vec: np.array = None) -> Tuple[float,float]:
        # 'scale' and 'auto' lambda according to gamma settings in: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        if type(kernel_lambda) == str:
            if nr_features is None:
                print(f"Cannot use kernel lambda {kernel_lambda} since number of features is unknown! Using lambda of 1.0.")
                kernel_lambda = 1.0
            else:            
                if   kernel_lambda == 'scale': kernel_lambda = 1 / (nr_features * distance_matrix.var())
                elif kernel_lambda == 'auto':  kernel_lambda = 1 / nr_features

        # Compute the kernel matrix from the distance matrix.
        K = compute_tree_metric_kernel(distance_matrix, kernel_lambda)        
        if save_kernel_mat_path is not None: np.save(save_kernel_mat_path, K)        
        # Trimm the class vector to the fraction used in the computation of the distance matrix.
        if class_vec is None: class_vec = self.classes_vec.copy()
        classes_fraction = class_vec
        if dist_mat_ids is not None: classes_fraction = classes_fraction[dist_mat_ids]        
        # Evaluate the kernel matrix using the SVM. Therefore, scale it to [0, 1]. The default setting, using the exp(-lambda), already ensures this. 
        K = K / K.max() 
        accs = cross_validate_Kfold_SVM([K], classes_fraction, n_iter=svm_n_iter)

        # Prepare and save the results.
        avg_acc = round(np.mean(accs[0])*100, c.N_DECIMALS)
        std_dev = round(np.std(accs[0]) *100, c.N_DECIMALS)
        self.cluster_stats.set_value(c.C_SVM_ACC, avg_acc)
        self.cluster_stats.set_value(c.C_SVM_STD_DEV, std_dev)
        # Return the results.
        return avg_acc, std_dev, kernel_lambda

    def save_iteration(self, i: int) -> None:
        self.cluster_stats.set_value(c.C_LEARNER_EPOCH, i)

def compute_tree_metric_kernel(D: np.array, kernel_lambda: float = 1.0) -> np.mat:
    K = np.exp(-kernel_lambda * D)
    return K # np.mat(K)

def compute_dist_mat(representations_csr, w: np.array, fraction_step: int = None, print_used_method: bool = True, verbose: bool = False) -> Tuple[np.array, np.array, bool]:
        
    def recursive_pdist(R, w: np.array, D: np.array = None):
        """
        Compute one row of the distance matrix per iteration. 
        Therefore, separate the zeroth row from the representations matrix, 
        tile it vertically to match the shape of the representations matrix (the remaining rows) and 
        compute the pairwise distance to all remaining rows.            
        """
        if D is None: D = np.array([], dtype=float)

        # Drop the zeroth row of the matrix.
        row0 = R[0]
        # row0 = row0.toarray()[0]
        R = R[1:]
        
        row0_tiled = np.tile(row0, R.shape[0])
        row0_tiled.reshape((R.shape[0], R.shape[1]))

        dist_mat_row = np.abs(R - row0_tiled) @ w.T 
        D = np.append(D, dist_mat_row)
        
        if R.shape[0] > 1:
            return recursive_pdist(R, w, D)
        else: 
            return D

    def looped_pdist(R, w: np.array):
        """
        Compute one row of the distance matrix per iteration. 
        Therefore, separate the zeroth row from the representations matrix, 
        tile it vertically to match the shape of the representations matrix (the remaining rows) and 
        compute the pairwise distance to all remaining rows.            
        """
        n = R.shape[0]
        D: np.array = None
        n_total = n * (n-1) // 2
        n_done: int = 0
        percentage_str: str = ''
        print()

        for g_id1 in range(n-1):
            r_g1 = R[g_id1]            
            dist_mat_row: List[float] = []
                        
            start_time = time()
            for g_id2 in range(g_id1, n):
                print(f"\rpdist for {n_done + g_id2}\t/{n_total}\tgraphs{percentage_str}", end="", flush=True)
            
                r_g2 = R[g_id2]
                dist_g1g2 = np.abs(r_g1 - r_g2) @ w.T
                dist_mat_row.append(dist_g1g2)
            
            r_todo = n - g_id1
            n_done = n_total - (r_todo * (r_todo + 1) // 2)            
            pairs_todo = n_total - n_done
            pairs_done_in_time = n-g_id1
            expected_time: str = convert_s_to_h_m_s((time() - start_time) * pairs_todo / pairs_done_in_time)
            percentage_str = f" [{round(n_done / n_total, 2)}%]\t- Estimated RT: {expected_time}"
            
            D = np.append(D, np.array(dist_mat_row, dtype=np.float32))        
        return D
            
    nr_graphs = representations_csr.shape[0]

    dist_matrix_indices = np.array(range(0, nr_graphs), dtype=int)        
    is_fraction = fraction_step is not None
    if is_fraction:
        # Use every x-th graph to compute a fraction of the distance matrix. 
        fraction_indices    = np.arange(start=0, stop=nr_graphs, step=fraction_step, dtype=int)
        dist_matrix_indices = np.array(fraction_indices, dtype=int)            
    
    title: str = ''
    if verbose: title += "Distance matrix computation"
    if print_used_method:
        fraction_description: str = f"using {round(dist_matrix_indices.shape[0]/nr_graphs*100)}% of DistMat"
        print(f"{title} ({fraction_description})", end="")

    R_fraction = representations_csr[dist_matrix_indices]

    # Using pdist requires a dense matrix and thus the operation todense().
    # This may require to much disc space. 
    try:            
        if print_used_method: print(" <scipy.spatial.distance.pdist>", end="")
        dist_matrix = pdist(R_fraction.todense(), lambda u, v: np.sum(np.abs(u-v)@w))
    except:
        try:
            np.tile(R_fraction[0], R_fraction.shape[0])
            if print_used_method: print(" <recursive pdist>             ", end="")
            dist_matrix = recursive_pdist(R_fraction, w)
        except:
            # If so, compute the dist matrix by brute force (two nested for loops).
            # This may take a while, but requires much less space.
            if print_used_method: print(" <looped pdist>             ", end="")
            dist_matrix = looped_pdist(R_fraction, w)
            
    if verbose: print()
    return squareform(dist_matrix), dist_matrix_indices, is_fraction