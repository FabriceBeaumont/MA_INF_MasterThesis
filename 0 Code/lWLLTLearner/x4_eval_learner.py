from typing import Tuple, List, Any, Dict 
import numpy as np
import pandas as pd
from abc import ABC
from os.path import exists, dirname
from os import makedirs, listdir
from logging import ERROR as LOG_ERR
# For the parsing of console input:
import sys, getopt, argparse

# from scipy.sparse import coo_matrix, save_npz, load_npz, csr_matrix
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d # Required passivly for "plt.axes(projection ='3d')" and "plt.axes.plot_wireframe"
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Own files:
from my_utils.file_writer import CSVWriter

from x2_wllt_constructor import WLLT, get_WLLT_from_dataset_name
from x3_wllt_edge_weight_learner import HyperParams
from cluster_evaluator import ClusterEvaluator, compute_dist_mat
import my_consts as c

from my_utils.decorators import get_logger
LOG = get_logger(filename="Loggers/x4_eval_learner.txt", level=LOG_ERR, create_file=True)

class LearnerEvaluation(ABC):
                           
    def __init__(self, dir_learner: str, dir_out: str = None, dir_wllt: str = None, wllt: WLLT = None):
        
        def _get_w_vec_and_epochs_from_dir(dir_in: str) -> Tuple[np.array, np.array]:
            edge_weight_matrix: List[np.array] = []
            ew_epochs: List[int] = []
            dist_mats: Dict[int, Tuple[np.array, np.array, bool]] = dict()

            ew_file_predix: str     = f"{c.FN_EDGE_WEIGHTS_E}"
            d_mat_file_predix: str   = f"{c.FN_DIST_MAT_E}"
            d_ids_file_predix: str   = f"{c.FN_DIST_IDS_E}"
            
            epoch_to_ew_filename_dir:   Dict[int, str] = c.get_file_name_versions(dir_in=dir_in, file_predix=ew_file_predix)
            epoch_to_dmat_filename_dir: Dict[int, str] = c.get_file_name_versions(dir_in=dir_out, file_predix=d_mat_file_predix)
            epoch_to_dids_filename_dir: Dict[int, str] = c.get_file_name_versions(dir_in=dir_out, file_predix=d_ids_file_predix)
                        
            # Load and save the learned edge weights - in the correct order!
            ew_epochs = list(epoch_to_ew_filename_dir.keys())
            ew_epochs.sort()
            for e in ew_epochs: 
                edge_weight_matrix.append(np.load(epoch_to_ew_filename_dir.get(e)).astype(np.float32))

            # If the distance matrix exist, assume that the distance ids also exist, and uses the same epochs!
            if epoch_to_dmat_filename_dir is not None:
                d_epochs = list(epoch_to_ew_filename_dir.keys())
                d_epochs.sort()
                for e in d_epochs:
                    d       = np.load(epoch_to_dmat_filename_dir.get(e)).astype(np.float32)
                    d_ids   = np.load(epoch_to_dids_filename_dir.get(e)).astype(np.float32)
                    is_frac = d_ids[0]
                    d_ids   = d_ids[1:]
                    dist_mats[e] = (d, d_ids, is_frac)                    

            return np.array(edge_weight_matrix, dtype=np.float32), np.array(ew_epochs, dtype=np.int32), dist_mats
        
        def _define_output_dir(output_dir: str) -> str:
            if output_dir is None:
                in_dir = dirname(self.dir_in)
                in_filename = self.dir_in.replace(in_dir + '/', "")
                in_dir += '_eval/'
                output_dir = f"{in_dir}E_{in_filename}"
            # In case more than one evaluation does exist, add other folders with suffix '_v<ctr>'.
            if exists(output_dir):
                i = 0
                while(exists(f"{output_dir}_v{i}")):
                    i += 1
                output_dir = f"{output_dir}_v{i}"
            
            if not exists(output_dir):
                makedirs(output_dir)
            return output_dir                   

        if not exists(dir_learner): 
            print(f"WARNING: Leaner directory {dir_learner} not found!")
            return None

        self.dir_in:    str = dir_learner        
        self.dir_out:   str = _define_output_dir(dir_out)        
        
        W, E, D = _get_w_vec_and_epochs_from_dir(self.dir_in)
        self.weight_matrix:     np.array = W
        self.learner_epochs:    np.array = E
        self.nr_epochs:         int      = len(self.learner_epochs)
        self.max_epoch:         int      = self.learner_epochs[-1]

        self.dist_mats: Dict[int, Tuple[np.array, np.array, bool]] = D

        # Save the path so store the evaluation results.
        if wllt is None:
            wllt: WLLT = get_WLLT_from_dataset_name(dir_wllt)
        self.wllt = wllt # Delete if not necessary

        self.layer_starts: np.array = wllt.get_layer_starts_from_file()
        # Trim the layer starts to the size, upto which the weights were actually initialized.        
        self.layer_starts = self.layer_starts[np.where(self.layer_starts <= self.get_nr_features())]

        # Read in the graph representations, but only these columns, that are represented in the weight vectors.
        self.R_csr = self.wllt.get_graph_repr_from_file(nr_features=self.get_nr_features()).tocsr()
        assert self.R_csr.shape[1] == len(self.weight_matrix[0]), f"Dimension missmatch: R:{self.R_csr.shape} vs. w_0:{len(self.weight_matrix[0])}"

        self.cluster_class_vec:     np.array = wllt.get_graph_classes_vec_from_file()
        self.cluster_classes_set:   np.array = np.unique(self.cluster_class_vec)

        # Class instance to compute more elaborate evaluation values.
        self.cluster_eval: ClusterEvaluator = ClusterEvaluator(self.cluster_class_vec, self.cluster_classes_set)
        self.dist_mat_ids: List[int] = list()

        ### Set up a csv-file. ###
        _col_names = self._get_csv_col_names()
        self._csv_writer: CSVWriter = CSVWriter(_col_names, index_col_name=self.col_cl_epoch, nr_rows=self.nr_epochs)
        self._csv_path: str = f"{self.dir_out}/{c.CSV_LEARNER_EVAL}"
        # Write the learner epoch to the csv file.
        r_c_v_epochs = list(zip(range(self.nr_epochs), [self.col_cl_epoch]*self.nr_epochs, self.learner_epochs))
        self._csv_writer.add_entries(r_c_v_epochs)
    
    def get_nr_learner_epochs(self) -> int:
        return self.weight_matrix.shape[0]
    
    def get_nr_features(self) -> int:
        return self.weight_matrix.shape[1]

    def get_dataset_name(self) -> str:
        return self.wllt.dataset_name

    def get_dist_mat_of_epoch(self, epoch: int) -> np.array:
        D: np.array = None
        tpl: Tuple[np.array, np.array, bool] = self.dist_mats.get(epoch)
        if tpl is not None:
            D, _, _ = tpl
        return D

    def _get_csv_col_names(self) -> List[str]:
        # Columns for the edge weight evaluations:
        # Use two indices, iterating over the layer starts, to get the index interval of each layer.
        self.col_meanwpl_d, self.col_maxwpl_d, self.col_minwpl_d, self.col_sumwpl_d = [], [], [], []
        for layer_id, _ in enumerate(self.layer_starts):
            self.col_maxwpl_d.append(f"{c.C_MAX_WEIGHT_PER_LAYER}{layer_id}")
            self.col_minwpl_d.append(f"{c.C_MIN_WEIGHT_PER_LAYER}{layer_id}")
            self.col_meanwpl_d.append(f"{c.C_MEAN_WEIGHT_PER_LAYER}{layer_id}")
            self.col_sumwpl_d.append(f"{c.C_SUM_WEIGHT_PER_LAYER}{layer_id}")
        
        self.col_total_weight_sum   = c.C_TOTAL_WEIGHT_SUM
        self.col_cl_epoch           = c.C_LEARNER_EPOCH
        
        _col_names: List[str] = [self.col_cl_epoch, self.col_total_weight_sum] + self.col_meanwpl_d + self.col_maxwpl_d + self.col_minwpl_d + self.col_sumwpl_d

        # Columns for the cluster statistics:
        cluster_stat_cols = self.cluster_eval.get_all_names()        

        _col_names += cluster_stat_cols

        return _col_names

    def _get_list_of_epochs_to_eval(self, epoch_limit: int = None, epochs: List[int] = None) -> np.array:
        if epoch_limit is not None and epoch_limit <= self.max_epoch:
            epochs = self.learner_epochs[np.where(self.learner_epochs <= epoch_limit)]
        elif epochs is not None:
            epochs.sort()            
            # If all entries are fractions, convert them to integers.
            if all(e <= 1.0 for e in epochs): 
                epochs = np.array([self.learner_epochs[int(percent * (self.nr_epochs - 1))] for percent in epochs], dtype=np.int32)
            else:
                # If the epochs are not fractions, make sure that all these are known epochs.
                is_subset = all(x in self.learner_epochs for x in epochs)
                if not is_subset:
                    LOG.error(f"Only {self.max_epoch} epochs are avaliable. Cannot evaluate epochs: {epochs}")
                    return None
        else:
            epochs = self.learner_epochs.copy()
        
        return np.array(list(epochs), dtype=int)
    
    def _get_row_ids_in_list(self, value_list, target_values: List[int]) -> List[int]:
        row_ids = [np.where(value_list == i)[0][0] for i in target_values]
        return row_ids    

    def compute_and_save_or_load_dist_mat(self, d_epochs: List[int]) -> None:        
        iterations = self._get_row_ids_in_list(self.learner_epochs, d_epochs)        
        print()
        for i_ptr, e in enumerate(d_epochs):
            i = iterations[i_ptr]
            w = self.weight_matrix[i]
            if self.dist_mats.get(e) is None:
                print(f"\rComputing distance matrix for epoch {e} ...", end="")
                dist_mat, dist_mat_ids, is_fraction = compute_dist_mat(self.R_csr, w)
                np.save(f"{self.dir_out}/{c.FN_DIST_MAT_E}{e}", dist_mat, allow_pickle=True)
                np.save(f"{self.dir_out}/{c.FN_DIST_IDS_E}{e}", np.append(np.array([is_fraction]), dist_mat_ids), allow_pickle=True)
                self.dist_mats[e] = (dist_mat, dist_mat_ids, is_fraction)
            else:
                print(f"\rDistance matrix for epoch {e} was loaded.", end="")                

    def write_cluster_eval_to_csv(self) -> None:        
        self._csv_writer.write_to_csv(self._csv_path)

    ### Evaluation methods ###

    def evaluate_learner_results(self, eval_svm: bool = True, eval_cl: bool = True, epoch_limit: int = None, w_epochs: List[int] = None, d_epochs: List[int] = [0, .5, 1.]):
        
        w_epochs: np.array = self._get_list_of_epochs_to_eval(epoch_limit, w_epochs)        
        d_epochs: np.array = self._get_list_of_epochs_to_eval(epoch_limit, d_epochs)
        
        self.compute_and_save_or_load_dist_mat(d_epochs)

        # Compute the evaluations.
        self.evaluate_edge_weight_stats_tocsv(epochs=w_epochs)
        self.evaluate_cluster_stats_tocsv(d_epochs=d_epochs, eval_cl=eval_cl, eval_svm=eval_svm)

        # Save the evaluations to file.
        self.write_cluster_eval_to_csv()
        self.write_learner_eval(dataset_name=self.get_dataset_name(), more_columns=[('DataStorage', self.dir_in)])        
        
        self.write_min_max_paths_to_file()
        self.plot_wllt_structure(epochs=d_epochs)
        self.plot_edge_weight_stats(epochs=w_epochs)
        self.plot_cluster_and_svm_stats(d_epochs=d_epochs, eval_cl=eval_cl, eval_svm=eval_svm)
        self.plot_dist_mats(d_epochs)
        print()

    def evaluate_edge_weight_stats_tocsv(self, epochs: np.array) -> None:
        """
        Compute and save the edge weight statistics to csv and plot them.
        The statistics inclue:
        - Total weight sum per epoch
        - Max, min and mean weight per layer - per epoch
        - Weight sum per layer - per epoch
        """
        # Evaluate the data for every epoch.
        for i, e in enumerate(epochs):

            w = self.weight_matrix[i]            

            # Save the sum of all used weight among all edges.
            total_weight_sum = np.sum(w)
            self._csv_writer.add_entry(i, self.col_total_weight_sum, total_weight_sum)
            
            first_layer_start = 0
            # Use two indices, iterating over the layer starts, to get the index interval of each layer.
            for layer_id, next_layer_start in enumerate(self.layer_starts):
                # Compute the mean weight for layer 'layer_id'.
                w_mean = w[first_layer_start:next_layer_start].mean()
                w_max  = w[first_layer_start:next_layer_start].max()
                w_min  = w[first_layer_start:next_layer_start].min()
                w_sum  = w[first_layer_start:next_layer_start].sum()
                # Get the name of the respective column.
                c_mean_name = self.col_meanwpl_d[layer_id]
                c_max_name  = self.col_maxwpl_d[layer_id]
                c_min_name  = self.col_minwpl_d[layer_id]
                c_sum_name  = self.col_sumwpl_d[layer_id]
                # Store the values in their respective columns.
                self._csv_writer.add_entry(i, c_mean_name, w_mean)
                self._csv_writer.add_entry(i, c_max_name, w_max)
                self._csv_writer.add_entry(i, c_min_name, w_min)
                self._csv_writer.add_entry(i, c_sum_name, w_sum)                
                
                # Set the next index after the last layer start, as the next layer start.
                first_layer_start = next_layer_start + 1 
        
    def evaluate_cluster_stats_tocsv(self, d_epochs: np.array, eval_cl: bool = True, eval_svm: bool = False) -> None:
        """
        Compute and save the cluster statistics to csv and plot them.
        The statistics inclue:
        - Max, min, mean intER cluster distance
        - Max, min, mean intRA cluster distance
        """        
        ### Compute the distance matrices and evaluate it. ###
        n = len(d_epochs)
        iterations = self._get_row_ids_in_list(self.learner_epochs, d_epochs)
        inter_intra_sme_base: Tuple[float, float] = None

        for i_ptr, e in enumerate(d_epochs):
            i = iterations[i_ptr]

            print_str = f"Computing {'Cluster IntER/IntRA-Stats. ' if eval_cl else ''}{'& ' if eval_svm and eval_cl else ''}{'SVM Acc.' if eval_svm else ''}"
            print(f"\r>\t{print_str} - {round((i_ptr + 1)*100/n)}%\t({i_ptr + 1}\t/{n} epochs)" + " "*5, end="")            

            dist_mat, self.dist_mat_ids, is_fraction = self.dist_mats.get(e)

            # Compute and save the sklearn scores.
            if eval_cl:
                inter_cl_dist_sum = self.cluster_eval.save_inter_cl_maxminmean(dist_mat, self.dist_mat_ids)
                intra_cl_dist_sum = self.cluster_eval.save_intra_cl_maxminmean(dist_mat, self.dist_mat_ids)
                # In the first iteration, normalize the inter and intra sme.
                if inter_intra_sme_base is None:
                    inter_intra_sme_base = inter_cl_dist_sum, intra_cl_dist_sum
                
                inter_cl_dist_sum = inter_cl_dist_sum / inter_intra_sme_base[0]
                intra_cl_dist_sum = intra_cl_dist_sum / inter_intra_sme_base[1]
                self.cluster_eval.cluster_stats.set_value(c.C_CL_DIST_INTER_SUM,        value=inter_cl_dist_sum)
                self.cluster_eval.cluster_stats.set_value(c.C_CL_DIST_INTRA_SUM,        value=intra_cl_dist_sum)
                self.cluster_eval.cluster_stats.set_value(c.C_CL_SAMPLE_MOVEMENT_ERROR, value=intra_cl_dist_sum - inter_cl_dist_sum)

                self.cluster_eval.save_sklearn_scores(dist_mat, self.dist_mat_ids)
            # Compute and save the SVM accuracy.
            if eval_svm:
                self.cluster_eval.save_SVM_acc(distance_matrix=dist_mat, dist_mat_ids=self.dist_mat_ids, nr_features=self.get_nr_features())            
                        
            # Save the restuls of the cluster evaluator to the csv table.
            name_values_list = self.cluster_eval.get_name_value_list()            
            for name, value in name_values_list:
                new_entries = [(i, name, value)]
                self._csv_writer.add_entries(new_entries)

            self.cluster_eval.reset_cluster_stats()
        print("\r")

    def write_min_max_paths_to_file(self, w: np.array = None) -> None:
        wllt_paths: Dict[float, np.array]               = self.wllt.get_all_paths_to_root(max_v_id=self.get_nr_features())
        vertex_map: Dict[int, Tuple[int, List[int]]]    = self.wllt.get_vertex_label_map_from_file(as_dict=True)
        if w is None: w: np.array[float] = self.weight_matrix[-1]

        min_path_dist: float        = np.inf
        max_path_dist: float        = 0.0
        min_path_leaves: List[int]  = []
        max_path_leaves: List[int]  = []
        # Find two leaves, representing the maximum and minimum path lenghts.
        for leaf, path in wllt_paths.items():
            
            path_dist: float = np.sum(w[path])

            # Find leaves with maximal path length.
            if path_dist > max_path_dist:
                max_path_leaves = [leaf]
                max_path_dist   = path_dist
            elif path_dist == max_path_dist:
                max_path_leaves += [leaf]
            # Find leaves with minimal path length.
            if path_dist < min_path_dist:
                min_path_leaves = [leaf]
                min_path_dist   = path_dist
            elif path_dist == min_path_dist:
                min_path_leaves += [leaf]

        ret_tuple = [("Minimal", min_path_leaves, min_path_dist), ("Maximal", max_path_leaves, max_path_dist)]

        _print_str: str = ''
        for title, leaves, path_dist in ret_tuple:
            example_leaf = leaves[0]
            _print_str  += f"{title} path length: {path_dist}"
            _print_str  += f"\nExample path:      P[{example_leaf}, root] ={wllt_paths.get(example_leaf)}"
            _print_str  += f"\nAll '{len(leaves)}' paths with this length:\n{leaves}"
            _, unfolding_tree_list = self.wllt.get_unfolding_tree(example_leaf, vertex_map)
            _print_str  += f"\nUnfolding tree:\n{unfolding_tree_list}"
            _print_str  += "\n\n\n"
                
        with open(f"{self.dir_out}/MinMaxUnfoldingExamples.txt", 'w') as f:
            f.write(_print_str)

        return None

    def plot_cluster_and_svm_stats(self, d_epochs: np.array, eval_cl: bool = True, eval_svm: bool = False) -> None:
        
        csv_df = self._csv_writer.convert_to_pandas_df()

        cols_other  = csv_df.columns
        nr_rows     = csv_df.shape[0]
                    
        xticks: List[str] = [str(e) for e in d_epochs]
        # The 'd_epochs' may NOT allign with the number of rows in the csv tabular.
        # Neigher may their values allign. Get the actual row indices by comparing with the iteration column:
        row_ids = self._get_row_ids_in_list(csv_df[c.C_LEARNER_EPOCH], d_epochs)
        # Delete the iteration column since it will no longer be needed.
        cols_iteration_name, cols_other = self.separate_columns_with_prefix(csv_df[cols_other], c.C_LEARNER_EPOCH)
                            
        # Plot cluster statistics.
        if eval_cl:
            print(f"\r>\tPlotting cluster statistics..." + " "*30, end="")
            # Inter cluster distance statistics.
            cols_inter, cols_other = self.separate_columns_with_prefix(csv_df[cols_other], c.C_CL_DIST_INTER_MAX_Cls)
            self.save_in_one_plot(csv_df, cols_inter, xticks, title = "Max inter cluster distance (May Increase)",  image_name = "plot_InterMaxClDist", rows=row_ids)
            cols_inter, cols_other = self.separate_columns_with_prefix(csv_df[cols_other], c.C_CL_DIST_INTER_MIN_Cls)
            self.save_in_one_plot(csv_df, cols_inter, xticks, title = "Min inter cluster distance (Increase)",      image_name = "plot_InterMinClDist", rows=row_ids)
            cols_inter, cols_other = self.separate_columns_with_prefix(csv_df[cols_other], c.C_CL_DIST_INTER_MEAN_Cls)
            self.save_in_one_plot(csv_df, cols_inter, xticks, title = "Mean inter cluster distance (May Increase)", image_name = "plot_InterMeanClDist", rows=row_ids)
            # Intra cluster distance statistics.
            cols_intra, cols_other = self.separate_columns_with_prefix(csv_df[cols_other], c.C_CL_DIST_INTRA_MAX_Cl)
            self.save_in_one_plot(csv_df, cols_intra, xticks, title = "Max intra cluster distance (Decrease)",      image_name = "plot_IntraMaxClDist", rows=row_ids)
            cols_intra, cols_other = self.separate_columns_with_prefix(csv_df[cols_other], c.C_CL_DIST_INTRA_MIN_Cl)
            self.save_in_one_plot(csv_df, cols_intra, xticks, title = "Min intra cluster distance (May Decrease)",  image_name = "plot_IntraMinClDist", rows=row_ids)
            cols_intra, cols_other = self.separate_columns_with_prefix(csv_df[cols_other], c.C_CL_DIST_INTRA_MEAN_Cl)
            self.save_in_one_plot(csv_df, cols_intra, xticks, title = "Mean intra cluster distance (Decrease)",     image_name = "plot_IntraMeanClDist", rows=row_ids)
            # Sample movement error.
            col_decrease, cols_other = self.separate_columns_with_prefix(csv_df[cols_other], c.C_CL_DIST_INTRA_SUM)
            col_increase, cols_other = self.separate_columns_with_prefix(csv_df[cols_other], c.C_CL_DIST_INTER_SUM)
            col_line,     cols_other = self.separate_columns_with_prefix(csv_df[cols_other], c.C_CL_SAMPLE_MOVEMENT_ERROR)            
            self.save_in_stacked_bar_plot(csv_df, col_decrease, col_increase, col_line, xticks, title = "Class difference sums", image_name = "plot_ClDiffSums", rows=row_ids)
        # Plot the SVM scores.        
        cols_svm_acc, cols_other    = self.separate_columns_with_prefix(csv_df[cols_other], c.C_SVM_ACC)
        cols_svm_stddev, cols_other = self.separate_columns_with_prefix(csv_df[cols_other], c.C_SVM_STD_DEV)
        col_svm_acc, col_svm_stddev = cols_svm_acc[0], cols_svm_stddev[0]
        if eval_svm:
            print(f"\r>\tPlotting svm statistics..." + " "*30, end="")
            self.save_in_one_plot_diff_axes(csv_df, col_svm_acc, col_svm_stddev, xticks, title = "SVM Evaluation",  image_name = "plot_SVM", rows=row_ids)
        # Plot the cluster scores, each in a single plot.
        self.save_in_separate_plots(csv_df, csv_df[cols_other], xticks, rows=row_ids)

    def plot_dist_mats(self, d_epochs: List[int]) -> None:
        iterations = self._get_row_ids_in_list(self.learner_epochs, d_epochs)      
        for i_ptr, e in enumerate(d_epochs):
            
            i = iterations[i_ptr]
            dist_mat, self.dist_mat_ids, is_fraction = self.dist_mats.get(e)
            
            # Plot the distance matrices with T-SNE.
            storage_path: str = f"{self.dir_out}/tSNE_e{e}{'_fraction' if is_fraction else ''}_i{i_ptr}"
            self.plot_distance_matrices_with_tsne(d_mat=dist_mat, mat_ids=self.dist_mat_ids, iter=e, is_fraction=is_fraction, storage_path=storage_path)

            # Plot the distance matrices as color gradient.
            storage_path: str = f"{self.dir_out}/plot_{c.FN_DIST_MAT_E}_e{e}{'_fraction' if is_fraction else ''}_i{i_ptr}"
            d_mat, d_ids = dist_mat, self.dist_mat_ids
            display_n = 100
            if len(self.dist_mat_ids) > display_n:
                d_mat, d_ids    = d_mat[0:display_n, 0:display_n], d_ids[:display_n]
                is_fraction     = True
            self.plot_distance_matrices_as_color_map(d_mat=d_mat, mat_ids=d_ids, iter=e, is_fraction=is_fraction, storage_path=storage_path)           
            
    def write_learner_eval(self, dataset_name: str, csv_file_path=c.CSV_MA_EVALUATION_PATH, more_columns: Tuple[str, Any] = None) -> None:
        
        def concatenate_vertically(matrix: np.array) -> List[str]:
            concat_list = []
            # Use a while loop and check if the index is valid. 
            # In case not all rows have the same lenght or invalid entries.
            n_col = len(matrix[0])
            if n_col <= 1: return [str(v) for v in matrix]

            for c_ptr in range(n_col):
                col_concat = ''        
                for row in matrix:
                    # Checking for valid index / column
                    try:
                        if len(col_concat) > 0: 
                            col_concat = f"{col_concat}, {row[c_ptr]}"
                        else:
                            col_concat = f"{row[c_ptr]}"
                    except IndexError: pass
                concat_list.append(col_concat)
                c_ptr = c_ptr + 1
            
            concat_list = [e for e in concat_list if e]
            return concat_list

        # Get the cluster stats columns:
        # Identify the rows, where the cluster stats are stored.
        iteration_col = self._csv_writer.get_index_column()        
        dist_mat_rows = np.greater_equal(iteration_col, 0)

        ### Assemble all data ###
        # First, write the name of the database and indicate the success of the run.
        row_init: int = np.nonzero(dist_mat_rows)[0][0]
        sucess_measures = [
            self._csv_writer.get_column(c.C_CL_SAMPLE_MOVEMENT_ERROR)[row_init] > np.min(self._csv_writer.get_column(c.C_CL_SAMPLE_MOVEMENT_ERROR)),    # Goal: Decrease the Sample Movement Error
            self._csv_writer.get_column(c.C_SCORE_CALINSKI_HARABASZ)[row_init]  < np.max(self._csv_writer.get_column(c.C_SCORE_CALINSKI_HARABASZ)),     # Goal: Increase the Calinski Harabasz score
            self._csv_writer.get_column(c.C_SCORE_DAVIES_BOULDIN)[row_init]     > np.min(self._csv_writer.get_column(c.C_SCORE_DAVIES_BOULDIN)),        # Goal: Decrease the Davies Bouldin score
            self._csv_writer.get_column(c.C_SCORE_SILHOUETTE)[row_init]         < np.max(self._csv_writer.get_column(c.C_SCORE_SILHOUETTE)),            # Goal: Increase the Silhouette score
            self._csv_writer.get_column(c.C_SVM_ACC)[row_init]                  < np.max(self._csv_writer.get_column(c.C_SVM_ACC))                      # Goal: Increase the SVM acc.
        ]
        success_col_names   = ['Improved SME', 'Improved Calinski Harabasz', 'Improved Silhouette', 'Improved Davies Bouldin', 'Improved SVM'] 
        for s in sucess_measures:
            success_col_values  = [{"y" if s else "n"} for s in sucess_measures]

        # Get the settings of the edge weight learner config file:
        hp = HyperParams()        
        settings_names, settings_values = hp.read_config_csv_file(path=f"{self.dir_in}/{c.CSV_EW_LEARNER_CONFIG}")
        settings_col_names  = [s.replace(" ", "") for s in settings_names]
        settings_col_values = concatenate_vertically(settings_values)

        # Get the additional columns:
        other_cols_names    = [x[0] for x in more_columns]
        other_cols_values   = [x[1] for x in more_columns]

        # Group the column names in preference for minimum or maximum values.
        keyvalue_max_col_names = [
            c.C_SVM_ACC,
            c.C_SCORE_CALINSKI_HARABASZ, 
            c.C_SCORE_SILHOUETTE, 
        ]
        keyvalue_min_col_names = [
            c.C_CL_SAMPLE_MOVEMENT_ERROR,
            c.C_SCORE_DAVIES_BOULDIN
        ]
        
        # Evaluate the grouped columns.
        keyvalue_max_values = []
        for col_name in keyvalue_max_col_names:
            col = self._csv_writer.get_column(col_name)[dist_mat_rows]
            iteration = np.argmax(col)
            value = col[iteration]
            keyvalue_max_values += [value, iteration]
        
        keyvalue_min_values = []
        for col_name in keyvalue_min_col_names:
            col = self._csv_writer.get_column(col_name)[dist_mat_rows]
            iteration = np.argmin(col)
            value = col[iteration]
            keyvalue_min_values += [value, iteration]

        # Initialize the column names as 'iter' - which will denote the iteration of the obtained value of the property in the previous column.
        keyvalue_names  = keyvalue_max_col_names + keyvalue_min_col_names
        keyvalue_col_names  = ["iter"] * len(keyvalue_names) * 2        
        # Now insert in every second entry the actual property names.
        for index, value in enumerate(keyvalue_names):
            keyvalue_col_names[index*2] = value
        # Initialize the column values.
        keyvalue_col_values = keyvalue_max_values + keyvalue_min_values

        ### Write/Append to csv row to file. ###
        # Prepare the header.
        header_row: str = ''        
        if not exists(csv_file_path): 
            header_row  = c.CSV_DELIMITER.join(["Dataset"] + success_col_names + settings_col_names + keyvalue_col_names + other_cols_names)
        
        # Prepare the data.
        value_row_str = [dataset_name] + [str(x) for x in success_col_values + settings_col_values + keyvalue_col_values + other_cols_values]
        value_row_arr = np.array(value_row_str).reshape(1, len(value_row_str))
        
        # Write the data and header to the csv file.        
        with open(csv_file_path, 'a') as f:
            np.savetxt(f, value_row_arr, delimiter=c.CSV_DELIMITER, header=header_row, fmt='%s')

    ### Plotting functions ####

    def parse_title_str(self, title: str) -> str:
        return title.replace(r'\N', '\n')

    def plot_wllt_structure(self, epochs: np.array) -> None:        
        # Plot the WLLT - using the edge weights as springs.
        print("\r\t\t- Plotting WLLTs..." + " "*30, end="")        
        row_ids = self._get_row_ids_in_list(self.learner_epochs, epochs)
        for e in row_ids:
            w = self.weight_matrix[e]
            self.wllt.save_wllt_to_png(dir_out=self.dir_out, edge_weights=w, title_postfix=f"_e{e}")
    
    def separate_columns_with_prefix(self, df, prefix: str) -> Tuple[List[str], List[str]]:
        selected_columns = [x for x in df.columns if x.startswith(prefix)]
        other_columns = list(set(df.columns).difference(selected_columns))            
        return selected_columns, other_columns

    def save_in_one_plot(self, df, cols: List[str], xticks_labels: List[str], title: str, image_name: str, rows: List[int] = None) -> None:
        title = self.parse_title_str(title)
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111, label=title)
        ax.set_title(title)
        # Do not allow for custom y-axis formatting with scientific notations.
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        ax.set_xlabel("Iterations"); ax.set_xticks(range(len(xticks_labels)))
        ax.set_xticklabels(xticks_labels) #, rotation=45)
        cols.sort()
        for col in cols:
            if rows is None:
                ax.plot(xticks_labels, df[col])
            else:
                ax.plot(xticks_labels, df[col][rows])
        
        ax.legend()
        fig.savefig(f"{self.dir_out}/{image_name}")
        fig.clf(); plt.cla(); plt.clf(); plt.close(fig); plt.close("all")
    
    def save_in_one_plot_diff_axes(self, df, col1, col2, xticks_labels: List[str], title: str, image_name: str, rows: List[int]) -> None:
        title = self.parse_title_str(title)
        # Create Plot:
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Iterations"); ax1.set_xticks(range(len(xticks_labels)))
        ax1.set_xticklabels(xticks_labels) #, rotation=45)
        
        # Axis 1:
        data_ax1, lbl_ax1 = df[col1][rows], col1
        col_ax1 = 'blue'
        ax1.set_ylabel(lbl_ax1, color = col_ax1) 
        ax1.tick_params(axis ='y', labelcolor = col_ax1) 
        ax1.plot(xticks_labels, data_ax1, color = col_ax1) 
        ax1.legend(loc='upper left')

        # Adding Twin Axes:
        ax2 = ax1.twinx()
        # Axis 2:
        data_ax2, lbl_ax2 = df[col2][rows], col2
        col_ax2 = 'black'
        ax2.set_ylabel(lbl_ax2, color = col_ax2)
        ax2.tick_params(axis ='y', labelcolor = col_ax2) 
        ax2.plot(xticks_labels, data_ax2, color = col_ax2) 
        ax2.legend(loc='upper right')

        # Save the plot:
        ax1.set_title(title)
        fig.savefig(f"{self.dir_out}/{image_name}")
        fig.clf()

    def save_in_separate_plots(self, df, cols: List[str], xticks_labels: List[str], rows: List[int] = None) -> None:            
        """
        Creata and save a separate plot for every column of values in the 'clos' list.
        """
        for col in cols:
            col_name = col.replace(" ", "_")
            fig = Figure()
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111, label=col)
            ax.set_title(self.parse_title_str(col))
            # Do not allow for custom y-axis formatting with scientific notations.
            ax.get_yaxis().get_major_formatter().set_useOffset(False)                
            if rows is None:
                ax.plot(xticks_labels, df[col])
            else:
                ax.plot(xticks_labels, df[col][rows])
            ax.set_xlabel("Iterations"); ax.set_xticks(range(len(xticks_labels)))
            ax.set_xticklabels(xticks_labels) #, rotation=45)
            fig.savefig(f"{self.dir_out}/plot_{col_name}")
            fig.clf(); plt.cla(); plt.clf(); plt.close(fig); plt.close("all")
    
    def save_in_stacked_bar_plot(self, df, col_decrease, col_increase, col_line, xticks_labels: List[str], title: str, image_name: str, rows: List[int]) -> None:
        # Prepare the data.
        title = self.parse_title_str(title)
        df = df.iloc[rows]
        _bar_width = 0.5
        data_ax1, lbl_bar1  =         np.array([x[0] for x in df[col_decrease].values.tolist()]), 'Different'   # 'IntRA cluster distance'
        data_ax2, lbl_bar2  =  (-1) * np.array([x[0] for x in df[col_increase].values.tolist()]), 'Same'        # 'IntER cluster distance'
        error,    lbl_line1 =         np.array([x[0] for x in df[col_line].values.tolist()]),     'SME'         # 'IntRA - IntER'
        x_ticks = np.arange(len(data_ax1))

        # Create Plot.        
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111, label=title)
        ax.set_title(title)
        # Plot the stacked bars.
        ax.bar(x_ticks, data_ax1, width=_bar_width, label=lbl_bar1,  color='green')
        ax.bar(x_ticks, data_ax2, width=_bar_width, label=lbl_bar2,  color='orange')
        ax.plot(x_ticks,   error,                   label=lbl_line1, color='red')

        ax.set_xlabel("Epochs"); ax.set_xticks(range(len(xticks_labels)))
        ax.set_xticklabels(xticks_labels) #, rotation=45)
        ax.set_ylabel("Distance sum")
        ax.legend()

        fig.savefig(f"{self.dir_out}/{image_name}")
        fig.clf(); plt.cla(); plt.clf(); plt.close(fig); plt.close("all")

    def plot_edge_weight_stats(self, epochs: np.array) -> None:        
        csv_df = self._csv_writer.convert_to_pandas_df()

        cols_other  = csv_df.columns
        nr_rows     = csv_df.shape[0]
                    
        cols_iteration_name, cols_other = self.separate_columns_with_prefix(csv_df[cols_other], c.C_LEARNER_EPOCH)
        xticks: List[str] = [str(e) for e in epochs]

        # Plot edge weight stats - per layer.
        print(f"\r>\tPlotting edge weights statistics..." + " "*30, end="")
        # Some groups of columns shall be displayed in the same plot. Separate the columns for each separate plot and plot them.
        cols_meanwpl, cols_other = self.separate_columns_with_prefix(csv_df[cols_other], c.C_MEAN_WEIGHT_PER_LAYER)
        self.save_in_one_plot(csv_df, cols_meanwpl, xticks, title = "Mean weights per Layer", image_name = "plot_MeanWpL")
        cols_maxwpl, cols_other = self.separate_columns_with_prefix(csv_df[cols_other], c.C_MAX_WEIGHT_PER_LAYER)
        self.save_in_one_plot(csv_df, cols_maxwpl, xticks, title = "Max weights per layer", image_name = "plot_MaxWpL")
        cols_minwpl, cols_other = self.separate_columns_with_prefix(csv_df[cols_other], c.C_MIN_WEIGHT_PER_LAYER)
        self.save_in_one_plot(csv_df, cols_minwpl, xticks, title = "Min weights per layer", image_name = "plot_MinWpL")
        
        cols_sumwpl, cols_other = self.separate_columns_with_prefix(csv_df[cols_other], c.C_SUM_WEIGHT_PER_LAYER)            
        self.save_in_separate_plots(csv_df, csv_df[cols_sumwpl], xticks)
        
        # Plot the edge weight stats - over all layers.
        cols_tws, cols_other = self.separate_columns_with_prefix(csv_df[cols_other], c.C_TOTAL_WEIGHT_SUM)            
        self.save_in_one_plot(csv_df, cols_tws, xticks, title = "Total weight sum", image_name = "plot_TotalWeightSum")

        # Plot the development of the weights per layer as 3d landscape.
        print(f"\r>\tPlotting edge weights 3d landscapes..." + " "*30, end="")
        self.plot_layer_edge_weight_landscapes(self.weight_matrix, iter_start=self.learner_epochs[0])
        print("\r", end="")

    def visualize_with_tsne(self, dist_mat: np.array, classes: np.array, tsne_metric: str = 'precomputed', tsne_iter: int = 1000, title: str = 'Scatter plot using t-SNE', storage_path: str = "scatter_plot_tsne") -> pd.DataFrame:
        """
        Given a square distance matrix 'dist_mat' and a vector of classes 'classes', this function will return a pandas DataFrame
        and a scatter plot where the datapoints from the matrix are plotted using t-SNE.
        
        Notice that t-SNE will not map two dots with zero distance to the same spot!
        """
        # Instantiate t-SNE.
        tsne = TSNE(random_state = 0, n_iter = tsne_iter, metric = tsne_metric, square_distances=True,) # learning_rate='auto')
        # Fit and transform.
        embedding_2d = tsne.fit_transform(dist_mat)
        # Store the embedding in a pandas DataFrame.        
        embedding_df = pd.DataFrame()
        embedding_df['comp1'] = embedding_2d[:,0]
        embedding_df['comp2'] = embedding_2d[:,1]
        embedding_df['y'] = classes
        
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111, label=title)        
        # Scatter plot the embedding.        
        ax.scatter(embedding_df.comp1, embedding_df.comp2, alpha=1., c=classes)
        
        ax.set_title(title)
        ax.set_xticks([]); ax.set_xlabel("")
        ax.set_yticks([]); ax.set_ylabel("")
        fig.savefig(storage_path)        
        fig.clf(); plt.cla(); plt.clf(); plt.close(fig); plt.close("all")
        return embedding_2d

    def plot_distance_matrices_with_tsne(self, d_mat: np.array, mat_ids: np.array, iter: int, is_fraction: bool, storage_path: str = None) -> None:
        classes = self.cluster_class_vec.copy()
        fraction_description: str   = f"{round(mat_ids.shape[0]/classes.shape[0]*100)}%-fraction"
        plot_title: str             = f"t-SNE Embedding - {fraction_description if is_fraction else ''} Epoch: {iter}"        
        
        if is_fraction: classes = classes[mat_ids]
        self.visualize_with_tsne(dist_mat=d_mat, classes=classes, title=plot_title, storage_path=storage_path)
    
    def plot_distance_matrices_as_color_map(self, d_mat: np.array, mat_ids: np.array, iter: int, is_fraction: bool, storage_path: str = None) -> None:
        classes = self.cluster_class_vec.copy()
        fraction_description: str   = f"{round(mat_ids.shape[0]/classes.shape[0]*100)}%-fraction"
        plot_title: str             = f"Distance matrix - {fraction_description if is_fraction else ''} Epoch: {iter}"
                
        if is_fraction: classes = classes[mat_ids]
        
        m, n = np.shape(d_mat)
        x_ticks = mat_ids[np.arange(0, m, m//10, dtype=np.int32)]        

        # fig = Figure(); canvas = FigureCanvas(fig)        
        fig, ax = plt.subplots(2, 3, gridspec_kw={'width_ratios': [2, 10, 1], 'height_ratios': [10, 2]}) #, figsize=(100,100))
        bottom = 0.1; top=1.-bottom; 
        wspace  = 0.02
        hspace  = 0.02
        plt.subplots_adjust(top=top, bottom=bottom, wspace=wspace, hspace=hspace)

        # 0,0 - Graph classes vertical.
        ax[0][0].imshow(np.expand_dims(classes, axis=1), alpha=0.8, cmap='inferno', aspect='auto')
        ax[0][0].set_xticks([])
        ax[0][0].set_ylabel("Graph classes"); ax[0][0].set_yticks(x_ticks)

        # Distance matrix.
        p = ax[0][1].imshow(d_mat, alpha=0.8, cmap='inferno', interpolation='nearest', aspect='auto')
        ax[0][1].set_title(plot_title)    
        ax[0][1].set_xticks([])
        ax[0][1].set_yticks([])
        ax[0][1].set_axis_off()        
        
        # 0,0 - Empty ax. 
        ax[0][2].set_xticks([]); ax[0][2].set_yticks([])
        ax[0][2].set_visible(False)

        # 1,0 - Empty ax. 
        ax[1][0].set_xticks([]); ax[1][0].set_yticks([])
        ax[1][0].set_visible(False)
        
        # 1,1 - Graph classes horizontal.
        ax[1][1].imshow(np.expand_dims(classes, axis=0), alpha=0.8, cmap='inferno', aspect='auto')
        ax[1][1].set_yticks([])
        ax[1][1].set_xlabel("Graph classes"); ax[1][1].set_xticks(x_ticks)

        # 1,2 - Empty ax. 
        ax[1][2].set_xticks([]); ax[1][2].set_yticks([])
        ax[1][2].set_visible(False)
        
        # Plot a colorbar for the dist matrix values.
        divider = make_axes_locatable(ax[0][2])
        colorbar_axes = divider.append_axes("left", size="50%", pad=0.01)
        fig.colorbar(p, ax=ax.ravel().tolist(), cax=colorbar_axes)

        fig.savefig(storage_path)
        fig.clf(); plt.cla(); plt.clf(); plt.close(fig); plt.close("all")

    def save_vectors_wireframe_plot(self, data: np.array, layer_nr: int, z_limits: Tuple[float, float], layers_first_lbl: int, layers_last_lbl: int, first_learner_iter: int, path: str, z_ax_log: bool = False) -> None:
        # If attempting to plot the z-axis in log-scale, but the data is not really big enough, do not plot.
        if z_ax_log and data.max()-data.min() < 100:
            return
        # Number of x-data points (nr of WL labels used in this layer).
        nr_wl_labels = data.shape[1]
        # Number of y-data points (nr of weight vectors).
        nr_weight_vecs = data.shape[0]

        # Construct the mesh grid
        x0, x1 = layers_first_lbl, layers_first_lbl + nr_wl_labels
        y0, y1 = first_learner_iter, first_learner_iter + nr_weight_vecs
        X_linspace = np.linspace(x0, x1, nr_wl_labels).astype(int)
        Y_linspace = np.linspace(y0, y1, nr_weight_vecs).astype(int)
        x, y = np.meshgrid(X_linspace, Y_linspace)
        # Define the z-values as the weights.
        z = data

        # Define a colormap amont the y-axis, that is to separate the learned iterations by color.        
        norm = plt.Normalize(1, nr_weight_vecs-1)
        colors = cm.viridis(norm(Y_linspace))
        if len(colors.shape) == 2:
            rcount, ccount = colors.shape
        else:
            rcount, ccount, _ = colors.shape

        # Assemble the wireframe plot.        
        fig = plt.figure()

        # Plot the data
        ax = plt.axes(projection ='3d')
        ax.plot_wireframe(x, y, z, rcount=rcount, ccount=ccount, color=colors)
        
        # Define the annotations.
        ax.set_xlabel(f"WL labels [{layers_first_lbl}:{layers_last_lbl}]")        
        ax.set_ylabel("Learner iterations")
        ax.set_yticks(range(0, nr_weight_vecs+1, max(1, nr_weight_vecs//min(nr_weight_vecs, 10))))
        ax.set_zlabel("Edge weight")
        
        title = f'WLLT edge weights in layer {layer_nr}'

        # Format the z-axis. This may change the title of the plot.
        if z_ax_log: 
            ax.set_zscale("log")
            path = f"{path}_zlog"
            title = f"{title}_zlog"        
        if z_limits[0] is not None: ax[1].set_zlim(z_limits)

        ax.set_title(title)
        fig.savefig(f"{path}.png")
        fig.clf(); plt.cla(); plt.clf(); plt.close(fig); plt.close("all")

    def plot_layer_edge_weight_landscapes(self, W: np.array, iter_start: int = 0, normalize_z: bool = False) -> None:
        """
        Plot for each layer, how the edge weights change over time - in each iteration.
        """
        global_min, global_max = None, None
        if normalize_z: global_min, global_max = W.min()-1, W.max()-1
        
        layer_start = 0
        # Iterate through all layers, ...
        for layer_id, next_layer_start in enumerate(self.layer_starts):
            # ...select the data (edge weights for all selected iterations for these layers)...
            data = W[:,layer_start:next_layer_start]
            path = f"{self.dir_out}/WLLT_weights_layer_d{layer_id}"
            # ... and plot them.

            self.save_vectors_wireframe_plot(data, layer_id, (global_min, global_max), layer_start, next_layer_start-1, iter_start, path)
            # self.save_vectors_wireframe_plot(data, layer_id, (global_min, global_max), layer_start, next_layer_start-1, iter_start, path, z_ax_log=True)
            layer_start = next_layer_start

def main(dataset_dir: str, learner_dir: str, dist_mat_eval_coarseness: int = 'avg', dir_out: str = None):
    
    def parse_d_epoch_param(dist_mat_eval_coarseness: str) -> List[float]:        
        if dist_mat_eval_coarseness == 'max':
            return [0, .5, 1.]
        elif dist_mat_eval_coarseness == 'avg':
            return [0, .2, .4, .8, 1.]
        elif dist_mat_eval_coarseness == 'min':
            return [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]        

    # Read the suitable WLLT.
    wllt: WLLT = WLLT()
    wllt = get_WLLT_from_dataset_name(dataset_dir)
    if wllt is None: return None

    # Set up the learner.
    d_epochs: List[float] = parse_d_epoch_param(dist_mat_eval_coarseness)    
    learner_dir: str = f"{c.get_datafiles_dir()}/{c.DN_DATAFILES}/{dataset_dir}/{learner_dir}"    
    learner_eval: LearnerEvaluation = LearnerEvaluation(dir_learner=learner_dir, dir_out=dir_out, wllt=wllt)
    if learner_eval is None: return None

    # Start the evaluation.
    learner_eval.evaluate_learner_results(d_epochs=d_epochs)
    LOG.info("Learner evaluation terminated.")

def parse_terminal_args() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
        default=None,
        dest='dataset_dir',
        help='Directory of the dataset directory (including a WLLT).',
        nargs=1,
        type=str)
    parser.add_argument('-l',
        default=None,
        dest='learner_dir',
        help='Directory name of the edge weight learner.',
        nargs=1,
        type=str)    
    # parser.add_argument('-e',
    #     default=None,
    #     dest='dist_epochs',
    #     help='Domain of the indices where to compute the complete distance matrices. E.g. 0 .2 .4 .8 1.',
    #     nargs='?',
    #     type=List[float])
    parser.add_argument('-n',
        default=5,
        dest='dist_mat_eval_step',
        help='Num in np.linspace(0, 1, num) to compute the indices where the complete distance matrix shall be evaluated. Domain: [3, 5, 10]. Setting -n 5 is equivalent to -e 0 .2 .4 .8 1.',
        nargs='?',
        type=float)
    args = parser.parse_args(sys.argv[1:])
        
    if args.dataset_dir is None or args.learner_dir is None:
        LOG.warn(f"Insufficient name of the dataset & wllt directory '{args.dataset_dir}' or learner directory '{args.learner_dir}'!")
    else:
        main(dataset_dir=args.dataset_dir[0], learner_dir=args.learner_dir[0], dist_mat_eval_coarseness=args.dist_mat_eval_step)    

if __name__ == "__main__":
    # Run in terminal as: python3 x4_eval_learner.py -d MUTAG -l GDL_x
    parse_terminal_args()