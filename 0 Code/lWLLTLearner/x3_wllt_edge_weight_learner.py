from importlib.util import LazyLoader
import numpy as np
import random as rand
from typing import Tuple, List, Any, Dict, Union
import datetime
from os import makedirs, listdir
from os.path import exists, dirname
from logging import ERROR as LOG_ERR
# For the parsing of console input:
import sys, getopt, argparse
# For OOP structures.
from dataclasses import dataclass
from abc import ABC, abstractmethod
from itertools import combinations, chain
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.ticker import MaxNLocator
from time import time

# Own files:
from x2_wllt_constructor import WLLT, get_WLLT_from_dataset_name
from cluster_evaluator import compute_dist_mat
from my_utils.RuntimeTimer import RuntimeTimer, convert_s_to_h_m_s
from my_utils.colorizer import get_ranom_hex_colors
import my_consts as c

# GLOBAL DEBUGGING LOGGER for this script
from my_utils.decorators import get_logger
LOG = get_logger(filename="Loggers/x3_wllt_edge_weight_learner.txt", level=LOG_ERR, create_file=True)

@dataclass
class HyperParams():
    _hyper_param_names  = [
        c.HP_WL_DEPTH, 
        c.HP_UPDATE_FREQ, 
        c.HP_UPDATE_SCOPE, 
        c.HP_UPDATE_INTENSITY, 
        c.HP_LEARNING_RATE, 
        c.HP_BATCH_SIZE, 
        c.HP_LAYER_GRADIENT, 
        c.HP_SINGLE_LAYER, 
        c.HP_CL_PULL_FACTOR, 
        c.HP_CL_PUSH_FACTOR, 
        c.HP_ABSOLUTE_FACTORS, 
        c.HP_HEAVIEST_EARTH_LIM, 
        c.HP_WEIGHT_CEILING,
        c.HP_DESCRIPTION
    ]
    _hyper_param_values = [None] * len(_hyper_param_names)

    def __init__(self, wl_depth: int = None, uf: str = None, us: str = None, ui: str = None, lr: float = None, bs: int = None, l_grad: np.array = None, single_layer: int = None, f_pull: Union[float, List[float]] = None, f_push: Union[float, List[float]] = None, abs_update: bool = None, heaviest_earth_thld: float = None, weight_ceiling: bool = None, descr: str = "LearnerInit"):
        self.config_file_path: str = c.CSV_EW_LEARNER_CONFIG
        self.update_selected_params(wl_depth, uf, us, ui, lr, bs, l_grad, single_layer, f_pull, f_push, abs_update, heaviest_earth_thld, weight_ceiling, descr)
    
    def set_config_file_path(self, path: str) -> None:
        self.config_file_path = path

    def set_hyper_param_value(self, param_name: str, value: Any) -> None:
        i = self._hyper_param_names.index(param_name)
        if i is not None:
            self._hyper_param_values[i] = value
        else:
            LOG.error(f"Unknown hyperparameter named {param_name} could not be set to value {value}!\nKnown params: {self._hyper_param_names}")

    def get_value(self, param_name: str) -> None:
        i = self._hyper_param_names.index(param_name)
        if i is not None:
            return self._hyper_param_values[i]
        else:
            LOG.error(f"Unknown hyperparameter named {param_name}!\nKnown params: {self._hyper_param_names}")

    def get_params_name_value_list(self) -> List[Tuple[str, Any]]:        
        return zip(self._hyper_param_names, self._hyper_param_values)

    def get_params_name_value_string(self) -> str:
        lst = self.get_params_name_value_list()
        ret_str = ""
        for k, v in lst:
            ret_str += f"{k}:\t{v}\n"

        return ret_str

    def update_selected_params(self, wl_depth: int = None, uf: str = None, us: str = None, ui: str = None, lr: float = None, bs: int = None, l_grad: np.array = None, single_layer: int = None, f_pull: Union[float, List[float]] = None, f_push: Union[float, List[float]] = None, abs_update: bool = None, heaviest_earth_thld: float = None, weight_ceiling: bool = None, descr: str = None) -> None:
        """
        This method allows to set and adjust key parameters of the weight learning procedure.
        """
        _hp_values = [wl_depth, uf, us, ui, lr, bs, l_grad, single_layer, f_pull, f_push, abs_update, heaviest_earth_thld, weight_ceiling, descr]
        for i, value in enumerate(_hp_values):
            if value is not None:
                self._hyper_param_values[i] = value
            
    def write_config_csv_file(self, path: str = None, write_mode: str = 'w', header_row: str = '', delimiter: str = c.CSV_DELIMITER) -> None:
        if path is None: path = self.config_file_path

        values: np.array = np.array([str(v) for v in self._hyper_param_values])
        if write_mode == 'w': header_row = delimiter.join(self._hyper_param_names)

        with open(path, write_mode) as f:
            np.savetxt(f, values.reshape(1, values.shape[0]), fmt='%s', delimiter=delimiter, header=header_row)    

    def read_config_csv_file(self, path: str, delimiter: str = c.CSV_DELIMITER) -> Tuple[str, Any]:
        """
        Returns the headder row and a list of list (matrix) for all rows in the config file:
        row, matrix
        """
        # hyper_param_values = np.loadtxt(path, delimiter=delimiter, skiprows=1)
        hyper_param_values = np.genfromtxt(path, dtype=str, delimiter=delimiter, encoding=None, skip_header=1) 
        return self._hyper_param_names, hyper_param_values

        
class EdgeWeightLearner(ABC):
    """   
    Interface for all edge weight learning implementations.
    Independent of hyperparameter values regarding the weight update frequency, scope or intensity.
    """
    def __init__(self, wllt: WLLT, nr_epochs: int, print_sme_estimate: bool = False, continue_existing_learner_dir: str = None, dir_out_suffix: str=''):
        
        def _get_nr_completed_epochs_from_files(dir_in: str) -> Tuple[np.array, np.array]:            
            # Read in the edge weight files.
            max_epoch: int = 0            
            if dir_in is not None:
                file_predix: str = f"{c.FN_EDGE_WEIGHTS_E}"
                file_name_dict = c.get_file_name_versions(dir_in=dir_in, file_predix=file_predix)
                epochs = list(file_name_dict.keys())
                max_epoch = int(np.max(epochs))

            return max_epoch
        
        self.wllt: WLLT = wllt
        if self.wllt is None: return None
        self.nr_epochs: int = nr_epochs

        self.print_sme_estimate: bool = print_sme_estimate
        self._continue_existing: bool = (continue_existing_learner_dir is not None)
        # The hyperparameters can be accessed and updated using the class instance.
        self.hyper_params: HyperParams = HyperParams()   
        self.timer: RuntimeTimer = RuntimeTimer()     
        self.path_in, self.path_out = self._init_io_paths(wllt_dir=self.wllt.get_output_dir(), existing_learner_dir=continue_existing_learner_dir, dir_out_suffix=dir_out_suffix)
        
        # The CURRENT epoch ot the edge weight iteration counter is the last COMPLETED iteration. 
        # That an edge weight file with this number has been stored.        
        # If a leaner is continued, there will be files stored in the directory and its maximum epoch will be used for the initialization.
        # Otherwise '0' will be the first epoch, and a copy of the WLLT edge weights is stored in such a file.
        self._completed_epochs: int = _get_nr_completed_epochs_from_files(self.path_in)

        if not self._continue_existing:
            # In this case, load the edge weights initialized in the WLLT.
            self.old_edge_weights: np.array = self.wllt.get_edge_weights_from_file()
            # Save the zero weights with suffix '0'.            
            np.save(self.get_path_edge_weights(iteration=self._completed_epochs, to_output_dir=True), self.old_edge_weights)
        else:
            # In this case, load the highest edge weights stored in the directory.            
            # There is no need to store them again.
            self.old_edge_weights = np.load(self.get_path_edge_weights(iteration=self._completed_epochs))               
        
        # Get the necessary datastructures from the wllt.
        self.graph_representations_csr = self.wllt.get_graph_repr_from_file().tocsr()
        self.nr_graphs, self.nr_wl_labels = self.graph_representations_csr.get_shape()

        # The vector 'v' indicating which class 'v[i]' graph 'i' has.
        self.graph_classes_vec: np.array = self.wllt.get_graph_classes_vec_from_file()        
        # The list of different classes represented in the dataset.
        self.different_graph_classes:     np.array = np.unique(self.graph_classes_vec)
        
        # Store the distances between all graphs in a lower triangle matrix vector.
        # Distances 'd(i, j) = d(j, i)' for 'i < j' are stored in entry '(n*(n-1)/2) - (n-i)*((n-i)-1)/2 + j - i - 1' (n = self.nr_graphs)
        nr_pairwise_distances = self.nr_graphs * max(1, (self.nr_graphs - 1) // 2)
        self.dist_matrix = np.zeros((nr_pairwise_distances), dtype=np.float32)
        # Since the dist matrix may just be computed for a fraction, store the graph indices its rows and columns represent.
        self.dist_matrix_indices = np.array(list(range(self.nr_graphs)), dtype=np.int32)

    def __repr__(self) -> str:
        return self._get_representation_str()

    def get_output_path(self) -> str:        
        return self.path_out

    def get_input_path(self) -> str:
        return self.path_in

    def get_path_edge_weights(self, iteration: int = None, new_iteration: bool = True, iteration_independent: bool = False, to_output_dir: bool = False) -> str:
        path_from_root = self.path_in
        if to_output_dir or self.path_in is None: path_from_root = self.path_out
        path = f"{path_from_root}/{c.FN_EDGE_WEIGHTS_E}"
        if iteration_independent:
            return path
        else:
            return self.append_iteration_to_filepath(path, iteration, new_iteration)

    def get_path_dist_mat(self, iteration: int = None, new_iteration: bool = True, appendix: str = "") -> str:
        path =  f"{self.path_out}/{c.FN_DIST_MAT_I}{appendix}"
        return self.append_iteration_to_filepath(path, iteration, new_iteration)

    def get_path_dist_mat_ids(self, iteration: int = None, new_iteration: bool = True, appendix: str = "") -> str:
        path =  f"{self.path_out}/{c.FN_DIST_MAT_IDS_I}{appendix}"
        return self.append_iteration_to_filepath(path, iteration, new_iteration)

    def get_distance(self, i: int, j: int) -> float:
        # Using the hints from: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist
        index = self.nr_graphs * i + j - ((i+2))*(i+1)
        return self.dist_matrix[index]

    def randomize_class_vec(self, new_graph_classes: List[int] = None) -> Tuple[float, float]:
        old_graph_class_vec = self.graph_classes_vec.copy()
        old_graph_classes   = self.different_graph_classes.copy()
        if new_graph_classes is None: 
            new_graph_classes = self.different_graph_classes
        else: 
            new_graph_classes = np.unique(new_graph_classes)
            self.different_graph_classes = new_graph_classes
        
        self.graph_classes_vec: np.array = np.random.choice(new_graph_classes, size=self.nr_graphs, replace=True)
        np.save(f"{self.path_out}/{c.FN_GRAPH_CLASSES.replace('.npy', '_randomized.npy')}", self.graph_classes_vec)
        
        # Compute the amount of same class assignment:
        n_same_class_assignments = (old_graph_class_vec == self.graph_classes_vec).sum()
        # Compute the amount of same avaliable classes:
        n_same_classes           = (old_graph_classes == self.different_graph_classes).sum()
        
        return n_same_class_assignments / self.nr_graphs, n_same_classes / self.nr_graphs

    def _compute_batch_size(self) -> int:
        """ The hyper parameter batch size is interpreted as a percentage of the dataset if it has
        value <= 1.0 and as an absolute size otherwise.
        """        
        batch_size = self.hyper_params.get_value(c.HP_BATCH_SIZE)
        # The batch size can be given as natrual number (absolut value) or as percentage (relative value) in (0,1]
        if batch_size <= 1.0:
            abs_batch_size = int(len(self.graph_classes_vec) * batch_size)
        else:
            abs_batch_size = int(batch_size)
        
        return max(2, abs_batch_size)

    def _compute_class_imbalance_factor_for_batches(self) -> Tuple[float, float]:
        """
        Given the constant batch size 'm' and that there are 'n' samples for every of 'c' classes in each batch, 
        return the factor to account for an imbalanced distribution when two samples of the same class, and 
        of two different classes are selected.
        
        There are 
            (n)(n-1)/2 * c
        possibilities to draw to samples from the SAME class.
            The division factor is '{n choose 2}'. That is '(n)(n-1)/2' possibilities to draw the first sample from one class and
            a second - different - sample from the same class. Divided by 2 since every sample was counted twice. 
            Multiplied by 'c' since this has to be done for every class.

        There are 
            (nn)*(c(c-1)/2)
        possibilities to draw to samples two DIFFERENT classes. 
        ('nn' for any two samples of two different classes and 'c(c-1)/2={c choose 2}' for the number of possibilities to select two such classes.

        Since both products contain the factor (nc/2), one may drop it by scaling. 
        Thus we get the following factors as return values:

        #SameClass:     (n-1)
        #DiffClass:     (n)(c-1)

        It is possible to use these factors '1 / #SameClass' and '1 / #DiffClass' directly.
        However their values may be close to zero and have a significant scaling effect on the weight update.

        To diminish this effect they are scaled with a factor S, such that the bigger factor has value of at most 1.0.

        This way the two factors that are returned have approximate values 
            1.0 and
            1 / ((n-1)n(c-1))
        The order is determined by identifying the maximum between '1 / #SameClass' and '1 / #DiffClass'.
        
        Return: Factor for samples in the same class:       S / #SameClass
        Return: Factor for samples in different classes:    S / #DiffClass
        """
        c: int = len(self.different_graph_classes)
        m: int = self._compute_batch_size()        
        n: int = m // c

        nr_same_class = n - 1
        nr_different_classes = n * (c - 1)
        
        if nr_different_classes == 0 or nr_same_class == 0: 
            LOG.warning(f"Number of clusters {c}, batch size {m} and thus cluster representation size in the batches {n} do not yield a suitable setting!")
            return None, None

        factor_same = 1 / nr_same_class
        factor_diff = 1 / nr_different_classes
        
        scale_to_unit = np.floor(1/max(factor_same, factor_diff))
        factor_same *= scale_to_unit
        factor_diff *= scale_to_unit
        
        return factor_same, factor_diff

    def _get_representation_str(self) -> str:
        return f"Edge Weight Learner. Outputs in: {self.path_out}"

    def _latest_edge_weight_iteration(self) -> int:
        """ Returns the highest iteration that has been completed. There should be learned edge weights for this iteration. """
        return self._completed_epochs
    
    def _next_edge_weight_iteration(self) -> int:
        """ Returns the next iteration that has to be completed. There should be no learned edge weights for this iteration yet. """
        return self._completed_epochs + 1

    def _increment_ew_epoch_ctr(self) -> None:
        self._completed_epochs += 1

    def append_iteration_to_filepath(self, path: str, iteration: int = None, new_iteration = True) -> str:
        """
        Returns the path to a file which stores an edge weight vector.
        If iterations is not None, and the value does not exeed the currently known iteration,
        the right path is returned.
        If iterations is None, and a new iteration is requested the path to an iteration, which has
        not been saved jet is returned (value of the iteration counter plus one).
        If iterations is None but an old iteration is requested the path to the last known iteration
        is returned (value of the iteration counter).
        """
        if iteration is not None and iteration <= self._latest_edge_weight_iteration():
            return f"{path}{iteration}.npy"
        elif new_iteration:
            return f"{path}{self._next_edge_weight_iteration()}.npy"
        else:
            return f"{path}{self._latest_edge_weight_iteration()}.npy"

    def fit(self) -> None:
        self.new_edge_weights: np.array = self._improve_edge_weights()
    
    def get_dist_mat_path(self) -> str:
        return f"{self.path_out}/dist_mat.npy"

    def get_last_saved_dist_mat(self) -> np.array:
        return np.load(self.get_dist_mat_path())
    
    def get_edge_weights_from_file(self, iteration: int = None) -> np.array:
        """
        Reads the edge weight vector from file.
        If iteration is set to None, the newest edge weights are returned.
        """
        # If the zeroth iteration is called, that is the one initialized by the wllt.
        # This one is saved during the initialization of the class insance.
        # In this case we have to check, whether this file may store weights for more edges, 
        # that how many we are considering at the moment. Thus prune the input in this case.        
        if iteration == 0: 
            return self.wllt.get_edge_weights_from_file()
        else:
            path: str = self.get_path_edge_weights(iteration, new_iteration=False)
            return np.load(path)

    def get_edge_weights(self) -> np.array:
        return self.new_edge_weights
    
    def save_current_dist_mat(self) -> None:
        return np.save(self.get_dist_mat_path(), self.dist_matrix.astype(np.float32))

    def update_hyper_params(self, wl_depth: int = None, uf: str = None, us: str = None, ui: str = None, lr: float = None, bs: int = None, l_grad: np.array = None, single_layer: int = None, f_pull: Union[float, List[float]] = None, f_push: Union[float, List[float]] = None, abs_update: bool = None, heaviest_earth_thld: float = None, weight_ceiling: bool = None, description: str = None) -> None:
        self.hyper_params.update_selected_params(wl_depth, uf, us, ui, lr, bs, l_grad, single_layer, f_pull, f_push, abs_update, heaviest_earth_thld, weight_ceiling, descr=description)
        self.write_config_csv(append=self._continue_existing)

    def load_hyper_param_values(self) -> None:
        self.hp_he_thld         = self.hyper_params.get_value(c.HP_HEAVIEST_EARTH_LIM)
        self.hp_single_layer    = self.hyper_params.get_value(c.HP_SINGLE_LAYER)
        self.hp_f_pull          = self.hyper_params.get_value(c.HP_CL_PULL_FACTOR)
        self.hp_f_push          = self.hyper_params.get_value(c.HP_CL_PUSH_FACTOR)
        self.hp_abs_pp_factor   = self.hyper_params.get_value(c.HP_ABSOLUTE_FACTORS)
        self.hp_learning_rate   = self.hyper_params.get_value(c.HP_LEARNING_RATE)
        self.hp_weight_ceiling  = self.hyper_params.get_value(c.HP_WEIGHT_CEILING)

    def write_config_csv(self, filename: str=c.CSV_EW_LEARNER_CONFIG, append: bool=False, suffix: str="") -> None:
        write_mode = 'w'
        if append: write_mode = 'a'

        path = f"{self.path_out}/{filename}{suffix}"
        self.hyper_params.write_config_csv_file(path, write_mode)

    def _convert_dist_matrix_indices_to_vector_index(self, i: int, j: int) -> int:
        """
        Converts the row and column index of a distance matrix to the vector index.
        Since the distance between the same element is zero, do not expect such an input.
        First, make sure that 'i' is the smaller row index and 'j' the bigger column index, to 
        reference the upper triangle, which is stored in some vector.
        Then compute the vector index to this matrix element stored in the vector.
        """
        if i == j: 
            return None
        elif i > j: 
            i, j = j, i
        # Check if the bigger index is to big.
        if j > self.nr_graphs: 
            LOG.error(f"Invalid distance matrix indices: j={j} > {self.nr_graphs} = nr graphs (i was {i})")
            return None

        return (self.nr_graphs*(self.nr_graphs-1)/2) - (self.nr_graphs-i)*((self.nr_graphs-i)-1)/2 + j - i - 1

    def get_graph_distance(self, i, j) -> float:
        """
        Takes indices (i, j) in a distance matrix (i, j < self.nr_graphs) and
        returns their distance.
        """
        k = self._convert_dist_matrix_indices_to_vector_index(i, j)
        
        if k is None:
            return 0.0
        else:
            return self.dist_matrix[k]    

    def _init_io_paths(self, wllt_dir: str, existing_learner_dir: str = None, dir_out_suffix: str = '') -> str:
        """
        Create an output directory 'GDL_<time_stamp>' at the same level as the used WLLT.        
        """        
        def get_timestamp_dir_name() -> str:
            time_stamp = datetime.datetime.now().strftime('%d_%Hh-%Mm')
            return f"GDL_{time_stamp}"
      
        path_from_root: str = dirname(wllt_dir)

        # Construct the input and output directory paths.
        if existing_learner_dir is None:
            # Do not use an existing dir. Create the output directory using a time stamp and possibly a suffix for the dir name.
            # There is no input directory.
            path_in     = None
            path_out    = f"{path_from_root}/{get_timestamp_dir_name()}{dir_out_suffix}"
        else:
            # Use an existing dir. It is saved as input directory.
            # Create the output directory INSIDE the input directory (subdirectory) using a time stamp, a 'continuation' suffix and possibly another given suffix.
            path_in     = f"{path_from_root}/{existing_learner_dir}"
            path_out    = f"{path_in}/{get_timestamp_dir_name()}_continuation{dir_out_suffix}"
        
        # Create the required directories.
        if exists(path_out):
            # If the path already exists, add a (not existing) version-suffix to it.
            version_ctr = 0
            while exists(f"{path_out}_v{version_ctr}"): version_ctr += 1
            path_out = f"{path_out}_v{version_ctr}"
        else:
            # If the path does not already exist, create it.
            makedirs(path_out)        
        
        return path_in, path_out
    
    @abstractmethod
    def _improve_edge_weights(self) -> np.array:
        """
        Try to improve the edge weights for the given WLLT, 
        returns them as array and 
        saves them to the file, specified by the WLLT.
        """        
        raise NotImplementedError
    
class Default_Learner(EdgeWeightLearner):
    def __init__(self, wllt: WLLT, nr_epochs: int = 1, print_sme_estimate: bool = False, continue_existing_learner_dir: str = None, dir_out_suffix: str=''):
                        
        super(Default_Learner, self).__init__(wllt=wllt, nr_epochs=nr_epochs, print_sme_estimate=print_sme_estimate, continue_existing_learner_dir=continue_existing_learner_dir, dir_out_suffix=dir_out_suffix)
        
        # Hyper parameter adjustments.
        uf = "Epoch"
        us = "All layers equal"
        ui = "Push and Pull"
        l_grad, abs_update = None, None
        self.update_hyper_params(uf=uf, us=us, ui=ui, l_grad=l_grad, abs_update=abs_update)
                
        self.layer_starts: np.array = wllt.get_layer_starts_from_file()        

        self.local_sample_movement_errors: List[float] = list()
        self.local_sample_movement_error: float = 0.0
        
    def __repr__(self):        
        return self._get_representation_str()

    def _get_representation_str(self):
        return_str: str = super()._get_representation_str()
        return_str += f"\nDefault Learner: Pic the {self.hyper_params.get_value(c.HP_HEAVIEST_EARTH_LIM)} most expensive differences in the graphs representation and change their weights by a percentage."
        return_str += self.hyper_params.get_params_name_value_string()
        return return_str

    def save_in_one_plot(self, values, title: str, image_name: str) -> None:
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111, label=title)
        ax.set_title(title)
        # Do not allow for custom y-axis formatting with scientific notations.
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        ax.set_xlabel("Iterations"); #ax.set_xticks(range(len(xticks_labels)))
        # ax.set_xticklabels(xticks_labels, rotation=45)
        ax.plot(values)
        ax.legend()        
        fig.savefig(f"{self.path_out}/{image_name}")
        fig.clf(); plt.cla(); plt.clf(); plt.close(fig); plt.close("all")

    def save_batch_histograms_plot(self, batchs_list: List[List], title: str, image_name: str, cl: np.array = None) -> None:
        if cl is None: cl = self.graph_classes_vec
        used_graphs = list(chain(*batchs_list))
        labels, values = zip(*Counter(used_graphs).items())
        color_cl = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']        
        n_have, n_need = len(color_cl), np.max(cl) + 1
        if n_need > n_have:
            color_cl += get_ranom_hex_colors(n=n_need-n_have)
        
        colors = [color_cl[i] for i in cl]

        indexes = np.arange(len(labels))

        width = 0.8
        _bar_width = 0.5
        
        # Create Plot.        
        fig = plt.figure()
        ax = fig.add_subplot(111, label=title)
        ax.set_title(title)
        # Plot the stacked bars.
        ax.bar(indexes, values, width=_bar_width, label="Graph frequencies in batches",  color=colors)

        ax.set_xlabel("Graphs")
        ax.set_xticks(indexes + width * 0.5); ax.set_xticklabels(labels) #, rotation=45)
        ax.set_ylabel("Frequency")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{self.path_out}/{image_name}")
        fig.clf(); plt.cla(); plt.clf(); plt.close(fig); plt.close("all")

    def _improve_edge_weights(self) -> np.array:
        
        def construct_batch_id_lists() -> List[np.array]:
            """
            Returns a list (with lenght: 'nr_epochs'), containing a list (with lenght: nr of classes),
            containing BATCH_SIZE many randomly shuffled ids of samples in the corresponding class.
            """
            rand.seed(42)
            graph_classes_ids = [np.where(self.graph_classes_vec == c)[0] for c in self.different_graph_classes]

            # Find the smallest class to compare its size with the batch size.
            min_class_size = np.inf
            min_class = None
            for i, member_list in enumerate(graph_classes_ids):
                if len(member_list) < min_class_size:
                    min_class_size = len(member_list)
                    min_class = self.different_graph_classes[i]                 
                    
            batch_size = self._compute_batch_size()

            # If the batch size divided by the number of classes is bigger than the amount of samples for the smallest class, report this.
            class_batch_size = max(1, batch_size // len(self.different_graph_classes))
            up_sampling = min_class_size < class_batch_size
            if up_sampling:
                msg = f"A batch size of {batch_size} is depreciated,"
                msg += "since there are {len(graph_classes)} classes which results in {class_batch_size} samples for each class.\n"
                msg += f"And class {min_class} has only {min_class_size} samples! NO upsampling implemented atm."
                LOG.warning(msg)

            # For each epoch select batch size / nr of classes many samples from every class.
            # This way the actual batch size may smaller, but all classes are represented by equally many examples.
            epoch_candidate_ids: List[np.array] = []
            for i in range(self.nr_epochs):                
                batch_candidates = np.array([np.random.choice(class_j, size=class_batch_size, replace=False) for class_j in graph_classes_ids]).flatten()
                np.random.shuffle(batch_candidates)
                epoch_candidate_ids += [batch_candidates]
                LOG.debug(f"Epoch {i}: {class_batch_size} samples per class ({self.different_graph_classes}):\n{batch_candidates}")

            LOG.debug(f"Batches for all {self.nr_epochs} epochs, with {class_batch_size} samples per class ({self.different_graph_classes}) are:\n{epoch_candidate_ids}")
            return epoch_candidate_ids
        
        def compute_full_sme(w: np.array) -> float:
            # Compute the full distance matrix.
            dist_mat, _, _ = compute_dist_mat(self.graph_representations_csr, w, print_used_method=False)
            # Compute the sum of all distances of all samples of the same class for each class.
            intra_cl_dist_sum, inter_cl_dist_sum = 0.0, 0.0
            for cl_name in self.different_graph_classes:
                cl_ids = np.where(self.graph_classes_vec == cl_name)[0]
                intra_cl_dist_sum += np.sum(dist_mat[np.ix_(cl_ids, cl_ids)])
            
            # Compute the sum of all distances of all samples of different classes for each pair of classes.
            pairwise_different_cl_ids = list(combinations(self.different_graph_classes, 2))
            for cl_tuple in pairwise_different_cl_ids:
                cl0_ids = np.where(self.graph_classes_vec == cl_tuple[0])[0]
                cl1_ids = np.where(self.graph_classes_vec == cl_tuple[1])[0]
                inter_cl_dist_sum += np.sum(dist_mat[np.ix_(cl0_ids, cl1_ids)])            
            
            # Scale the distance just to keep the values down.
            return (intra_cl_dist_sum - inter_cl_dist_sum)

        def write_runtimes_to_file(batch_size: int) -> None:
            rt_names: List[str]  = self.timer.get_all_runtime_names()
            rt_values: List[str] = self.timer.get_all_runtime_values(convert_float_to_str=False)
            
            rt_name:    str = rt_names[0]
            rt_str:     str = convert_s_to_h_m_s(rt_values[0])
            rt_epoch:   str = convert_s_to_h_m_s(rt_values[0]/self.nr_epochs)
            rt_batch:   str = convert_s_to_h_m_s(rt_values[0]/self.nr_epochs/batch_size)

            rt_header:  str = c.CSV_DELIMITER.join(["Name",  "Total runtime", "Nr. Epochs",    "Runtime per epoch",  "Batch siye",   "Runtime per batch" ])
            rt_print:   str = c.CSV_DELIMITER.join([rt_name,  rt_str,      str(self.nr_epochs), rt_epoch,         str(batch_size),    rt_batch            ])
            rt_lines: List[str] = [rt_header, rt_print]

            with open(f"{self.get_output_path()}/Runtimes.csv", 'w') as f:
                np.savetxt(f, rt_lines, fmt='%s', delimiter=c.CSV_DELIMITER)

        # Load the edge weights. Either from the wllt, or as results from an existing Learner.
        w: np.array = self.get_edge_weights_from_file(iteration=self._completed_epochs)
        
        # Load various hyper parameters. Define the batches.
        self.load_hyper_param_values()
        same_diff_class_imba_tuple: Tuple[float, float] = self._compute_class_imbalance_factor_for_batches()        
        layer_gradient: List[float] = np.repeat([1.0], len(self.layer_starts))
        batches_ids: List[np.array] = construct_batch_id_lists()

        _print_suffix: str = ''
        _max_prefix_length: int = len(f"Epoch {self._completed_epochs + self.nr_epochs}: 100.00%\t({self.nr_epochs}\t/{self.nr_epochs})" + ' ' *5)        
        epochs_todo = range(self._completed_epochs + 1, self._completed_epochs + self.nr_epochs + 1)
        for i, epoch in enumerate(epochs_todo):
            start_time = time()
            # Assemble a print string.
            if self.print_sme_estimate:
                sme = round(compute_full_sme(w), 2)
                # Save the first iteration sme, for comparison, and norming all other sme's to mean zero.
                if i == 0: zero_sme = sme
                sme = sme / zero_sme - 1.0
                _print_sme = f"SME: {round(sme, 2)}"                
            _print_prefix = f"Epoch {epoch}: {round((i+1)*100/self.nr_epochs, 2)}%\t({i+1}\t/{self.nr_epochs}){_print_sme if self.print_sme_estimate else ''}"
            _print_prefix += " " * (_max_prefix_length - len(_print_prefix))

            w = self.run_training_epoch(w, batches_ids[i], same_diff_class_imba_tuple, layer_gradient, _print_prefix, _print_suffix)
            
            # Estimate the remaining runtime for all next epochs.
            n_todo = len(epochs_todo) - i
            expected_time: str = convert_s_to_h_m_s((time() - start_time) * n_todo)
            _print_suffix = f" RT.Estimation: {expected_time}"

            # Every tenth epoch, save the computed edge weights. Notice, to avoid division by zero if 'self.nr_epochs' < 10.
            condition_save_edge_weight = ((i+1) % (self.nr_epochs//min(self.nr_epochs, 10)) == 0)
            if condition_save_edge_weight:                
                np.save(self.get_path_edge_weights(iteration=self._latest_edge_weight_iteration(), to_output_dir=True), w)
        
        print()
        if self.print_sme_estimate: self.save_in_one_plot(self.local_sample_movement_errors, title="Local sample movement error", image_name="plot_local_sample_movement_error.png")
        write_runtimes_to_file(batch_size=len(batches_ids[0]))
        self.write_config_csv(suffix=f"_{self.wllt.dataset_name}")
        self.save_batch_histograms_plot(batches_ids, title=f"Frequencies of all graphs in\nall {len(batches_ids)} batches", image_name="plot_batch_barchary.png")
        # Update the information on how many epochs were performed.
        self._completed_epochs += self.nr_epochs
        LOG.info(f"All {self.nr_epochs} epochs complete. Storing and plotting the evaluations.")
        print()        
        return w

    # KEY LEARNER METHODS

    def get_candidate_indices(self, difference: np.array, single_layer: int = None) -> np.array:
        """
        Given the vector of the weighted differences between the graph representations, 
        decide which weights will be updated.
        If the 'single_layer' flag is positive, the candidates are computed using only weights to leaves of the WLLT.
        If 'heaviest_earth_threshold' is zero, understand this as 'heaviest_earth_threshold=np.INF'.
        """
        def get_candidate_indices_of_layer(difference: np.array, he_thrshld: int) -> np.array:
            # Sort the array and get the n-highest values (where n = 'he_thrshld').
            values_small_to_high = np.sort(difference)
            # Discard differences with value zero. 
            # Such entries do not contribute to the distance computation between these graphs and shall not be changed.
            values_small_to_high = values_small_to_high[np.where(values_small_to_high!=0.0)]
            
            if len(values_small_to_high) > he_thrshld:
                min_value = values_small_to_high[-he_thrshld]
            else:
                # If there are not enough highest values, select the smallest (non-zero) of them.
                # All values are bigger than this smalles one, and will be selected in the next step.
                min_value = values_small_to_high[0]

            # Notice that only the INDICES of the values are used.
            # Thus more weights than just 'self.heaviest_earch_threshold' many may be updated, if distances occur more frequently.            
            candidate_indices = np.where(difference >= min_value)[0]
            return candidate_indices

        if self.hp_he_thld == 0.0:
            # If the heaviest earth threshold is set to zero (which in general makes no sense, since no weight would be updated,)
            # interpret this such that ALL weights are updated. This is the same as setting 'heaviest_earth_thld=np.INF'.
            candidate_indices = list(range(len(difference)))
            return candidate_indices
        else:        
            # Select indices from all layers separately.
            all_layer_candidates = np.array([], dtype=int)
            first_layer_vertex   = None
            chosen_layer_starts  = None
            
            # Select the layers from which the candidates will be chosen.
            if single_layer is not None: 
                chosen_layer_starts = [self.layer_starts[single_layer]]
                if single_layer == 0:
                    first_layer_vertex = 0
                else:
                    # The first vertex of layer x has index one higher than the last of layer x-1.
                    first_layer_vertex = self.layer_starts[single_layer-1] + 1
            else:
                chosen_layer_starts = self.layer_starts
                first_layer_vertex = 0
            
            # Iterate throught the selected layers and chose candidates.
            for last_layer_vertex in chosen_layer_starts:              
                layer_size = last_layer_vertex - first_layer_vertex
                assert layer_size > 0, "Layer selection in 'get_candidate_indices' went wrong."
                # If there are no differences in this layer, change nothing and continue with the next layer.
                if np.sum(difference[first_layer_vertex:last_layer_vertex]) == 0: continue
                # If 'heaviest_earth_thld' is in the interval (0,1), it indicates the percentage for the current layer, 
                # of highest values that shall be updated.
                if self.hp_he_thld == 'all':
                    heaviest_earth_thld_abs = layer_size
                # If the threshold is passed on as a percentage ([0, 1]),  compute the absolut value.
                elif 0 < self.hp_he_thld and self.hp_he_thld <= 1.0:
                    heaviest_earth_thld_abs = int(self.hp_he_thld * layer_size)
                # Otherwise expect an absolut value right away.
                else: 
                    heaviest_earth_thld_abs = int(self.hp_he_thld)

                this_layer_candidates = get_candidate_indices_of_layer(difference[first_layer_vertex:last_layer_vertex], heaviest_earth_thld_abs)
                
                # Offset the candidate indices from the local layer.
                this_layer_candidates = this_layer_candidates + int(first_layer_vertex)
                all_layer_candidates = np.append(all_layer_candidates, this_layer_candidates)
                first_layer_vertex = last_layer_vertex

            return all_layer_candidates.astype(int)

    def update_edge_weights(self, cand_ids: np.array, same_class: bool, edge_weights: np.array, same_diff_class_imba_tuple: Tuple[float, float]) -> Tuple[np.array, np.array]:
        """        
        This method pics the 'self.heaviest_earth_threshold' highest differences in the weighted difference vector.
        It contains the most expensive eath movings, given the current weight vector.
        (Note: If the selection would be independent of the weight vector, it would not change for all iterations.)
        It then proceeds to update the edge weights concerning these differences/wl-labels.

        If the graphs are in
            - the same class and thus shall have a small distance,  decrease..
            - different classes and thus shall have a big distance, increase..
        ..the involved edge weights by 'percentage' percent.
        """
        # Get the required hyper parameter values.        
        same_cl_factor, diff_cl_factor = same_diff_class_imba_tuple
        
        # Compute the delta weights for the candidate indices. Zip them with the edge weight indices that they are refering to.
        weight_delta_factor = 0.0
        if same_class:
            weight_delta_factor = (-1) * self.hp_f_pull * same_cl_factor                    
        else:
            weight_delta_factor =  (1) * self.hp_f_push * diff_cl_factor                    

        # Computation of \Delta w.
        if self.hp_abs_pp_factor:
            # In this case, understand the push and pull factors as fixed increment, which is added to the weight.                 
            weight_delta_for_cand: np.array = np.repeat(weight_delta_factor, len(cand_ids))
        else:
            # Compute the factor which scales the weight delta. It contains several other factors.                
            weight_delta_for_cand: np.array = weight_delta_factor * edge_weights[cand_ids]

        return weight_delta_for_cand
    
    def run_training_epoch(self, edge_weights: np.array, batch_ids: np.array, same_diff_class_imba_tuple: Tuple[float, float], layer_gradient: List[float], _print_prefix: str = '', _print_suffix: str = '') -> np.array:
        """
        This method takes a set of edge weights, updates it and returns the updated weights.
        The update procesure goes as follows:
            - Compute the Tree Wasserstein Distance between two graphs
                (That is the normalized weighted distance between their representations - which are wl-label histograms.)
            - Save this distance in a sparse distance matrix.
            - Call a method to update the edge weights. Usually this only concernes weights 
              on the path between the distances of the grpahs representations.
        """
        
        def compute_epoch_weight_delta(w_delta_sum_ctr: np.array) -> np.array:
            # Apply the computed weight updates. Take the mean of all weight updates.        
            epoch_w_delta: np.array  = np.zeros(n_weights)
            layer_id: int = 0
            for i, sum_ctr_tuple in enumerate(w_delta_sum_ctr):
                # Keep trak of the index of the layer of the current weights - to get the right layer_gradient.
                if self.layer_starts[layer_id] <= i: layer_id += 1

                # Compute the mean for all changes that shall be applied to an edge.
                if sum_ctr_tuple[1] > 0: 
                    epoch_w_delta[i] = layer_gradient[layer_id] * sum_ctr_tuple[0]
                    # If not an absolute update, average the updates.
                    if not self.hp_abs_pp_factor:
                        epoch_w_delta[i] /= sum_ctr_tuple[1]
            
            return epoch_w_delta

        nr_push = 0; nr_pull = 0; set_of_touched_weights = set()

        # Iterate over the upper triangle of the distance matrix between all graphs:
        pair_ctr:           int = 1
        batch_size:         int = len(batch_ids)
        n_weights:          int = len(edge_weights)
        nr_of_graph_pairs:  int = int(batch_size * (batch_size - 1) / 2)
        # Keep trak of all weight deltas for every edge weight. Later, the mean will be taken and used to update the weights.
        weight_delta_sum_ctrs: np.array = np.array([(float(0.0), int(0)) for _ in range(n_weights)])

        if self.print_sme_estimate:
            # Trak the sample movements during the improvement step:
            # Compute the distance sum of the same class samples in the batch.
            same_cl_dist_sum_old: float = 0.0
            same_cl_dist_sum_new: float = 0.0
            # Compute the distance sum of the different class samples in the batch.
            diff_cl_dist_sum_old: float = 0.0
            diff_cl_dist_sum_new: float = 0.0
            nr_improvements: int = 0
            nr_degradations: int = 0            
            nr_same_sme: int     = 0 

        self.timer.start_timer(c.RT_Epoch)
        for i0, g0_id in enumerate(batch_ids):
            g0_repr     = self.graph_representations_csr[g0_id]
            g0_class    = self.graph_classes_vec[g0_id]
            
            for g1_id in batch_ids[i0+1:]:
                _print_str = f"{_print_prefix}\tPairs: \t{round(pair_ctr*100/nr_of_graph_pairs, 2)}%\t({pair_ctr}\t/{nr_of_graph_pairs})\t{_print_suffix}" + " " * 3
                 # Clean the terminal print. The usage of '\t' allows for previous values to remain even after using '\r'.
                print(f"\r" + " " * len(_print_str), end="", flush=True)
                print(f"\r{_print_str}", end="", flush=True)

                g1_repr     = self.graph_representations_csr[g1_id]
                g1_class    = self.graph_classes_vec[g1_id]
                same_class_sample = g0_class == g1_class
           
                repr_diff: np.array = np.absolute(g0_repr - g1_repr).toarray()[0]
                # Compute the weighted difference vector.
                weighted_repr_diff_vec: np.array = np.multiply(repr_diff, edge_weights)
                
                # Track the individual sample movement:                
                dist_old: float = np.sum(weighted_repr_diff_vec)
                # Compute the indices of the weights that shall be updated.
                candidate_ids: np.array = self.get_candidate_indices(weighted_repr_diff_vec, self.hp_single_layer)

                if dist_old != 0.0 and len(candidate_ids) > 0:
                    # Compute the weight delta arising from these two graph representations. Do not apply it yet!
                    weight_delta_for_cand = self.update_edge_weights(candidate_ids, same_class_sample, edge_weights, same_diff_class_imba_tuple)
                    # Accumulate the weight deltas for every edge weight. 
                    # Since the same edge may be updated many times, after one epoch the mean of all updates will be applied.                    
                    for i, w_delta in enumerate(weight_delta_for_cand):
                        weight_delta_sum_ctrs[candidate_ids[i]][0] += w_delta
                        weight_delta_sum_ctrs[candidate_ids[i]][1] += 1

                    if self.print_sme_estimate:
                        self.timer.stop_timer(c.RT_Epoch, append=True)
                        # Track the individual sample movement:
                        tmp_updated_weight = edge_weights.copy()
                        tmp_weight_delta_vecs: np.array = np.array([(float(0.0), int(0)) for _ in range(n_weights)])
                        for i, w_delta in enumerate(weight_delta_for_cand): 
                            tmp_weight_delta_vecs[candidate_ids[i]][0] += w_delta
                            tmp_weight_delta_vecs[candidate_ids[i]][1] += 1
                        for i, sum_ctr_tuple in enumerate(tmp_weight_delta_vecs):
                            if sum_ctr_tuple[1] > 0:
                                tmp_updated_weight[i] += self.hp_learning_rate * sum_ctr_tuple[0]
                                # If not an abslute update, average the updates.
                                if not self.hp_abs_pp_factor: tmp_updated_weight[i] /= sum_ctr_tuple[1]
                        dist_new: float = np.sum(np.multiply(repr_diff, tmp_updated_weight))

                        distance_diff: float = dist_old - dist_new
                        sample_improvement: bool = None
                        # If the samples are from different classes, their distances contribute negatively to the error.
                        # That is if their distance increases, the error decreases as desired.
                        # Thus there has been an improvement, if the distance difference is negative - that is the new distance is greater.
                        # Analog for the same class, desiring a decrease in distances, and thus a positive distance difference.
                        if same_class_sample:
                            self.local_sample_movement_error -= distance_diff
                            if distance_diff > 0:
                                sample_improvement = True
                            elif distance_diff < 0:
                                sample_improvement = False
                        else:
                            self.local_sample_movement_error += distance_diff
                            if distance_diff < 0:
                                sample_improvement = True
                            elif distance_diff > 0:
                                sample_improvement = False
                        if distance_diff == 0.0: sample_improvement = None
                        self.timer.start_timer(c.RT_Epoch)
                        
                    # Update the epoch statistic.
                    set_of_touched_weights.update(tuple(candidate_ids))
                    if same_class_sample:
                        nr_pull += 1
                        if self.print_sme_estimate: same_cl_dist_sum_old += dist_old
                    else: 
                        nr_push += 1
                        if self.print_sme_estimate: diff_cl_dist_sum_old += dist_old                        
                    
                    if self.print_sme_estimate:
                        if sample_improvement is None: nr_same_sme += 1
                        elif sample_improvement: nr_improvements += 1
                        else: nr_degradations += 1
                        print(f"\t#I/#D/#S: {nr_improvements}/{nr_degradations}/{nr_same_sme} ", end="", flush=True)

                pair_ctr += 1
        if self.print_sme_estimate: self.local_sample_movement_errors.append(self.local_sample_movement_error)
        
        # w' = w + mean(\nu \Delta w)        
        epoch_weight_delta = compute_epoch_weight_delta(weight_delta_sum_ctrs)
        edge_weights += self.hp_learning_rate * epoch_weight_delta
        
        # Clip the weights to either [0, inf] or [0, 2].
        ceiling = 2.0 if self.hp_weight_ceiling else np.inf
        edge_weights = np.clip(edge_weights, a_min=0.0, a_max = ceiling)
        self.timer.stop_timer(c.RT_Epoch, append=True)

        if self.print_sme_estimate:
            # Evaluate the updated sample movement:
            for i0, g0_id in enumerate(batch_ids):
                g0_repr     = self.graph_representations_csr[g0_id]
                g0_class    = self.graph_classes_vec[g0_id]            
                for g1_id in batch_ids[i0+1:]:                
                    g1_repr     = self.graph_representations_csr[g1_id]
                    g1_class    = self.graph_classes_vec[g1_id]
                    repr_diff: np.array = np.absolute(g0_repr - g1_repr).toarray()[0]                
                    weighted_repr_diff_vec: np.array = np.multiply(repr_diff, edge_weights)                
                    if g0_class == g1_class: same_cl_dist_sum_new += np.sum(weighted_repr_diff_vec)                        
                    else:                    diff_cl_dist_sum_new += np.sum(weighted_repr_diff_vec)

            sample_movement_error_old: float = round(same_cl_dist_sum_old - diff_cl_dist_sum_old, 2)
            sample_movement_error_new: float = round(same_cl_dist_sum_new - diff_cl_dist_sum_new, 2)  
            batch_improvement: bool = None 
            if sample_movement_error_old > sample_movement_error_new: batch_improvement = True
            if sample_movement_error_old < sample_movement_error_new: batch_improvement = True

            movement_percent: float = round((1-(sample_movement_error_new/sample_movement_error_old))*100, 2)
            g_or_l: str = '<' if not batch_improvement else '>'
            print(f"\tBatchSME{'Degrad.' if not batch_improvement else 'Improv.'}: {movement_percent}%", end="")

        self._increment_ew_epoch_ctr()
        LOG.debug(f"Iteration {self._latest_edge_weight_iteration()}: #Push = {nr_push}, #Pull = {nr_pull}, #WUpdates: {len(set_of_touched_weights)}")
        return edge_weights.astype(np.float32)
    
class LeafWeights_Learner(EdgeWeightLearner):

    def __init__(self, wllt: WLLT, epochs: int = 1):
        super(LeafWeights_Learner, self).__init__(wllt=wllt)
        self.training_epochs: int = epochs

    def _improve_edge_weights(self) -> np.array:
        self.run_training_epochs(self.epochs)
    
    def run_training_epochs(nr_iterations: int):
        pass


def main(dataset_name: str = "MUTAG", wl_depth: int = 5, nr_epochs: int = 10, lr: float = 1.0, bs: int = 0.05, l_grad = None, single_layer: int = None, f_pull: Union[float, List[float]] = 0.4, f_push: Union[float, List[float]] = 0.4, abs_update: bool = False, heaviest_earth_thld: float = 'all', weight_ceiling: bool = True, print_sme_estimate: bool = False, cont_exist_learner_dir: str = None, shuffle_class_vec: bool = False, description: str = '', dir_out_suffix: str='') -> str:
    """
    Execute the learning process of the Default Learner. 
    This includes loading a WLLT from file, initializing a new or existing Default Learner,
    setting up its parameters and executing a run.
    The method returns the output directory of the Learner.
    """
    # Load the WLLT.
    wllt = get_WLLT_from_dataset_name(dataset_name, wl_depth)
    if wllt is None: return None
    # Initialize the edge weight learner implementation.
    learner = Default_Learner(wllt, nr_epochs=nr_epochs, print_sme_estimate=print_sme_estimate, continue_existing_learner_dir=cont_exist_learner_dir, dir_out_suffix=dir_out_suffix)
    
    if shuffle_class_vec: 
        perc_class_vec, perc_classes = learner.randomize_class_vec()
        description += f" {round(perc_class_vec, 2)}%-same class vec & {round(perc_classes, 2)}%-same classes"
    # Set the hyperparameters of this learner.
    learner.update_hyper_params(
        wl_depth=wllt.get_wl_iteration(), 
        lr=lr, 
        bs=bs, 
        l_grad=l_grad,
        single_layer=single_layer,
        f_pull=f_pull, 
        f_push=f_push, 
        heaviest_earth_thld=heaviest_earth_thld,
        abs_update=abs_update,
        weight_ceiling=weight_ceiling,
        description=f"{learner._completed_epochs + 1} - {learner._completed_epochs + learner.nr_epochs + 1}{' - ' + description if description != '' else ''}"
    )
    # Run the learner.
    learner.fit()
    
    out_path = learner.get_output_path()
    print(out_path)
    # Delete classes explicitly:
    del learner; del wllt
    LOG.info("WLLT Learner terminated.")
    return out_path

### Experiments ###

def experiment1_wl_depths_nr_epochs_grid_search(dataset_name: str = "MUTAG", wl_depths: List[int] = [4], bs:float = 0.05, nr_epochs: List[int] = 500, lr: float = 1.0, pp_f: float = 0.2):
    print(f"Starting grid search for: Dataset {dataset_name}. WL-depths: {wl_depths}. Nr-epochs: {nr_epochs}")
    for d in wl_depths:
        print(f"Ex1: wl_depth = {d},\tnr_epochs = {nr_epochs}")
        main(dataset_name, wl_depth=d, lr=lr, nr_epochs=nr_epochs, bs=bs, f_pull=pp_f, f_push=pp_f, dir_out_suffix="Exp1", description="Exp1")

def experiment2_pp_factors_grid_search(dataset_name = "MUTAG", wl_depth: int = 4, nr_epochs: int = 200, bs: int = 0.05, lr: float=1.0, single_layer: int = None, abs_update: bool = False, heaviest_earth_thld: float = 0.6):
    pp_factors_up   = np.array([0.1, 0.4])
    print(f"Starting grid search for: Dataset {dataset_name}. PP-factors: {pp_factors_up}")
    
    # Run tests with equal push pull:
    for pp_factor in pp_factors_up:
        print(f"Ex2: pp_factor = {pp_factor}")
        main(dataset_name, wl_depth=wl_depth, nr_epochs=nr_epochs, bs=bs, lr=lr, f_pull=pp_factor, f_push=pp_factor, single_layer=single_layer, heaviest_earth_thld=heaviest_earth_thld, abs_update=abs_update, dir_out_suffix="Exp2", description="Exp2")

    # Run tests with different push pull:
    pull_vs_push_down = [(pp_factors_up[0], pp_factors_up[-1]), (pp_factors_up[-1], pp_factors_up[0])]
    for f_push, f_pull in pull_vs_push_down:
        print(f"Ex2: f_push = {f_push},\tf_pull = {f_pull}")
        main(dataset_name, wl_depth=wl_depth, nr_epochs=nr_epochs, bs=bs, lr=lr, f_pull=f_pull, f_push=f_push, abs_update=abs_update, dir_out_suffix="Exp2", description="Exp2")

def experiment3_first_pull_than_push(dataset_name: str = "MUTAG", wl_depth: int = 4, nr_epochs: int = 200, bs: float = 0.05, f_pushs: List[float] = [0.05, 0.3], f_pulls: List[float] = [0.3, 0.05]) -> None:
    # Run learner with first configs.
    print(f"Ex3: First run: f_push={f_pushs[0]}, f_pull={f_pulls[0]}")
    learner_path: str   = main(dataset_name=dataset_name, wl_depth=wl_depth, nr_epochs=nr_epochs, bs=bs, f_push=f_pushs[0], f_pull=f_pulls[0], dir_out_suffix="Exp3", description="Exp3")
    if learner_path is None: return None
    learner_dir: str    = learner_path.split('/')[-1]
    # Continue to run same learner with second configs.
    print(f"Ex3: Second run: f_push={f_pushs[1]}, f_pull={f_pulls[1]}")
    main(              dataset_name=dataset_name, wl_depth=wl_depth, nr_epochs=nr_epochs, bs=bs, f_push=f_pushs[1], f_pull=f_pulls[1], cont_exist_learner_dir=learner_dir, description="Exp3_cont")

def experiment4_randomized_class_vec(dataset_name: str = "MUTAG", wl_depth: int = 4, nr_epochs: int = 200, bs: float = 0.05, p_factor: float = 0.2) -> None:
    # Run a learner with an altered class vector.
    print(f"Ex4: Artificial run - false class vec")
    main(dataset_name=dataset_name, wl_depth=wl_depth, nr_epochs=nr_epochs, bs=bs, f_push=p_factor, f_pull=p_factor, shuffle_class_vec=True, dir_out_suffix="Exp4", description="Exp4")
    # Run a learner with the same config, but the original class vector.
    print(f"Ex4: Compare run - normal class vec")
    main(dataset_name=dataset_name, wl_depth=wl_depth, nr_epochs=nr_epochs, bs=bs, f_push=p_factor, f_pull=p_factor, dir_out_suffix="Exp4", description=f"Exp4")

def run_grid_searches(dataset_name: str = "MUTAG", bs: float = 0.05):
    experiment1_wl_depths_nr_epochs_grid_search(dataset_name, bs=bs)
    experiment2_pp_factors_grid_search(dataset_name, bs=bs)
    experiment3_first_pull_than_push(dataset_name, bs=bs)
    experiment4_randomized_class_vec(dataset_name, bs=bs)    

def parse_terminal_args() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
        default=['MUTAG'],
        dest='dataset_names',
        help='Provide TU Dortmund dataset names.',
        type=str,
        nargs='+')
    parser.add_argument('-l',
        default=None,
        dest='wllt_layers',
        help='Number of WLLT layers that shall be used.',
        type=int,
        nargs='+')
    parser.add_argument('-e',
        default=None,
        dest='nr_epochs',
        help='Number of learning epochs.',
        type=int)
    args = parser.parse_args(sys.argv[1:])
        
    if args.wllt_layers != None and args.nr_epochs != None:
        for dataset_name in args.dataset_names:
            for wl_depth in args.wllt_layers:
                main(dataset_name=dataset_name, wl_depth=wl_depth, nr_epochs=args.nr_epochs)
    else:
        for dataset_name in args.dataset_names:
            run_grid_searches(dataset_name=dataset_name)

def default_runs(dataset_name: str) -> None:
    main(dataset_name=dataset_name, wl_depth=4, nr_epochs=100,  bs=.05, dir_out_suffix="_normal")
    main(dataset_name=dataset_name, wl_depth=4, nr_epochs=100,  bs=.05, single_layer=2, dir_out_suffix="_Layer2")
    main(dataset_name=dataset_name, wl_depth=4, nr_epochs=100,  bs=.05, single_layer=4, dir_out_suffix="_Layer4")
    main(dataset_name=dataset_name, wl_depth=4, nr_epochs=100,  bs=.05, f_pull=0.1, f_push=0.9, dir_out_suffix="_HighPush")
    main(dataset_name=dataset_name, wl_depth=4, nr_epochs=100,  bs=.05, f_pull=0.9, f_push=0.1, dir_out_suffix="_HighPull")
    main(dataset_name=dataset_name, wl_depth=4, nr_epochs=100,  bs=.05, heaviest_earth_thld=0.1, abs_update=True, dir_out_suffix="_abshe0.5")    
    main(dataset_name=dataset_name, wl_depth=4, nr_epochs=1000, bs=.05, dir_out_suffix="_e1000")
    

if __name__ == "__main__":
    # Run in terminal as: python3 x3_wllt_edge_weight_learner.py -d MUTAG -i 3 -e 100
    parse_terminal_args()