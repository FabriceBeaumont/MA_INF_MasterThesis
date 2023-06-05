import numpy as np
from os import getcwd
from typing import List, Dict, Tuple
from os import listdir

def get_datafiles_dir() -> str:
    cwd_arr = np.array(getcwd().split('/'))
    x = (cwd_arr == 'WLLTProgram')
    i = np.where(x)[0][0]
    in_dir = "/".join(cwd_arr[:i])
    return in_dir

def get_program_dir() -> str:
    dir = getcwd()
    cwd_arr = np.array(getcwd().split('/'))
    x = (cwd_arr == 'WLLTMetricLearner') # TODO: December: why does this not work with a string flag?
    if not all(x == False):
        i = np.where(x)[0][0]
        dir = "/".join(cwd_arr[:i])
    return dir

def get_epoch_from_fn(file_name: str, prefix: str, file_type: str) -> int:
    return int(file_name.replace(prefix, "").replace(file_type, ""))

def get_file_name_versions(dir_in: str, file_predix: str, file_type: str = ".npy") -> Dict[int, str]:    
    epoch_to_ew_filename_dict: Dict[int, str] = dict()

    for file in listdir(dir_in):        
        if file.startswith(file_predix):
            e = get_epoch_from_fn(file_name=file, prefix=file_predix, file_type=file_type)            
            epoch_to_ew_filename_dict[e] = f"{dir_in}/{file}"
    
    if len(epoch_to_ew_filename_dict) == 0:
        return None
    return epoch_to_ew_filename_dict

# Constant values
DUMMY_INDEX: int                = -1
DUMMY_LABEL: int                = DUMMY_INDEX
CSV_DELIMITER: str              = '\t'
N_DECIMALS: int                 = 3

# Directory names:
DN_DATAFILES: str               = "Datafiles"
DN_WLLT: str                    = "WLLT"
DN_WLLT_LEARNER: str            = "WLLTLearner"

# File names:
x = get_program_dir()
CSV_MA_EVALUATION_PATH: str     = f"{x}/MA_Evaluations.csv"
CSV_LEARNER_EVAL: str           = "LearnerEvaluations.csv"
CSV_EW_LEARNER_CONFIG: str      = "LearnerConfig.csv"

FN_ADJ_LISTS: str               = "adj_lists.npy"
FN_VERTEX_LABELS: str           = "original_vertex_labels.npy"
FN_GRAPH_VERTICES: str          = "graph_vertices.npy"
FN_GRAPH_CLASSES: str           = "graph_classes.npy"

FN_META_GRAPH_REPR: str         = "WLLT_META_graph_representations.npz"
FN_META_INFO: str               = "WLLT_META_info.txt"
FN_META_LAYER_STARTS: str       = "WLLT_META_layer_starts.npy"
FN_META_PARENT_LIST: str        = "WLLT_META_parent_list.npy"
FN_META_EDGE_WEIGHTS: str       = "WLLT_META_edge_weights.npy"
FN_META_MEAN_WL_CHANGE: str     = "WLLT_META_mean_wl_change.npy"
FN_META_WL_MAP: str             = "WLLT_META_vertex_label_map.npy"

FN_PREFIX_VERTEX_LABELS_D: str  = "WLLT_vertex_labels_d"

FN_EDGE_WEIGHTS_E: str          = "WLLT_learned_edge_weights_e"
FN_DIST_MAT_E: str              = "EVAL_dist_mat_e"
FN_DIST_IDS_E: str              = "EVAL_dist_ids_e"
FN_DIST_MAT_I: str              = "WLLT_dist_mat"
FN_DIST_MAT_IDS_E: str          = "WLLT_dist_mat_ids"

# Evaluation columns:
C_MEAN_WEIGHT_PER_LAYER: str    = "MeanWpL"
C_MAX_WEIGHT_PER_LAYER: str     = "MaxWpL"
C_MIN_WEIGHT_PER_LAYER: str     = "MinWpL"
C_SUM_WEIGHT_PER_LAYER: str     = "SumWpL"
C_TOTAL_WEIGHT_SUM: str         = "Total WLLT weight sum"

C_SCORE_SILHOUETTE: str         = r"Score Silhouette\N(Near 1, Separation)"
C_SCORE_DAVIES_BOULDIN: str     = r"Score Davies Bouldin\N(Decrease, FartherApart, LessDispersed)"
C_SCORE_CALINSKI_HARABASZ: str  = r"Score Calinski Harabasz\N(Increase, Dense, Separation)"

C_SVM_ACC: str                  = "SVM Accuracy"
C_SVM_STD_DEV: str              = "SVM Standard Deviation"

C_CL_DIST_INTRA_MEAN_Cl: str    = "Mean Intra Cl Dist Cl"
C_CL_DIST_INTRA_MAX_Cl: str     = "Max Intra Cl Dist Cl"
C_CL_DIST_INTRA_MIN_Cl: str     = "Min Intra Cl Dist Cl"
C_CL_DIST_INTRA_SUM: str        = "Sum Intra Cl Dist"

C_CL_DIST_INTER_MEAN_Cls: str   = "Mean Inter Cl Dist Cls"
C_CL_DIST_INTER_MAX_Cls: str    = "Max Inter Cl Dist Cls"
C_CL_DIST_INTER_MIN_Cls: str    = "Min Inter Cl Dist Cls"
C_CL_DIST_INTER_SUM: str        = "Sum Inter Cl Dist"

C_CL_SAMPLE_MOVEMENT_ERROR: str = r"Sample Movement Error\N(Decrease)"

C_LEARNER_EPOCH: str            = "Learner Epoch"

# Hyper Params
HP_WL_DEPTH: str                = "WLLT Depth"                       # 0     The number of WLLT layers, that are considered.
HP_UPDATE_FREQ: str             = "Update Frequency"                 # 1     # E.g. every datapoint, batch, iteration, ...
HP_UPDATE_SCOPE: str            = "Update Scope"                     # 2     # E.g. all layers, only leaves, weighted layers, ...
HP_UPDATE_INTENSITY: str        = "Update Intensity"                 # 3     # E.g. fixed increments, pulling clusters closer and or pushing, ...                       
HP_LEARNING_RATE: str           = "Learning Rate"                    # 4
HP_BATCH_SIZE: str              = "Batch Size"                       # 5
HP_LAYER_GRADIENT: str          = "Layer Gradient"                   # 6
HP_SINGLE_LAYER: str            = "Train Weights For Only Layer"     # 7    
HP_CL_PULL_FACTOR: str          = "Cluster Pull Factor"              # 8     # The percentage value to which the selected edge weights will be diminished. This percentage will be weighted with the number of graphs used in the current learning iteration.
HP_CL_PUSH_FACTOR: str          = "Cluster Push Factor"              # 9     # The percentage value to which the selected edge weights will be increased. This percentage will be weighted with the number of graphs used in the current learning iteration.
HP_ABSOLUTE_FACTORS: str        = "Pull/Push Factors Are Absolute"   # 10
HP_HEAVIEST_EARTH_LIM: str      = "Heaviest Earth Threshold"         # 11    # Number or percentage of how many of the eaviest differences in the histograms between two graphs will be evaluated and used as a base for updating the respective weights. The value is logically ceiled by the number of edges in the tree. If zero, all weights will be updated.
HP_WEIGHT_CEILING: str          = "Weights in [0, 2]"                # 12    # If set to true, all individual weight can have value of at most '2.0'.
HP_DESCRIPTION: str             = "Description"                      # 16

# Runtimes
RT_Epoch: str                   = "RT.Epoch"

# Graph kernels
K_NOG: str                      = "NoG"                              # \cite{2019_Schulz_CONF}