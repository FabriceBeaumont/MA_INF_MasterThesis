from typing import List, Dict, Tuple, Any, Union
from collections import Counter
import numpy as np
from os.path import exists
from os import mkdir, makedirs
from abc import ABC, abstractmethod
# For the parsing of console input:
import sys, getopt, argparse

from scipy.sparse import coo_matrix, save_npz, load_npz

import matplotlib.pyplot as plt
import networkx as nx

# Own files:
from my_utils.decorators import get_logger
from my_utils.functions import is_bijection
import my_consts as c

# GLOBAL DEBUGGING LOGGER for this script
LOG = get_logger(filename="Loggers/x2_wllt_constructor.txt", create_file=True)

HISTOGRAM_PRECISION = 3
MEAN_WL_CHANGE_WARN = 5
WLLT_META_DELIMITER = "\t"

############################################################################## 
##### WLLT ###################################################################
##############################################################################

class WLLT():
    """    
    Every WLLT consists of the following files:
        WLLT_META_info.txt                          containing paths to other files, depth.
        WLLT_META_layer_starts.npy                  containing the highest WLLT-vertex-index for every layer.
        WLLT_META_graphs_wllt_representation.npz    containing an 'scipy.sparse.coo_matrix' of the (normalized) frequencies of all wl-labels in every graph.
        WLLT_META_edge_weights.npy                  containing an array of edge weights at position v for every edge (w,v). Has #wl-labels many lines.
        WLLT_META_mean_wl_change.npy                containing for every iteration the mean of how many WL-labels have changed compared to the last iteration (that is if they are not bijective).

        WLLT_path_lists_d<depth>.npy                containing the wllt as list of paths of length <depth>. Only the one with the highest d is needed. Has #layer-<depth> many lines.
        WLLT_vertex_labels_d<depth>.npy             containing an array of wl-labels at depth <depth> for every vertex in the whole database.
    """
    
    def __init__(self, ask_for_early_stopp: bool = True):
        self.dataset_name: str  = ""
        # This 'wl_iteration' counter indicates the last completed iteration. There should be files (WL-vertex-labels) for this value.
        # Since the original vertex labels are considered as zeroth WL-vertex-labels, this iteration has been completed at initialization.
        self._wl_iteration: int = 0
        # This 'n' counter will also be used to define the next wl-label in the hashing process.
        self.n: int             = 0
        self.dir_in: str        = ""
        self.dir_out: str       = ""
        self.ask_for_early_stopp: bool = ask_for_early_stopp

    def __repr__(self) -> str:
        str_representation: str = f"WLLT - wl-iteration: {self.get_wl_iteration()}, n: {self.n}. File names:"
        str_representation += f"\n\tOutput dir.:        {self.dir_out}"
        str_representation += f"\n\tAdjacency lists:    {self.get_adj_lists_file_name()}"
        str_representation += f"\n\tVertex ids:         {self.get_graph_vertices_file_name()}"
        str_representation += f"\n\tGraph classes:      {self.get_graph_classes_file_name()}"
        str_representation += "\n"
        str_representation += f"\n\tMeta:               {self.get_meta_file_name()}"
        str_representation += f"\n\tParent list:        {self.get_parent_list_file_name()}"
        str_representation += f"\n\tVertex labels:      {self.get_vertex_labels_file_name()}"
        str_representation += f"\n\tVertex label map:   {self.get_vertex_label_map_file_name()}"        
        str_representation += f"\n\tGraph repr.:        {self.get_graph_representations_file_name()}"
        str_representation += f"\n\tEdge weights:       {self.get_edge_weights_file_name()}"
        return str_representation

    def get_nr_wl_labels(self) -> int:
        return self.n

    def increment_nr_wl_labels(self) -> None:
        self.n += 1

    ### File name - Getters ###

    def get_meta_file_name(self) -> str:
        return f"{self.dir_out}/{c.FN_META_INFO}"

    def get_adj_lists_file_name(self) -> str:
        return f"{self.dir_in}/{c.FN_ADJ_LISTS}"
    
    def get_graph_vertices_file_name(self) -> str:
        return f"{self.dir_in}/{c.FN_GRAPH_VERTICES}"
    
    def get_graph_classes_file_name(self) -> str:
        return f"{self.dir_in}/{c.FN_GRAPH_CLASSES}"

    def get_output_dir(self) -> str:
        return self.dir_out
    
    def get_input_dir(self) -> str:
        return self.dir_in

    # The 'wl_iteration' counter is sensitive, since it is used to index files.
    # Thus all operations to manipulate it are encapsulated in the folowing five methods.
    def _latest_wl_iteration(self) -> int:
        """ Returns the highest iteration that has been completed. There should be WL-vertex-labels stored for this iteration. """
        return self._wl_iteration
    
    def _next_wl_iteration(self) -> int:
        """ Returns the next iteration that has to be completed. There should be no WL-vertex-labels stored for this iteration yet. """
        return self._wl_iteration + 1

    def _increment_wl_iteration_ctr(self) -> None:
        self._wl_iteration += 1
    
    def _set_wl_iteration_ctr(self, value: int) -> None:
        self._wl_iteration = value

    def get_wl_iteration(self) -> int:
        return self._wl_iteration

    def get_layer_starts_file_name(self) -> str:
        return f"{self.dir_out}/{c.FN_META_LAYER_STARTS}"

    def get_parent_list_file_name(self) -> str:
        return f"{self.dir_out}/{c.FN_META_PARENT_LIST}"

    def get_vertex_labels_file_name(self, depth: int = None) -> str:
        """ If no depth is passed, the name of the most recent created file is returned. """
        if depth is None: depth = self._latest_wl_iteration()
        if depth == -1: depth = 0        
        return f"{self.dir_out}/{c.FN_PREFIX_VERTEX_LABELS_D}{depth}.npy"

    def get_vertex_label_map_file_name(self) -> str:
        return f"{self.dir_out}/{c.FN_META_WL_MAP}"

    def get_graph_representations_file_name(self) -> str:
        return f"{self.dir_out}/{c.FN_META_GRAPH_REPR}"

    def get_edge_weights_file_name(self) -> str:        
        return f"{self.dir_out}/{c.FN_META_EDGE_WEIGHTS}"

    def get_mean_wl_change_file_name(self) -> str:
        return f"{self.dir_out}/{c.FN_META_MEAN_WL_CHANGE}.npy"

    def get_all_file_paths(self) -> List[str]:        
        l = [self.dir_out,
            self.get_adj_lists_file_name(),
            self.get_graph_vertices_file_name(),
            self.get_graph_classes_file_name(), 
            self.get_meta_file_name(), #
            self.get_layer_starts_file_name(),
            self.get_parent_list_file_name(),
            self.get_vertex_labels_file_name(),
            self.get_vertex_label_map_file_name(),
            self.get_graph_representations_file_name(),
            self.get_edge_weights_file_name(),
            self.get_mean_wl_change_file_name()
        ]
        return l

    ### Data from file - Getters ###
    
    def get_data_up_to_iter_from_file(self, file_name: str) -> np.array:
        arr = np.load(file_name, allow_pickle=True)
        # Get only the layer starts, up to the last iteration. Since the layer-end of the zeroth iteration is stored in the 
        # zeroth position of the array, the pointer has to be incremented by one.
        return arr[:self._latest_wl_iteration() + 1]

    def get_layer_starts_from_file(self) -> np.array:
        layer_starts: np.array = self.get_data_up_to_iter_from_file(self.get_layer_starts_file_name())
        return layer_starts.astype(np.int32)

    def get_highest_leaf_from_file(self) -> int:
        """ 
        Returns the highest leaf (WLLT vertex) that is stored to file. Notice that it consideres all layer starts up to the latest wl_iteration,
        and returns only the last layer start - which is the highest leaf.
        Thus if this values is reduced, the WLLT can be considered to be smaller, than it has been computed already.
        """
        layer_starts = np.load(self.get_layer_starts_file_name(), allow_pickle=True)
        return int(layer_starts[self._latest_wl_iteration()])

    def get_wl_labels_layer_wise(self) -> List[List[int]]:
        """
        This method returns a list of lists of integers such that list 'i' contains
        all wl-labels that are used at depth 'i' in the WLLT.
        Since the wl-labels are in range '[0, n]', this can be done by only storing the "outer most right"
        wl-label (layer start) to get the ranges of wl-labels used per layer.
        """
        old_layer_starts = self.get_layer_starts_from_file().astype(int)
        layer_starts = np.append([0], old_layer_starts)
        
        wl_labels = []
        for i in range(len(layer_starts) - 1):
            wl_labels.append(list(range(layer_starts[i], layer_starts[i+1])))

        return wl_labels

    def get_parent_list_from_file(self) -> np.array:
        """ 
        Returns the parent list stored to file. Notice that it consideres all parents up to the highest index in the layer starts vector.
        Since the read-in of the layer starts vector depends on the latest wl_iteration, it is possible to consider the WLLT to be smaller, 
        than it has been computed already. That is to return not all parents that are stored, but all up to a considered layer in the WLLT.
        """
        highest_leaf = self.get_highest_leaf_from_file()
        parent_list = np.load(self.get_parent_list_file_name()).astype(int)
        return parent_list[:highest_leaf]

    def get_vertex_labels_from_file(self, depth: int = None, with_tailing_dummy_label: bool = True) -> np.array:        
        vertex_labels = np.load(self.get_vertex_labels_file_name(depth)).astype(int)
        
        if with_tailing_dummy_label:
            vertex_labels = np.append(vertex_labels, np.array([c.DUMMY_LABEL], dtype=int))
        return vertex_labels

    def get_vertex_label_map_from_file(self, as_dict=False) -> Union[np.array, Dict[int, Tuple[int, List[int]]]]:
        vertex_label_map = np.load(self.get_vertex_label_map_file_name(), allow_pickle=True)

        if not as_dict:
            return vertex_label_map
        else:
            vertex_label_dict = dict(zip(range(len(vertex_label_map)), vertex_label_map))
            return vertex_label_dict

    def get_original_vertex_labels_from_file(self) -> np.array:
        original_vertex_labels = np.load(f"{self.dir_in}/{c.FN_VERTEX_LABELS}")

        # Remove tailing dummy labels.
        original_vertex_labels = np.append(original_vertex_labels, np.array([c.DUMMY_LABEL], dtype=int))
        return original_vertex_labels

    def get_graph_repr_from_file(self, nr_features: int = None):
        """
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix
        This method returns a 'scipy.sparse.coo_matrix' which contains the (normalized) frequencies
        of ALL wl-labels (in the whole wllt) that can be applied in their respective iteration to the 
        labels of a graph.
        The sparse vectors of these wl-label frequencies are the rows of the matrix. Thus row i stores
        this vector for graph i.
        The columns correspond to the wl-labels in the wllt. Thus the (normalized) frequency
        of wl-label 'j' in graph 'i' is stored as entry (i, j).
        """        
        M = load_npz(self.get_graph_representations_file_name()).tocoo()    # -> scipy.sparse.csr.csr_matrix
        data, row, col = M.data, M.row, M.col

        if nr_features is None:
            # Notice that the WLLT may have been initialized from the META file, 
            # using a lesser wl-iteration depth than was computed earlier.
            # Thus omit the other columns, if necessary.
            layer_start = self.get_layer_starts_from_file()[self.get_wl_iteration()]
            nr_features = layer_start

        # All values must be lower than 'layer_start', since the indixes of the WLLT edges
        # indicate their parent vertex. Thus all edges that are in the subtree have a name lower than 'layer_start'.
        # On the other hand using 'range(layer_start)' as limiting index works perfectly finde.
        selected_feature_ids = np.where(col < nr_features)
        
        data = data[selected_feature_ids].astype(np.float32)
        row  = row[selected_feature_ids]
        col  = col[selected_feature_ids]
        if len(data) == 0:
            trimmed_graph_repr_coo = coo_matrix((data, (row, col)), shape=(0, 0)) 
        else:
            trimmed_graph_repr_coo = coo_matrix((data, (row, col)))
                
        return trimmed_graph_repr_coo

    def get_edge_weights_from_file(self) -> np.array:
        layer_start = self.get_layer_starts_from_file()[self.get_wl_iteration()]
        
        all_weights = np.load(self.get_edge_weights_file_name())
        return all_weights[:layer_start]

    def get_adj_lists_from_file(self) -> np.array:
        return np.load(self.get_adj_lists_file_name()).astype(int)

    def get_graph_vertices_from_file(self) -> np.array:
        return np.load(self.get_graph_vertices_file_name()).astype(int)

    def get_graph_classes_vec_from_file(self) -> np.array:
        return np.load(self.get_graph_classes_file_name()).astype(int)

    def get_graph_classes_from_file(self) -> np.array:
        return np.unique(self.get_graph_classes_vec_from_file())

    def get_nr_graphs_from_class_file(self) -> int:
        return self.get_graph_classes_vec_from_file().shape[0]

    def get_highest_layer_index_from_file(self) -> int:
        """ Let 'L' be the path from the root to a leaf, than the returned value is '|L|-1'. """
        return len(self.get_layer_starts_from_file())

    def get_graphs_vertices_wl_labels(self, depth: int = None, vertices_matrix: np.array = None) -> np.array:
        """
        Returns a matrix containing all graphs, with the wl-labels of iteration 'd'.
        By default 'd' is the highest constructed labeling.

        The output is a matrix such that row 'i' contains the 
        wl-labels in iteration 'd' of graph 'i'; and tailing DUMMY-LABELS.
        """
        if depth is None or depth >= self.get_wl_iteration(): 
            return None
        
        wl_labels = self.get_vertex_labels_from_file(depth, with_tailing_dummy_label=True)
        if vertices_matrix is None:
            vertices_matrix = self.get_graph_vertices_from_file()

        # Use the values in 'vertices_matrix' as in indices in 'wl_labels' to find out what their new value is.
        wl_label_matrix = wl_labels[vertices_matrix]
        return wl_label_matrix    
    
    def get_common_ancestor_of_same_layer_vertices(self, v: int, w: int, parent_list: np.array = None) -> int:
        """
        Simultaneously and iteratively compares the parents of the passed vertices.
        If the verticies do not have the same path length to their common ancestor, that is 
        if they are not on the same layer, this is not the right method for the task!
        Returns the leasr common ancestor, if found (if the vertices are on the same layer).
        """
        if parent_list is None:
            parent_list = self.get_parent_list_from_file()

        parent_v = parent_list[v]
        parent_w = parent_list[w]

        while parent_v != parent_w:
            parent_v = parent_list[parent_v]
            parent_w = parent_list[parent_w]

            # It is allowed that both parents are the only negative value, which is the root.
            # It cannot be, that just one parent is negative, since then the indices are not at the same layer!
            # If this case still appears, break since otherwise the while loop would traverses the tree multiple times and returns false information.
            if (parent_v * parent_w < 0): break

        if parent_v == parent_w:
            return parent_v
        else:
            LOG.error(f"Fct = 'get_common_ancestor({v}, {w})'! No common ancestor was found!")
            return None

    def get_path_to_root(self, v, parent_list: np.array = None) -> np.array:
        """
        Iterates through the np.array of parents until the root is reached. 
        Returns the list of visited tree vertices.
        """
        if parent_list is None:
            parent_list = self.get_parent_list_from_file()
        
        path = np.array([v], dtype=int)
        parent = parent_list[v]
        path = np.append(path, [parent])

        while parent != -1:
            parent = parent_list[parent]
            path = np.append(path, [parent])

        return path

    def get_all_paths_to_root(self, parent_list: np.array = None, depth: int = None, max_v_id: int = None) -> Dict[int, np.array]:
        """
        Iterates through all leaves and calls the 'get_path_to_root' function to construct a 
        parth to the root for all leaves.
        Returns a dictionarys mapping a leav index (key) to the path from it to the root (as np.array of size #wllt_depth).
        """
        layer_starts = self.get_layer_starts_from_file()

        if depth is None: 
            if max_v_id is None:
                depth = self._latest_wl_iteration()
            else:
                for d, v in enumerate(layer_starts):
                    if v >= max_v_id:
                        depth = d
                        break
        
        leaves_layer_start  = depth - 1
        leaves_layer_end    = depth
        leaves_layer_ids    = range(layer_starts[leaves_layer_start], layer_starts[leaves_layer_end])
        paths_dict: dict    = dict(zip(leaves_layer_ids, np.array([])))

        if parent_list is None: parent_list = self.get_parent_list_from_file()
        for leaf in leaves_layer_ids:
            tmp_path: np.array = self.get_path_to_root(leaf, parent_list)
            paths_dict[leaf] = tmp_path

        return paths_dict

    def get_unfolding_tree(self, label: int, label_map: Dict[int, Tuple[int, List[int]]] = None) -> Tuple[List, str]:        
        
        def unfold_parent_recursive(label: int, indent_nr: int = 0) -> Tuple[List[Tuple], str]:
            v_N_hash: Tuple[int, List[int]] = label_map.get(label)
            indent: str = '.' * indent_nr
            if v_N_hash is None:              return None, ''
            label_v, labels_N = v_N_hash[0], v_N_hash[1]
            if label_v      == c.DUMMY_LABEL:  return None, None
            if len(labels_N) == 0:             return None, None
            if labels_N[0]  == c.DUMMY_LABEL:  return None, None
            
            unfolded_v, unfolded_v_str = unfold_parent_recursive(label_v, indent_nr=indent_nr+1)
            unfolded_N = []
            unfolded_N_str = ''
            # Get the unfolding trees for all labels in the neighborhood - ONCE. No repetitions necessary.
            for n in np.unique(list(v_N_hash[1])):
                unfolded_n, n_str = unfold_parent_recursive(n, indent_nr=indent_nr+1)
                unfolded_N.append(unfolded_n)
                if n_str is not None:
                    unfolded_N_str += f"\n{indent}{n_str}" 
            
            title = f"{indent}{label}:"
            if len(title) < 7: title += "\t"

            ret_str = f"{title}\t{label_v}|{labels_N}"
            if unfolded_v_str is not None: ret_str += f"\n{indent}{unfolded_v_str}"
            ret_str += f"{unfolded_N_str}"
            return [label, (unfolded_v, unfolded_N)], ret_str

        if label_map is None: label_map: Dict[int, Tuple[int, List[int]]] = self.get_vertex_label_map_from_file(as_dict=True)
        unfolding_tree = [label]        

        u_tree, unfolding_tree_str = unfold_parent_recursive(label)
        unfolding_tree.append(u_tree)
        
        return unfolding_tree, unfolding_tree_str


    def get_common_ancestors_of_different_layer_vertices(self, v: int, w: int, parent_list: np.array = None) -> Tuple[int, List[int]]:
        """
        Constructs the complete paths from the input tree vertices 'v' and 'w' to the root
        and computes the intersection of these paths.
        Returns both the maximum value of the intersection (the wl-label/tree vertex furthest from the root),
        and the hole list of common ancestors (path from the least common ancestor to the root) - which is created as a byproduct.
        """
        if parent_list is None:
            parent_list = self.get_parent_list_from_file()

        path_v = self.get_path_to_root(v, parent_list)
        path_w = self.get_path_to_root(w, parent_list)

        common_elements = np.intersect1d(path_v, path_w, assume_uniqe=False)

        return common_elements.max(), common_elements

    def get_mean_max_nr_vertices(self) -> Tuple[int, int]:
        """
        Returns the mean and maximum number of vertices, by iterating over all 
        vertices from the graph-vertices-file and comparing the non-dummy-entries.
        """
        vertices_mat = self.get_graph_vertices_from_file()
        nr_graphs = vertices_mat.shape[0]
        max_nr_of_vertices: int = vertices_mat.shape[1]
        nr_vertices_sum = 0
        DUMMY_INDEX = -1

        for row in vertices_mat:
            # Count how many values are not the DUMMY_INDEX. That is how many vertices there are.
            tmp_nr_vertices = np.nonzero(row!=DUMMY_INDEX)[0].shape[0]            
            nr_vertices_sum += tmp_nr_vertices

        mean_nr_vertices = nr_vertices_sum // nr_graphs
        
        return mean_nr_vertices, max_nr_of_vertices

    def get_nr_of_graphs(self) -> List[int]:
        vertices_mat = self.get_graph_vertices_from_file()
        return vertices_mat.shape[0]

    ### File maintainance ###   
    
    def append_to_nparr(self, file_name: str, new_list: List, first_file_iteration: int = 0) -> None:
        stored_list = []
        if self.get_wl_iteration() > first_file_iteration:
            stored_list = self.get_data_up_to_iter_from_file(file_name)        
        
        stored_list = np.append(stored_list, new_list)
        self.write_np_to_file(file_name, stored_list)

    def append_to_vertex_map(self, new_list: List) -> None:
        stored_list = []
        if self.get_wl_iteration() >= 0:
            stored_list = self.get_vertex_label_map_from_file()        
        
        stored_list = np.vstack((stored_list, new_list))
        self.write_np_to_file(self.get_vertex_label_map_file_name(), stored_list)

    def update_graph_representation_coo_for_layers(self, iterations: List[int] = None) -> None:
        """ 
        This method updates the sparse matrix of graph representations.
        To initialize this matrix, simply do not pass an argument for the 'iterations'-List.
        In this case the matrix is initialized by empty lists.

        If the 'iterations'-List is not None, the graph representations are updated using all specified layers of the wllt.
        It is assumed that these layes (and thus the required files) already exist. 
        
        Thus this method will be executed after each batch of layer constructions.

        The update process reads in the already constructed matrix. It then loads the 
        wl-vertex labels for all iterations for all graphs and computes their representation.
        Since this representation only concernes a few occurring labels, this is stored in a sparse manner.

        - The data are the normalized frequencies of all wl-labels (for all graphs).
        - The column indices correspond to the wl-label.
        - The row indices correspond to the graph id.
        """
        def _construct_all_iterations_labels_matrix() -> np.array:
            """
            Stack all wl-labellings for every graph.
            The output is a matrix such that its 'i'-th row contains the wl-labels
            for every iteration of all vertices in graph 'i'.
            For an easy access, the maximum size of all graphs is returned as well. Notice that 
            all wl-labellings are padded with as many DUMMY_LABELS as needed to be as big as the biggest graph.
            """
            # Read the list of all graph vertices for every graph from file.
            vertices_matrix: np.array = self.get_graph_vertices_from_file()
            nr_of_graphs:       int = vertices_matrix.shape[0]
            max_nr_of_vertices: int = vertices_matrix.shape[1]            
            # Allocate space to store all wl-label for all vertices for every iteration (col), for every graph (row).            
            # Allocate space for one more iteration, since the zeroth iteration is contained as well.
            wl_label_matrices: np.array = np.empty((nr_of_graphs, max_nr_of_vertices * (self.get_wl_iteration() + 1)), dtype=float)

            # Iterate over all iterations and append the wl-labeling in this iteration for all graphs.
            for iteration in range(self.get_wl_iteration() + 1): 
                wl_labels = self.get_vertex_labels_from_file(iteration, with_tailing_dummy_label=True)
                # Map the wl-labels (in iteration 'iteration') of all vertices to the vertices of all graphs.
                wl_label_matrix = wl_labels[vertices_matrix]
                wl_label_matrices[:,max_nr_of_vertices*iteration:max_nr_of_vertices*(iteration+1)] = wl_label_matrix
                           
            return wl_label_matrices

        data, row, col = None, None, None                

        # If the 'iterations' list is not empty, assume that former layers and thus a graph representation does exists.
        # Read it from file in order to extend it.
        if iterations is None:
            data, row, col = np.array([]), np.array([]), np.array([])            
        else:
            M = self.get_graph_repr_from_file()            
            data, row, col = M.data, M.row, M.col

            # The graph representation is computed as a normalized wl-label histogram. Since the normalization with big graphs, 
            # and less frequent wl-labels will cause vanishing histogram entries, scale the normalization with the mean number of graphs.
            mean_nr_vertices, max_nr_of_vertices = self.get_mean_max_nr_vertices()
                        
            # In order to avoid two for-loops (one over the iterations and one over all graphs),
            # prepare the wl-label-data for all informations in a list.
            # That is for every graph the concatenated list of its wl-label for every vertex and iteration.
            wl_label_matrices: np.array = _construct_all_iterations_labels_matrix()

            # Iterate through all graphs and save the (normalized) frequencies of all wl-labels for every iteration that occured in them.                        
            for g_id, all_wl_labels in enumerate(wl_label_matrices):
                # Compute a histogram for the wl-labels in the graph.
                ctr = Counter(all_wl_labels)                
                # Rule out the DUMMY-LABEL. 
                # (Only) for the normalization it is crutial to know, how many vertixes are in the current graph.
                nr_of_vertices_in_g: int = max_nr_of_vertices
                ids_of_padding = np.where(all_wl_labels[:max_nr_of_vertices]==c.DUMMY_LABEL)[0]
                # If there exists a DUMMY_LABEL in the  wl-label string, its position indicates the size of the graph.
                # Otherwise the graph is one of the biggest with size 'max_nr_of_vertices'.
                if len(ids_of_padding) > 0: nr_of_vertices_in_g = ids_of_padding[0]
                # The histogram shall be normalized to account for different graph sizes.
                # To prevent vanishing values, this is scaled with the mean graph size.
                normalization_factor = mean_nr_vertices / nr_of_vertices_in_g

                # Now iterate over all wl-labels and add the normalized frequencies for this graph (its representation) to the sparse matrix.                
                # Notice that the order of the entries does not matter at all in the initialization of a sparse matrix.                
                # Only insert normalized frequencies that are not zero - these are the ones stored in the Counter. 
                # Since all other frequencies will be understood as zero by default interpretation of the sparse matrix.
                
                # Add respective entries in 'row', 'col', 'data' to sparsely store the wl-labels frequencies.
                # But not the DUMMY_LABEL.
                if ctr[c.DUMMY_LABEL] > 0: ctr.pop(c.DUMMY_LABEL)                
                normalized_frequencies = normalization_factor * np.array(list(ctr.values()))
                np.round_(normalized_frequencies, HISTOGRAM_PRECISION)
                # There is a row for every graph. Notive that 'len(ctr)' many entries will be added to the data, one for each occuring label in the graph.
                row     = np.append(row, [g_id]*len(ctr))
                # There is a column for every wl-label in all iterations in the whole dataset. The index equals the value of the wl-label.
                col     = np.append(col, list(ctr.keys()))
                data    = np.append(data, normalized_frequencies)
            
        # Save the graph representation matrix.
        nr_graphs: int = self.get_nr_graphs_from_class_file()
        # Assemble the coo matrix using the data.
        graph_repr_coo = coo_matrix((data, (row, col)), shape=(nr_graphs, self.get_nr_wl_labels()))
        # Save the matrix to file.
        self.write_scipy_sparse_to_file(self.get_graph_representations_file_name(), graph_repr_coo)

    ### File writers ###

    def write_np_to_file(self, file_name: str, data: Any) -> None:
        np.save(f"{file_name}", data)

    def write_scipy_sparse_to_file(self, file_name: str, coo_matrix) -> None:
        """ The 'coo_matrix' is expected to be a 'scipy.sparse.coo.coo_matrix'."""
        save_npz(f"{file_name}", coo_matrix.tocsr())

    def write_wllt_to_file(self) -> List[str]:
        header_row  = "# Wl-depth; number of wl-labels; number of computed weights. Then all files needed to read in this WLLT."
        data_row    = f"{self.get_wl_iteration()}, {self.get_nr_wl_labels()}, 0"
        file_rows   = self.get_all_file_paths()
        all_rows = [header_row]+[data_row]+file_rows
        np.savetxt(f"{self.get_meta_file_name()}", all_rows, delimiter=WLLT_META_DELIMITER, fmt='%s')
    
    def set_edge_weights(self, values: np.array) -> None:
        self.write_np_to_file(self.get_edge_weights_file_name(), values)

    ### WLLT-Initialization methods ###

    def initialize_WLLT_files_from_adj_lists(self, dir_in: str, dataset_name:str = None) -> None:
        
        def initialize_tree_parents_file() -> None:
            """
            This method initializes the file which stores all parents in the tree.
            The first layer (original labels, zeroth wl-labelling) are all children of the artificial vertex with label -1.
            Thus this root is their parent.
            
            Now, when ever a new layer is created (which will be the next unused integer), simply add its parent to the end of existing parent vector.
            """
            # Construct the parent list of the wllt. All initial labels have the artificial root '-1' as parent.
            first_layers_parent_array: np.array = np.array([-1]*self.get_nr_wl_labels(), dtype=int)

            # Save the parent vector.
            self.write_np_to_file(self.get_parent_list_file_name(), first_layers_parent_array)

        def initialize_graph_representation() -> None:            
            self.update_graph_representation_coo_for_layers()

        # First, check if the files do exist! If not, the WLLT procedure stops here without saving any data.
        essential_files = [c.FN_ADJ_LISTS, c.FN_GRAPH_CLASSES, c.FN_GRAPH_VERTICES, c.FN_VERTEX_LABELS]        
        for essential_file in essential_files:
            path = f"{dir_in}/{essential_file}"
            if not exists(path):
                LOG.error(f"ERROR: NO SUCH FILE {path}! No WLLT will be created!")
                return None

        self.dataset_name = dataset_name        

        # Store the directory, where the WLLT and all created files shall be stored.        
        self.dir_in  = dir_in
        self.dir_out = f"{self.dir_in}/{c.DN_WLLT}"
        if not exists(self.dir_out): makedirs(self.dir_out)
        
        # Read in the vertex labels and identify the distinct ones. All these are children of the artificial root note and construct the zeroth WL-layer.        
        original_vertex_labels      = self.get_original_vertex_labels_from_file()
        original_vertex_label_set   = np.unique(original_vertex_labels)
        self.n = len(original_vertex_label_set)
                    
        # Construct the tree paths from the root to all original labels and store them to file.
        initialize_tree_parents_file()

        # Save the size of the first layer to file.
        layer_size=[len(original_vertex_label_set)]
        self.append_to_nparr(self.get_layer_starts_file_name(), layer_size)
        
        # Initialize the wl-uniqueness file.
        self.append_to_nparr(self.get_mean_wl_change_file_name(), [])

        # Save the vertex labels again, as zeroth labelling.
        self.write_np_to_file(self.get_vertex_labels_file_name(0), original_vertex_labels)        

        # Save the vertex label map for layer zero. 
        # That is for every one of the original labels, the artificial DUMMY_LABEL.
        artificial_hash_for_original_labels = [(o, tuple(np.sort([c.DUMMY_LABEL]))) for o in original_vertex_label_set]
        self.write_np_to_file(self.get_vertex_label_map_file_name(), artificial_hash_for_original_labels)

        # Construct the tree paths from the root to all original labels and store them to file.
        # Since this method will use the vertex labels of the 'last iteration', this has to be done after the
        # vertex label file has been stored.
        initialize_graph_representation()

        # Save the whole wllt to a file.
        self.write_wllt_to_file()        
    
    def initialize_WLLT_from_existing_WLLT(self, dir_out: str, wl_iteration: int = None):
        """
        This method searches for a META file in the given output directory and initializes key values of the WLLT constructor.
        """        
        try:
            wllt_meta_file = f"{dir_out}/{c.FN_META_INFO}"
            meta_data = np.loadtxt(wllt_meta_file, delimiter="\t", dtype=str, skiprows=1)

            # Set the wllt iteration.
            key_values: str = meta_data[0].split(", ")
            i = int(key_values[0])
            if wl_iteration is not None: 
                i = min(i, wl_iteration)
            self._set_wl_iteration_ctr(i)            
            
            # Set the dataset name and output dir (WLLT folder).
            dir_out_list = dir_out.split('/')
            self.dataset_name   = dir_out_list[-2]
            self.dir_out        = dir_out
            self.dir_in         = "/".join(dir_out_list[:-1])
            
            # Set the number of tree vertices.
            # Figure out how many vertices this tree contains, by finding the index of the last vertex in the layer.
            # This has to be done AFTER the definition of the 'output_dir'. Otherwise the file name will be incomplete.
            layer_start = self.get_layer_starts_from_file()[i]
            self.n = layer_start

        except (FileNotFoundError, IOError):
            LOG.error(f"ERROR:\tCould not find META-file '{wllt_meta_file}'!")

    def compute_nr_of_non_bijective_changes(self, new_wl_label_list: np.array, old_wl_label_list: np.array, vertices_matrix: np.array) -> float:
        """
        Takes a list of wl-labels for all vertices and the list of vertices for every graph.
        Computes the level of distinct
        """
        new_vertex_labels_per_graph = new_wl_label_list[vertices_matrix]
        old_vertex_labels_per_graph = old_wl_label_list[vertices_matrix]
        
        # Read the list of all graph vertices for every graph from file.        
        max_nr_of_vertices: int = vertices_matrix.shape[1]
        nr_of_changes: int      = 0

        for graph_id, new_graph_wl_labels in enumerate(new_vertex_labels_per_graph):                        
            old_graph_wl_labels = old_vertex_labels_per_graph[graph_id]
            
            # The DUMMY_LABEL does not need to be excluded, but excluding it results in more meaningull values.
            nr_of_vertices = max_nr_of_vertices
            # Remove the DUMMY_LABEL padding.
            ids_of_padding = np.where(new_graph_wl_labels==c.DUMMY_LABEL)[0]
            # If there exists a DUMMY_LABEL in the  wl-label string, its position indicates the size of the graph.
            # Otherwise the graph is one of the biggest with size 'max_nr_of_vertices'.
            if len(ids_of_padding) > 0: 
                nr_of_vertices = ids_of_padding[0]
                # Since the indexing starts at zero, the cut of index is offsetted by one.
                new_graph_wl_labels = new_graph_wl_labels[:nr_of_vertices-1]
                old_graph_wl_labels = old_graph_wl_labels[:nr_of_vertices-1]
            
            if is_bijection(new_graph_wl_labels, old_graph_wl_labels): nr_of_changes += 1
                
        return nr_of_changes

    ### Add WLLT layer ###

    def add_WLLT_layers(self, nr_new_layers: int = 1) -> None:
        """
        This method generates layers of WLLT-labels.
        """
        # If the tree has not been initialized, use the meta.
        if self.dir_out == "":
            LOG.error("Fct 'add_WLLT_layers' is trying to add layers to a not initialized tree! Make sure the initialization is complete!")
            return None

        # Load the matrix containing all neighborhoods (adjacency lists).
        # Notice that the smallest label is 0. A value of -1 does not indicate a value but is a DUMMY LABEL.
        # In order to use the adjacency lists as indices in the wl-labels file, the last index (-1) must be '-1' again.
        neighborhood_indixes: np.array      = self.get_adj_lists_from_file()
        all_old_wl_labels: np.array         = self.get_vertex_labels_from_file(with_tailing_dummy_label=True)        
        all_old_wllt_parents: np.array      = self.get_parent_list_from_file()
        vertices_matrix: np.array           = self.get_graph_vertices_from_file()

        nr_vertices: int        = all_old_wl_labels.shape[0]
        all_new_wl_labels       = np.zeros(nr_vertices, dtype=int)
        all_new_wllt_parents    = list()
        new_layer_starts = list()
        new_layer_sizes: List[Tuple[int, int]] = list()        
        nr_wl_cange_list: List[float] = np.load(self.get_mean_wl_change_file_name(), allow_pickle=True)

        no_changes_in_last_iter: bool = False

        new_iterations = range(self._next_wl_iteration(), self._next_wl_iteration() + nr_new_layers)

        for ptr, i in enumerate(new_iterations):
            LOG.info(f"{self.dir_out},\tWL-iter:\t{i}")
            hash_to_wl_label_dict: Dict[Tuple, int] = dict()
            # Notice here: The neighborhood indices contain tailing DUMMY INDICES (-1).
            # We will keep these values, since the last entry in 'all_old_wl_labels' is again a -1.
            # This entry thus can be interpreted both as an dummy-index, or a dummy-label.
            neighborhood_labels_with_dummy: np.array = all_old_wl_labels[neighborhood_indixes]
            
            for vertex_id, neighborhood_labels_with_dummy in enumerate(neighborhood_labels_with_dummy):                
                old_vertex_wl_label = all_old_wl_labels[vertex_id]                

                neighborhood_labels = neighborhood_labels_with_dummy[neighborhood_labels_with_dummy != c.DUMMY_LABEL]
                conc: Tuple[int, Tuple] = (old_vertex_wl_label, tuple(np.sort(neighborhood_labels)))
                requrire_new_label: bool = conc not in hash_to_wl_label_dict
                
                if requrire_new_label:
                    # Save the new wl-label.
                    new_vertex_wl_label = self.get_nr_wl_labels()
                    hash_to_wl_label_dict[conc] = new_vertex_wl_label
                    self.increment_nr_wl_labels()

                    all_new_wllt_parents.append(old_vertex_wl_label)
                
                # Save the new wl label of the current vertex.
                all_new_wl_labels[vertex_id] = hash_to_wl_label_dict[conc]                

            # Append the DUMMY_LABEL '-1' such that: 'all_new_wl_labels[-1] = -1'
            all_new_wl_labels = np.append(all_new_wl_labels, np.array([c.DUMMY_LABEL], dtype=int))


            # Compute the mean levels of change. If it is close to zero, report that more iterations may be depreciated.
            nr_of_changes = self.compute_nr_of_non_bijective_changes(all_new_wl_labels, all_old_wl_labels, vertices_matrix)
            nr_of_graphs = vertices_matrix.shape[0]
            nr_wl_cange_list = np.append(nr_wl_cange_list, nr_of_changes)
                        
            mean_wl_change_percentage = round(nr_of_changes / nr_of_graphs * 100, 2)
            if nr_of_changes == 0:
                nr_remaining_iterations = len(new_iterations) - ptr - 1
                if nr_remaining_iterations > 0:
                    if no_changes_in_last_iter is False:
                        no_changes_in_last_iter = True
                    else:
                        LOG.warning(f"No graph representations (out of {nr_of_graphs} graphs) have changes (non-bijectively) in the last two iterations!")
                        input_yn = 'n'
                        if self.ask_for_early_stopp:
                            input_yn: str = input(f"Do you wish to continue? ({nr_remaining_iterations} iteration{'s' if nr_remaining_iterations > 1 else ''} remaining) [y/n]")
                        if input_yn == 'n':
                            LOG.info(f"No more WL-iterations computed, since on no WL-label changeds non-bijectively.")
                            new_iterations = new_iterations[:ptr]
                            break
            elif mean_wl_change_percentage < MEAN_WL_CHANGE_WARN:
                LOG.warning(f"On average only {mean_wl_change_percentage}% (<{MEAN_WL_CHANGE_WARN}%) of all WL-labels have changed non-bijectively. [{nr_of_changes}/{nr_of_graphs}]")
              
            # And store them to file.
            self.write_np_to_file(self.get_vertex_labels_file_name(i), all_new_wl_labels)

            # Sort the wl-hash dictionary to get the hash values in the right order. This way the indices of the result can be used as key.
            # Notice, that the wl_labels are the values in the dictionary. Thus it is sorted using them, the 1-st entry of the dict-element.
            hash_map = [conc for conc, w_label in sorted(hash_to_wl_label_dict.items(), key=lambda ele: ele[1])]
            # Save the vertex label map, by appending the hashes to the existing file.
            self.append_to_vertex_map(hash_map)

            new_layer_starts.append(self.get_nr_wl_labels())
            new_layer_sizes.append((i, len(all_new_wl_labels)))

            # Update the vector used to get the last wl-labels.
            all_old_wl_labels = np.copy(all_new_wl_labels)

            # After the termination of this iteration, increment the number of wl-iterations.
            self._increment_wl_iteration_ctr()
        
        LOG.info("\t> Writing data to file.")
        # Since the vector of tree parents is not used in the method itself, but rather a documentation of the complete tree,
        # it can be updated after all layers have been created.
        self.write_np_to_file(self.get_parent_list_file_name(), np.append(all_old_wllt_parents, all_new_wllt_parents))

        # Update the layer starts.
        self.append_to_nparr(self.get_layer_starts_file_name(), new_layer_starts)

        # Replace the estimated mean change between the wl-labels, with the appended list.
        np.save(self.get_mean_wl_change_file_name(), nr_wl_cange_list)

        # Update the graph representations.
        print("\t Updating the graph representations file ...", end="")
        self.update_graph_representation_coo_for_layers(new_iterations)
        print("\r")

        # Update the information stored in the meta file.
        self.write_wllt_to_file()
        LOG.info(f"Finished adding {nr_new_layers} new layers.")

    ### Graph information method ###

    def get_graph_info_excel_str(self, graph_id: int, save_to_file: bool = False, dir_out_prefix: str = "") -> str:
        """
        Retrieves all information known about the graph with id 'graph_id' and returns it in a print-string.
        If a 'output_file_name' is given, the print string is saved to a '.txt'-file with that name.

        The retrieved information is:
            - The graphs id, vertex indices, neighborhood lists for every vertex.
            - For every iteration/layer of the wllt: the wl-labels for all vertices (including the originals at iteration zero).
            - The graph representation as a normalized sparse histogram over all wl-labels in the tree.
        """        
        graph_vertex_list   = list(self.get_graph_vertices_from_file()[graph_id])
        graph_class         = self.get_graph_classes_vec_from_file()[graph_id]

        ctr = Counter(graph_vertex_list)
        nr_of_padding = ctr[c.DUMMY_LABEL]
        nr_vertices = len(graph_vertex_list) - nr_of_padding
        graph_vertex_list   = graph_vertex_list[:nr_vertices]

        graph_repr_dict       = dict()
        graph_representation  = self.get_graph_repr_from_file().getrow(graph_id).toarray()[0]
        for k, v in enumerate(graph_representation):
            if v != 0.0:
                graph_repr_dict[k] = v

        mean_nr_vertices, _ = self.get_mean_max_nr_vertices()
        inverse_normalization = nr_vertices / mean_nr_vertices
        abs_graph_repr_dict   = dict(zip(graph_repr_dict.keys(), [int(round(inverse_normalization * i,0)) for i in graph_repr_dict.values()]))
        
        graph_neighborhoods   = self.get_adj_lists_from_file()[graph_vertex_list]
        wl_layers       = self.get_wl_labels_layer_wise()

        # Assemble the excel print:
        print_str = f"Dataset\t{self.dataset_name}"
        print_str += f"\nGraph id\t{graph_id}"
        print_str += f"\nGraph class\t{graph_class}"
        graph_vertices_str = "\t".join([f"v{i}" for i in graph_vertex_list])
        print_str += f"\nVertices\t" + graph_vertices_str
        print_str += f"\nn\t{nr_vertices}"
        print_str += "\n"
        print_str += f"\nNeighborhood lists"        
        max_neighborhood_size: int = 0
        neighborhoods_str: str = ""
        for v, N in enumerate(graph_neighborhoods):
            neighbor_str = ""
            if len(N) > max_neighborhood_size: max_neighborhood_size = len(N)
            for n in N:
                if n == c.DUMMY_LABEL:
                    break
                neighbor_str += f"\t{n}"
            neighborhoods_str += f"\n{graph_vertex_list[v]}" + neighbor_str

        print_str += f"\nvertex_id\t" + "\t".join([f"n{i}" for i in range(max_neighborhood_size - 1)])
        print_str += neighborhoods_str

        print_str += "\n"
        print_str += f"\nVertex wl-labels per iteration:"
        print_str += f"\nIteration\tWL-lbls" + "\t" * len(graph_neighborhoods) + "\tWL-labels in the graph"
        print_str += "\n\t" + graph_vertices_str
        for d in range(self._latest_wl_iteration() + 1):
            print_str += f"\n{d}"
            graph_vertex_labels = self.get_vertex_labels_from_file(d, with_tailing_dummy_label=False)[graph_vertex_list]
            wl_labels_d = "\t".join([f"{i}" for i in graph_vertex_labels])
            print_str += f"\t{wl_labels_d}"
            print_str += f"\n\t" + "\t" * len(graph_neighborhoods) + f"\t[{wl_layers[d][0]} ... {wl_layers[d][-1]}]"

        print_str += "\n"
        print_str += f"\nWL-Label\tRepr-Value\tAbs frequency"
        for label, value in graph_repr_dict.items():
            print_str += f"\n{label}\t{value}\t{abs_graph_repr_dict[label]}"

        print_str += "\n"
        print_str += f"\nAll WL-Labels\tRepr-Value\tAbs frequency"
        for label, v in enumerate(graph_representation):
            value = str(v)
            if v == 0.0: value = ''
            print_str += f"\n{label}\t{value}"

        # If specified, save the print to file.
        if dir_out_prefix != "":
            if not exists(dir_out_prefix): mkdir(dir_out_prefix)
            dir_out_prefix += "/"
        if save_to_file:
            file_name = f"{dir_out_prefix}info_{self.dataset_name}_graph_{graph_id}.ods"
            with open(file_name, 'w') as f:
                f.write(print_str)
            
            print(f"Information on graph {graph_id}\tsaved to file '{file_name}'.")
        
        self.save_graph_to_png(graph_id, graph_vertex_list, graph_neighborhoods, dir_out_prefix=dir_out_prefix)
        return print_str

    def save_graph_to_png(self, graph_id: int, vertex_list: List[int], graph_neighborhoods, dir_out_prefix: str = None):
        """
		To execute this, a downgrade to matplotlib version to 2.2.3 may ba necessary.
		pip install matplotlib==2.2.3.
		:return:
		"""
        def convert_adj_list_to_edge_list() -> List[Tuple[int, int]]:
            edge_set = set([])
            for v, N in enumerate(graph_neighborhoods):
                for n in N:
                    if n != -1 and vertex_list[v] < n:
                        edge = (vertex_list[v], n)
                        edge_set.add(edge)                        
            
            return list(edge_set)

        def get_networkx_graph():
            networkx_graph = nx.Graph()
            nodes = vertex_list
            nodes.sort()
            networkx_graph.add_nodes_from(nodes)
            networkx_graph.add_edges_from(edge_list)

            return networkx_graph

        edge_list = convert_adj_list_to_edge_list()
        vertex_labels = self.get_vertex_labels_from_file(0, with_tailing_dummy_label=False)[vertex_list]
        graph_class = self.get_graph_classes_vec_from_file()[graph_id]

        # Construct vertex annotations.
        vertex_annotation_strings = [f"\n\n$\ell(${v}$)=$ {vertex_labels[i]}" for i, v in enumerate(vertex_list)]
        plot_annotation_dictionary = dict(zip(vertex_list, vertex_annotation_strings))
		        		
        figure = plt.figure()

		# Plot the graph and its legend.	
        networkx_graph = get_networkx_graph()
        nx.draw(networkx_graph, labels=plot_annotation_dictionary, with_labels=True)
        plt.legend()
        
		# Construction of the title
        title_str  = f"{self.dataset_name}-Graph {graph_id}"
        title_str += f"\nn={len(vertex_list)}, m={int(len(edge_list) / 2)}, c={graph_class}"
        plt.title(title_str)
		
		# Display everything together
        figure.savefig(f"{dir_out_prefix}plot_graph{graph_id}.png", bbox_inches='tight', nbins=0, pad_inches=0.0)        
        plt.close(figure)

    def save_wllt_to_png(self, dir_out: str = "", edge_weights: np.array = None, title_postfix: str = "", layer_id_lim: int = None, verbose: bool = False):
        """
		To execute this, a downgrade to matplotlib version to 2.2.3 may ba necessary.
		pip install matplotlib==2.2.3.
		:return:
		"""
        def normalize(arr, t_min, t_max):
            """ Normalize a one dimensional array to the interval [t_min, t_max]. """
            norm_arr = []
            diff = t_max - t_min
            diff_arr = max(arr) - min(arr)
            for i in arr:
                temp = (((i - min(arr))*diff)/diff_arr) + t_min
                norm_arr.append(temp)
            return norm_arr

        def get_networkx_tree(vertex_list: List[int], edge_list: List[Tuple[int, int]], edge_weights: Dict[Tuple[int, int], float], edge_colors: Dict[Tuple[int, int], float]):
            networkx_graph = nx.Graph()
            nodes = vertex_list
            # nodes.sort()
            networkx_graph.add_nodes_from(nodes)
            networkx_graph.add_edges_from(edge_list)
            attrs = {}
            for e in edge_list:
                attrs[e] = dict(zip(['weight', 'color'], [edge_weights[e], edge_colors[e]]))

            nx.set_edge_attributes(networkx_graph, attrs)
            return networkx_graph
        
        def compute_layer_limit(layer_id_limit: int, l_starts: np.array) -> int:
            # Since the tree may be huge, discard lower layers.
            # If no layer_limit has been passed along, find the highest layer with less than 'vertex_display_threshold' many vertices.
            vertex_display_threshold = 407 # 150
            if layer_id_limit is None:
                if l_starts[0] > vertex_display_threshold:
                    print(f"\r\t\tNo WLLT figure plotted, since the first layer has already {l_starts[0]} (>{vertex_display_threshold}) vertices. That is to huge!.")
                else:
                    # Assume the highest layer has not to many vertices and can be plotted.
                    layer_id_limit = len(l_starts) - 1
                    for layer_id, layer_start_id in enumerate(l_starts):
                        # Go through all layers, and if there is one layer that is to big, plot up to the layer BEFORE it.
                        if layer_start_id > vertex_display_threshold:
                            # All layers after this will be bigger. Thus take the layer BEFORE and stop the search.                        
                            layer_id_limit = layer_id-1
                            break
            return layer_id_limit 

        def plot_wllt(vertex_list, parents_list, edge_weights, dir_out: str) -> None:
            # If not all entries are equal...
            if not all(elem == edge_weights[0] for elem in edge_weights):
                # ...normalize the edge weights to the interval [0.2,1].
                # Depending on the tree structure, this may result in a good plot of small and big edges.
                edge_weights = normalize(edge_weights, 0.2, 1)

            w_unique = np.unique(edge_weights)
            edge_colors = dict(zip(w_unique, range(1, len(w_unique)+1)))
                            
            # Add the artificial root (-1) to the vertex list.
            vertex_list = [-1] + vertex_list
            edge_list = np.column_stack((np.arange(len(parents_list), dtype=int), parents_list))

            # Construct the data for the NetworkX graph.
            edge_tuple_list = list()
            edge_weights_dict = dict()
            edge_colors_dict  = dict()
            for i, e in enumerate(edge_list):
                edge = (e[0], e[1])
                edge_tuple_list.append(edge)
                edge_weights_dict[edge] = edge_weights[i]
                edge_colors_dict[edge]  = edge_colors[edge_weights[i]]

            # NetworkX graph assembly.
            networkx_graph = get_networkx_tree(vertex_list, edge_tuple_list, edge_weights_dict, edge_colors_dict)
                            
            figure = plt.figure()
            # Plot the graph and its legend.	
            pos = nx.kamada_kawai_layout(networkx_graph)
            edges   = networkx_graph.edges()
            colors  = [networkx_graph[u][v]['color'] for u,v in edges]        
            pos = nx.kamada_kawai_layout(networkx_graph)
            options = {
                "node_color": "#EEEEEE",
                "edge_color": colors,
                "width": 2.0,
                "edge_cmap": plt.cm.tab10, # https://matplotlib.org/stable/tutorials/colors/colormaps.html
                "with_labels": True,
            }
            nx.draw(networkx_graph, pos, **options)
                    
            # Construction of the title
            title_str  = f"WLLT_l{layer_id_lim+1}_{title_postfix}"        
            plt.title(title_str)
            
            if dir_out != "":
                if not exists(dir_out): mkdir(dir_out)
                dir_out += "/"

            # Display everything together
            file_name = f"{dir_out}plot_wllt_l{layer_id_lim+1}{title_postfix}.png"
            figure.savefig(f"{file_name}", bbox_inches='tight', nbins=0, pad_inches=0.0)
            print("\r", end="")        
            if verbose: print(f"\r\tWLLT figure saved to file '{file_name}'.", end="")
            plt.close(figure)

        vertex_list = list(range(self.get_nr_wl_labels()))
        parents_list = self.get_parent_list_from_file().astype(int)

        # Edge weights definition.
        if edge_weights is None: edge_weights = np.ones(len(parents_list))               

        layer_starts = self.get_layer_starts_from_file()        
        layer_id_lim = compute_layer_limit(layer_id_lim, layer_starts)
        if layer_id_lim is None: return None

        for layer_id in range(0, layer_id_lim + 1):
            max_vertex_id = layer_starts[layer_id]
            # Crop the data to only this highest layer.
            plot_wllt(vertex_list[:max_vertex_id], parents_list[:max_vertex_id], edge_weights[:max_vertex_id], dir_out)        

def get_wllt_dirin_dirout(dataset_name: str) -> Tuple[str, str]:
    in_dir = f"{c.get_datafiles_dir()}/{c.DN_DATAFILES}/{dataset_name}"    
    if not exists(in_dir): 
        print(f"Datafiles not found! Searching for '{in_dir}'.\nYou may need to run 'x1_dataset_to_globalAdjList.py -d {in_dir}' first.")
        return None, None
    dir_out = f"{in_dir}/{c.DN_WLLT}"
    return in_dir, dir_out

def get_WLLT_from_dataset_name(dataset_name: str, wl_depth: int = None) -> WLLT:    
    dir_in, dir_out = get_wllt_dirin_dirout(dataset_name)

    if dir_out is None or not exists(dir_out):
        print(f"WLLT does not exist! Searching '{dir_out}'!")
        return None

    # Read the already constructed WLLT.
    wllt = WLLT()
    wllt.initialize_WLLT_from_existing_WLLT(dir_out=dir_out, wl_iteration=wl_depth)
    return wllt

############################################################################## 
##### EDGE WEIGHT LEARNER ####################################################
##############################################################################

### Edge weight initialization interface ###
class EdgeWeightInitializator(ABC):
    """
    This class ensures that different edge weight initializations are performed in the same way.
    For all implementations, a WLLT is read in. When a implementatin is choosen, its constructor
    may require some parameters. The constructor will call the super method right away, which 
    triggers the chosen edge weight implementation.
    Thus the edge weights will be computed and stored to the file specified by the WLLT, 
    right after calling the constructor. They can be accessed with a method too.
    """

    def __init__(self, wllt: WLLT):
        self.wllt: WLLT = wllt
        self.file_name_edge_weights: str = wllt.get_edge_weights_file_name()
        self.edge_weights: np.array = self._initialize_edge_weights()
        
        np.save(self.file_name_edge_weights, self.edge_weights)        

    def get_edge_weights(self) -> np.array:
        return self.edge_weights

    @abstractmethod
    def _initialize_edge_weights(self) -> np.array:
        """
        Initialized edge weights for the given WLLT, 
        returns them as array and 
        saves them to the file, specified by the WLLT.
        """        
        raise NotImplementedError
        
### Edge weight initialization implementations ###
class RandomEdgeWeightInitializator(EdgeWeightInitializator):

    def __init__(self, wllt: WLLT, min_random: float = 0.0, max_random : float = 1.0):
        self.min: float = min_random
        self.max: float = max_random
        super(RandomEdgeWeightInitializator, self).__init__(wllt=wllt)

    def _initialize_edge_weights(self) -> np.array:
        return self.min + np.random.rand(self.wllt.get_nr_wl_labels()) * (self.max - self.min)

class ConstantEdgeWeightInitializator(EdgeWeightInitializator):

    def __init__(self, wllt: WLLT, const_value: float = 1.0):
        self.const: float = const_value
        super(ConstantEdgeWeightInitializator, self).__init__(wllt=wllt)

    def _initialize_edge_weights(self) -> np.array:
        return self.const + np.zeros((self.wllt.get_nr_wl_labels()))

class LayerBasedEdgeWeightInitializator(EdgeWeightInitializator):

    def __init__(self, wllt: WLLT, layer_weight_sums: Any = 1.0):
        """
        Initializes the edge weights such that every edge weights in layer L has value 'layer_sum/#L'.
        Thus every layer has the same weighted sum. 
        Layers closer to the root tend to be smaller, thus their edge weights are heavier.
        """
        self.layer_starts: float        = wllt.get_layer_starts_from_file()        
        if type(layer_weight_sums) == float: layer_weight_sums: List[float] = [layer_weight_sums] * len(self.layer_starts)
        self.layer_weight_sums: float = layer_weight_sums
        super(LayerBasedEdgeWeightInitializator, self).__init__(wllt=wllt)

    def _initialize_edge_weights(self) -> np.array:        
        w = np.zeros((self.wllt.get_nr_wl_labels()))
        last_layer_start = 0
        for layer_id, next_layer_start in enumerate(self.layer_starts):
            # Compute the size of layer 'layer_id'.
            layer_size = next_layer_start - last_layer_start
            # Compute the weights for all edges in this layer.            
            value = self.layer_weight_sums[layer_id] / layer_size
            # Save the computed edge weights in this.
            w[last_layer_start:next_layer_start] = value

            # Set the next index after the last layer start, as the next layer start.
            last_layer_start = next_layer_start

        return w

class FRMEdgeWeightInitializator(EdgeWeightInitializator):

    def __init__(self, wllt: WLLT, frm_setting):        
        self.frm_setting: float = frm_setting
        super(FRMEdgeWeightInitializator, self).__init__(wllt=wllt)

    def _initialize_edge_weights(self) -> np.array:
        print("This method has NOT been implemented yet! It will probably take to much time for now.")
        return self.frm_setting()

############################################################################## 
##### MAIN ###################################################################
##############################################################################

def main(dataset_name: str = "MUTAG", nr_new_layers: int = 11, ask_for_early_stopp: bool = True, print_info_and_plots_graphs: List[int] = None, edge_weight_mode: str = 'const'):
    dir_in, dir_out = get_wllt_dirin_dirout(dataset_name)    
    
    # Construct and saved the WLLT to file.
    wllt = WLLT(ask_for_early_stopp)
    possible_meta_file = f"{dir_out}/{c.FN_META_INFO}"
    if exists(possible_meta_file):  
        wllt.initialize_WLLT_from_existing_WLLT(dir_out=dir_out)
    else:
        wllt.initialize_WLLT_files_from_adj_lists(dir_in=dir_in, dataset_name=dataset_name)    
    
    wllt.add_WLLT_layers(nr_new_layers=nr_new_layers)    
    wllt.save_wllt_to_png(dir_out=dir_out)

    # Add edge weights.
    if edge_weight_mode == 'const':
        const_value = 1.0
        edge_weight_initializator = ConstantEdgeWeightInitializator(wllt=wllt, const_value=const_value)
    elif edge_weight_mode == 'layer_size':
        edge_weight_initializator = LayerBasedEdgeWeightInitializator(wllt=wllt)
    elif edge_weight_mode == 'exp':
        n = len(wllt.get_layer_starts_file_name())
        layer_weight_sums = [np.exp(x) for x in range(-n, 0)]
        edge_weight_initializator = LayerBasedEdgeWeightInitializator(wllt=wllt, layer_weight_sums=layer_weight_sums)
    else:
        edge_weight_initializator = ConstantEdgeWeightInitializator(wllt=wllt, const_value=1.0)
    print(edge_weight_initializator.get_edge_weights())

    if print_info_and_plots_graphs is not None:
        dir_graph_outs = f"{dir_out}/Graph_informations"
        for g_id in print_info_and_plots_graphs:
            wllt.get_graph_info_excel_str(g_id, save_to_file=True, dir_out_prefix=dir_graph_outs)  

    LOG.info("WLLT Constructor terminated.")

def parse_terminal_args() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
        default=['MUTAG'],
        dest='dataset_names',
        help='Provide TU Dortmund dataset names.',
        type=str,
        nargs='+')
    parser.add_argument('-n',
        default=10,
        dest='nr_new_layers',
        help='Number of new WLLT layers.',
        type=int)
    parser.add_argument('-a',
        default=False,
        dest='ask_for_early_stopp',
        help='Shall the program ask for early stopping?',
        type=bool)
    args = parser.parse_args(sys.argv[1:])
        
    for dataset_name in args.dataset_names:
        main(dataset_name=dataset_name, nr_new_layers=args.nr_new_layers, ask_for_early_stopp=args.ask_for_early_stopp)

if __name__ == "__main__":
    # Run in terminal as: python3 x2_wllt_constructor.py -d MUTAG -i 3 -e 100
    parse_terminal_args()