from typing import List, Dict, Tuple, Union
import numpy as np
from os import makedirs
from os.path import exists
from logging import ERROR as LOG_ERR
# For the parsing of console input:
import sys, getopt, argparse

# Own files:
from dataset_loaders import DatasetLoader, get_suitable_dataloader
import my_consts as c

# GLOBAL DEBUGGING LOGGER for this script
from my_utils.decorators import get_logger
LOG = get_logger(filename="Loggers/x1_dataset_to_globalAdjList.txt", level=LOG_ERR, create_file=True)

### Main computation methods ###

def get_adjacency_lists_from_GraKel_graph(graph: List) -> Tuple[List[List[int]], int]:
    """
        Gets a simple grakel graph. Returns a list for each vertex, consisting of 
        the global vertex id and all its neighbors (with global incies).
        Notice that the vertex ids in the GraKel datasets start with index '1'!

        Notice that a List of List of integers is returned and NOT a numpy array.
        That is due to the face, that the neighborhoods can have different sizes and the grouped stroage
        of such different sized arrays is depreciated. In the outer method, all these Lists are expanded to
        then fit in a numpy matirx with tailing dummy entries if the neighborhood is smaller that the maximum one.

        This is equivalent to using the grakel.Graph class and Graph(graphs[i][0]).neighbors(v_id).

        The return value has the following format (using global indices only):
        List[
            v1, N(v1)[0], N(v1)[1], N(v1)[2], ...
            v2, N(v2)[0], N(v2)[1], N(v2)[2], ...
        ], MaximumNeighborhoodSize: int
    """
    neighbor_dict: Dict[int, List[int]] = dict()
    
    # Iterate over all edges in the graph.
    for v1, v2 in graph[0]:
        # Add neighbor vertex 2 to vertex 1. Notice that the edges are considered as directed and the neighborhood too (out-going neighborhood).
        if neighbor_dict.get(v1) is None:
            neighbor_dict[v1] = [v2]
        else:
            neighbor_dict[v1] += [v2]
    
    # For better readability only, sort all neighborhoods in the dictionary.
    for k, v in neighbor_dict.items():
        v_sorted = v.copy()
        v_sorted.sort()
        neighbor_dict[k] = v_sorted

    # To account for neighbors without a neighborhood, we have to check the list of vertices.    
    # For every vertex in the graph, that has not been found in the edge set, create an empty neighborhood.
    for v in graph[1]:
        if not neighbor_dict.get(v):
            neighbor_dict[v] = list()


    # Convert the dictionary in a list of lists (for vertex v: [v, N(v)_1, N(v)_2,...]) and return this.
    neighborhood_list: List[List[int]] = [[v] + N for v, N in neighbor_dict.items()]
    max_neighborhood_size = max(len(N) for N in neighbor_dict.values())
    return neighborhood_list, max_neighborhood_size

def convert_adjacency_lists_to_matrix(adjacency_lists: List[List[int]], max_neighborhood_size: int) -> np.array:
    """
    The 'adjacency list' contains a list of lists of different sizes.
    Each such list is constructed as the vertex id, and the vertex ids of its neighborhood.

    Simplify this structure by only storing the neighbor ids in a matrix at the row, indicated by the vertex id.
    Since numpy cannot handle a matrix with different row lenghts, fill up all neighborhoods with the DUMMY INDEX -1.
    """    
    N = len(adjacency_lists)
    M = max_neighborhood_size
    adjacency_mat = c.DUMMY_INDEX * np.ones((N, M), dtype=int)

    # Writing the neighborhoods to the rows.
    for vertex_and_neighbors_list in adjacency_lists:
        # If the vertex does not have any neighbors, the list will only contain the vertex itself.
        # In this case nothing has to be done, since the DUMMY_INDICES indicate an empty neighborhood by default.
        if len(vertex_and_neighbors_list) == 1:
            continue
        vertex_id = vertex_and_neighbors_list[0]
        j = len(vertex_and_neighbors_list) - 1
        # Exclude the first value, since it is the vertex index and not part of its (true) neighborhood.        
        adjacency_mat[vertex_id,:j] = vertex_and_neighbors_list[1:]

    return adjacency_mat

def convert_list_of_lists_to_matrix(list_of_lists: List[List[int]], max_list_lenght: int = None, start_off_set: int = 0) -> np.array:
    """
    This method is used to convert a list of lists, which contains lists of different different lengths (e.g. adjacency lists or vertex indices)
    into a matrix which can be easily stored to and loaded from file.    
    
    Since numpy cannot handle a matrix with different row lenghts, all lists (elements in the list) will be padded with the DUMMY INDEX -1.
    """
    if max_list_lenght is None:
        max_list_lenght = max([len(list) for list in list_of_lists])

    N = len(list_of_lists)
    M = max_list_lenght
    matrix = c.DUMMY_INDEX * np.ones((N, M), dtype=int)
    
    for i, list in enumerate(list_of_lists):
        matrix[i,:len(list) - start_off_set] = list[start_off_set:]

    return matrix

def convert_matrix_row_to_adjacency_list(row: np.array) -> np.array:
    # Return the row up to the first c.DUMMY_INDEX. This was the padding in order to store all in a matrix shape.
    indices_of_dummy_values = np.nonzero(row==c.DUMMY_INDEX)[0]

    if indices_of_dummy_values.shape[0] == 0:
        return row
    else:
        return row[:indices_of_dummy_values[0]]

def construct_global_adj_list_matrix(dataset_name: str, reduced_graph_nr: int = Union[int, List[int]], dir_suffix: str = '') -> None:
    print(f"Constructing global adj list for dataset {dataset_name}")
    grakel_loader: DatasetLoader = get_suitable_dataloader(dataset_name)
    if grakel_loader is None: return None
    # Keep graphs without edges but eliminate graphs without vertices.
    graphs, classes = grakel_loader.get_general_dataset_cleaned_graphs_and_classes(dataset_name, delete_vertexles_graphs=True, delete_edgeles_graphs=False, require_edge_labels=False)
    if reduced_graph_nr is not None:
        if type(reduced_graph_nr) is int:
            graphs, classes = graphs[:reduced_graph_nr], classes[:reduced_graph_nr]
        if type(reduced_graph_nr) is list:
            graphs, classes = graphs[reduced_graph_nr], classes[reduced_graph_nr]

    # Ensure that the vertex indices are from an interval [0, n].
    # This is later used to use the vertex indices as vector indices.
    graphs, _ = grakel_loader.map_dataset_vertices_to_range(graphs)

    # Ensure that the original vertex label from an interval [0, m].
    # This is later used to get trivial wl-labels which can be used as vector indices. 
    # The original vertex labels are considered as wl-labels in the zeroths wl-iteration.
    graphs, _ = grakel_loader.map_vertex_labels_to_range(graphs)

    adjacency_lists = []
    # The maximum neighborhood size will determine the number of columns when storing all adjacency lists.
    max_neighborhood_size = 0
    # Store all vertex labels in one list for the complete graph.
    vertex_label_list = np.array([], dtype=int)
    # Store all vertices per graph. This information will be needed to construct graph representation vectors.
    graph_vertices_list = list()
    max_vertex_set_size = 0
    
    # Note that the graphs consist of [EdgeSet{(v1, v2), ...}, VertexLabelDict{v1: l1, v2: l2, ...}]
    i = 0    
    for g in graphs:
        i += 1
        # Store all vertices per graph.
        graph_vertices_list.append(np.array(list(g[1].keys()), dtype=int))
        max_vertex_set_size = max(max_vertex_set_size, len(g[1]))

        # Save the global vertex id and its label in a continuous array for all vertices in all graphs.
        # Notice that this step only works, if the order of the vertex indices is not changed in the 'uniquefiy_vertex_indices'-function.
        vertex_label_list = np.append(vertex_label_list, list(g[1].values()))

        # Accumulate the neighborhoods/adjacency lists. Notice that these lists of lists (neighborhoods per vertex)
        # may have different lenghts and thus cannot directly be stored in a numpy matrix.
        g_adjacency_lists, tmp_max_neighborhood_size = get_adjacency_lists_from_GraKel_graph(g)
        adjacency_lists += g_adjacency_lists
        max_neighborhood_size = max(max_neighborhood_size, tmp_max_neighborhood_size)

    adjacency_mat       = convert_adjacency_lists_to_matrix(adjacency_lists, max_neighborhood_size)
    graph_vertices_mat  = convert_list_of_lists_to_matrix(graph_vertices_list, max_vertex_set_size)    

    # Save the adjacency lists and the original vertex labels to file.  
    storage_directory = f"{c.get_datafiles_dir()}/{c.DN_DATAFILES}/{dataset_name}{dir_suffix}"
    if not exists(storage_directory): makedirs(storage_directory)

    np_save(f"{storage_directory}/{c.FN_GRAPH_CLASSES}",    classes)             # Format: np.array:     [class_g0, class_g1, ...]
    np_save(f"{storage_directory}/{c.FN_GRAPH_VERTICES}",   graph_vertices_mat)  # Format: np.array:     [np.array[g0_v1, g0_v2, ... c.DUMMY_INDEX], np.array[g1_v1, ... c.DUMMY_INDEX], ...]
    np_save(f"{storage_directory}/{c.FN_ADJ_LISTS}",        adjacency_mat)       # Format: np.array:     [np.array[N(v1)_1, N(v1)_2, ... c.DUMMY_INDEX], np.array[N(v2)_1, ... c.DUMMY_INDEX], ...]
    np_save(f"{storage_directory}/{c.FN_VERTEX_LABELS}",    vertex_label_list)   # Format: List[int]:    [l(v1), l(v2), ...]
    # Save the map from the local indices (graph id, local index) to the global indices to file.    
    LOG.info(f"Process terminated successully. Files can be found in: {storage_directory}")
    
### Methods for reading and writing to file ###

def load_adj_lists(file_name: str) -> np.array:
    return np.load(file_name)

def load_vertex_labels(file_name: str) -> np.array:
    return np.load(file_name)

def load_graph_vertices(file_name: str) -> np.array:
    return np.load(file_name, allow_pickle=True)

def write_local_gl_v_id_map_to_readable_file(storage_dir:str, local_global_dict, file_name:str = "0_local_global_index_map-DEBUGGING") -> None:
    lines = [f"{gId_local_vId}\t-> {v}\n" for gId_local_vId, v in local_global_dict.items()]
    with open(f"{storage_dir}/{file_name}.txt", 'w') as f:
        f.write("# (graph id, local vertex id) -> global vertex id\n")
        f.writelines(lines)

def write_new_v_lbls_to_readable_file(storage_dir:str, vertex_labels_dict, file_name:str = "0_new_vertex_labels_map-DEBUGGING") -> None:
    if vertex_labels_dict is not None:
        lines = [f"{old}\t-> {new}\n" for old, new in vertex_labels_dict.items()]
        with open(f"{storage_dir}/{file_name}.txt", 'w') as f:
            f.write("# old_vertex_label -> new_vertex_label\n")
            f.writelines(lines)

def np_save(file_name: str, array: np.array) -> None:
    np.save(f"{file_name}", array)    

### Main, tests and execution ###

def main(dataset_name: str = "MUTAG", reduced_graph_nr: int = Union[int, List[int]], dir_suffix: str = ''):
    construct_global_adj_list_matrix(dataset_name=dataset_name, reduced_graph_nr=reduced_graph_nr, dir_suffix=dir_suffix)
    LOG.info("DatasetToGolbalAdjList terminated.")

def parse_terminal_args() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
        default=["AIDS", "PROTEINS", "NCI1", "DD", "ENZYMES", "Tox21_AHR"],
        dest='dataset_names',
        help='Provide TU Dortmund dataset names.',
        type=str,
        nargs='+') 
    args = parser.parse_args(sys.argv[1:])
    
    for dataset_name in args.dataset_names:
        main(dataset_name=dataset_name)

if __name__ == "__main__": 
    # Run in terminal as: python3 x1_dataset_to_globalAdjList.py -d MUTAG AIDS        
    parse_terminal_args()