from typing import Dict, List, Set, Tuple
import numpy as np 
import logging
import functools
import operator
import networkx as nx
import os
import zipfile
from collections import Counter
from abc import ABC, abstractmethod

# GraKel # https://ysig.github.io/GraKeL/0.1a8/documentation.html
from grakel.datasets import fetch_dataset, get_dataset_info

# OGB # Installation: https://ogb.stanford.edu/docs/home/
from ogb.graphproppred import GraphPropPredDataset

# GLOBAL DEBUGGING LOGGER for this script
from my_utils.decorators import get_logger
LOG = get_logger(filename="Loggers/dataset_loader_log.txt", level=logging.ERROR, create_file=True)
VERBOSE = False

class GraphStruct():
    # https://github.com/nd7141/graph_datasets
    def __init__(self, edges, node_labels, node_attributes, edge_labels, edge_attributes):
        '''
        Graph representation in simple python data structures
        :param edges:           list of edges,                  e.g. [(0,1), (0,2), (1,0), ...]
        :param node_labels:     dictionary of labels,           e.g. {(0: 'A', 1: 'B', ...}
        :param node_attributes: dictionary of attributes,       e.g. {(0: '0.25', 1: '0.33', ...}
        :param edge_labels:     dictionary of edge labels,      e.g. {(0,1): 'Q', (0,2): 'W', (1,0}: 'Q', ...}
        :param edge_attributes: dictionary of edge attributes,  e.g. {(0,1): '0.1', (0,2): '0.3', (1,0}: '0.1', ...}
        '''
        self.edges              = edges
        self.edge_labels        = edge_labels
        self.edge_attributes    = edge_attributes
        self.nodes              = set(functools.reduce(operator.iconcat, self.edges, []))
        self.node_labels        = {node: node_labels[node]     for node in self.nodes} if len(node_labels) else dict()
        self.node_attributes    = {node: node_attributes[node] for node in self.nodes} if len(node_attributes) else dict()

    def is_edge_label_directed(self):
        el = self.edge_labels
        for e in el:
            if el[e] != el[(e[1], e[0])]:
                return True

        return False

    def is_edge_attribute_directed(self):
        ea = self.edge_attributes
        for e in ea:
            if ea[e] != ea[(e[1], e[0])]:
                return True

        return False

    def convert_to_nx(self):
        G = nx.Graph()
        if self.is_edge_label_directed() or self.is_edge_attribute_directed():
            G = nx.DiGraph()
        G.add_edges_from(self.edges)
        nx.set_edge_attributes(G, self.edge_labels,     'edge_label')       if len(self.edge_labels) else None
        nx.set_edge_attributes(G, self.edge_attributes, 'edge_attribute')   if len(self.edge_attributes) else None
        nx.set_node_attributes(G, self.node_labels,     'node_label')       if len(self.node_labels) else None
        nx.set_node_attributes(G, self.node_attributes, 'node_attribute')   if len(self.node_attributes) else None
        return G

class GraphDataset():
    # https://github.com/nd7141/graph_datasets
    @staticmethod
    def extract_folder(zip_folder, output):
        with zipfile.ZipFile(zip_folder, 'r') as f:
            f.extractall(output)

    def get_filenames(self, input_folder):
        fns = os.listdir(input_folder)
        graphs_fn = indicator_fn = graph_labels_fn = \
            node_labels_fn = edge_labels_fn = \
            edge_attributes_fn = node_attributes_fn = graph_attributes_fn = None
        for fn in fns:
            if 'A.txt' in fn:
                graphs_fn = input_folder + fn
            elif '_graph_indicator.txt' in fn:
                indicator_fn = input_folder + fn
            elif '_graph_labels.txt' in fn:
                graph_labels_fn = input_folder + fn
            elif '_node_labels.txt' in fn:
                node_labels_fn = input_folder + fn
            elif '_edge_labels.txt' in fn:
                edge_labels_fn = input_folder + fn
            elif '_node_attributes.txt' in fn:
                node_attributes_fn = input_folder + fn
            elif '_edge_attributes.txt' in fn:
                edge_attributes_fn = input_folder + fn
            elif '_graph_attributes.txt' in fn:
                graph_attributes_fn = input_folder + fn
        return graphs_fn, indicator_fn, graph_labels_fn, node_labels_fn, edge_labels_fn, \
               edge_attributes_fn, node_attributes_fn, graph_attributes_fn

    def read_graphs(self, input_folder):
        graphs_fn, indicator_fn, graph_labels_fn, node_labels_fn, edge_labels_fn, \
        edge_attributes_fn, node_attributes_fn, graph_attributes_fn = self.get_filenames(input_folder)

        if edge_labels_fn:      edge_labels_f = open(edge_labels_fn)
        if edge_attributes_fn:  edge_attributes_f = open(edge_attributes_fn)

        with open(indicator_fn) as f:
            nodes2graph = dict()
            for i, line in enumerate(f):
                nodes2graph[i + 1] = int(line.strip())

        node_labels = dict()
        if node_labels_fn:
            with open(node_labels_fn) as f:
                for i, line in enumerate(f):
                    node_labels[i + 1] = line.strip()

        node_attributes = dict()
        if node_attributes_fn:
            with open(node_attributes_fn) as f:
                for i, line in enumerate(f):
                    node_attributes[i + 1] = line.strip()

        if graph_attributes_fn:
            graph_attributes = dict()
            with open(graph_attributes_fn) as f:
                for i, line in enumerate(f):
                    graph_attributes[i + 1] = line.strip()

        new_graphs = []
        with open(graphs_fn) as f:
            current_graph = 1
            edges = []
            edge_labels = dict()
            edge_attributes = dict()
            for i, line in enumerate(f):
                l = line.strip().split(',')
                u, v = int(l[0]), int(l[1])
                g1, g2 = nodes2graph[u], nodes2graph[v]
                assert g1 == g2, 'Nodes should be connected in the same graph. Line {}, graphs {} {}'. \
                    format(i, g1, g2)

                # Assume that indicators are sorted.
                if g1 != current_graph:
                    # print(g1, current_graph, edges)
                    G = GraphStruct(edges, node_labels, node_attributes, edge_labels, edge_attributes)

                    new_graphs.append(G)

                    edges = []
                    edge_labels = dict()
                    edge_attributes = dict()
                    current_graph += 1
                    # if current_graph % 1000 == 0: print('Finished {} dataset'.format(current_graph - 1))

                edges.append((u, v))
                if edge_labels_fn:
                    edge_labels[(u, v)] = next(edge_labels_f).strip()
                if edge_attributes_fn:
                    edge_attributes[(u, v)] = next(edge_attributes_f).strip()

        # Last graph.
        if len(edges) > 0:
            G = GraphStruct(edges, node_labels, node_attributes, edge_labels, edge_attributes)
            new_graphs.append(G)

        if edge_labels_fn:      edge_labels_f.close()
        if edge_attributes_fn:  edge_attributes_f.close()

        return new_graphs

    def read_labels(self, dataset, input_folder):
        graph_labels = dict()
        with open(input_folder + dataset + '_graph_labels.txt') as f:
            for i, label in enumerate(f):
                graph_labels[i] = label.strip()
        return graph_labels

    def read_dataset(self, dataset, input_folder):
        assert os.path.exists(input_folder), f'Path to dataset should contain folder {dataset}'
        graphs = self.read_graphs(input_folder)
        labels = self.read_labels(dataset, input_folder)
        return graphs, labels

    def compute_stats(self, graphs, labels):
        if len(graphs) > 0:
            num_nodes = [len(g.nodes) for g in graphs]
            num_edges = [len(g.edges) / 2 for g in graphs]
            c = Counter(labels.values())
            least, most = c.most_common()[-1][1], c.most_common()[0][1]
            return len(graphs), np.mean(num_nodes), np.mean(num_edges), len(c), least, most
        return 0, 0, 0, 0, 0, 0

    def convert_to_nx_graphs(self, graphs):
        return [g.convert_to_nx() for g in graphs]

    def save_graphs_graphml(self, graphs, output_folder):
        nx_graphs = self.convert_to_nx_graphs(graphs)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        for ix, g in enumerate(nx_graphs):
            nx.write_graphml(g, output_folder + f'{ix}.graphml')

    def save_graphs_edgelist(self, graphs, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        for ix, g in enumerate(graphs):
            fn = f'{ix}.edgelist'
            with open(output_folder + fn, 'w+') as f:
                for e in g.edges:
                    f.write(f"{e[0]} {e[1]}\n")

class DatasetLoader(ABC):
    HAS_V_LABELS    = 'has_vertex_labels'
    HAS_E_LABELS    = 'has_edge_labels'
    HAS_V_ATTS      = 'has_vertex_attributes'
    HAS_E_ATTS      = 'has_edge_attributes'
    G_DOWNLOAD      = 'download_link'
    G_DESCRIPTION   = 'description'

    def __init__(self):
        self.framework_name: str = "DatasetLoaderSuperClass"
        self._v_feature_label_map:      Dict[int, int]             = dict()
        self._e_feature_label_map:      Dict[Tuple[int, int], int] = dict()
        self._general_graph_datasets:   Dict[str, Tuple[np.array, np.array]] = dict()        
        self._global_to_local_map:      Dict[Tuple[int, int], int] = dict()
        
    def get_dataset_information(self, dataset_name: str) -> str:
        return self._dataset_info_dict().get(dataset_name)[self.G_DESCRIPTION]
    
    def get_vertex_feature_map(self) -> Dict[int, int]:
        return self._v_feature_label_map

    def get_edge_feature_map(self) -> Dict[Tuple[int, int], int]:
        return self._e_feature_label_map

    def get_global_to_local_map(self) -> Dict[Tuple[int, int], int]:
        return self._global_to_local_map

    def _get_max_v_feature(self) -> int:
        v_feature_ctr = 0
        if len(self._v_feature_label_map) > 0:
            v_feature_ctr = max(self._v_feature_label_map.values())
        return v_feature_ctr

    def _get_max_e_feature(self) -> int:
        e_feature_ctr = 0
        if len(self._e_feature_label_map) > 0:
            e_feature_ctr = max(self._e_feature_label_map.values())
        return e_feature_ctr

    def dataset_is_known(self, dataset_name: str) -> bool:
        return dataset_name in self._dataset_info_dict().keys()

    def print_known_datasets(self) -> None:
        print(f"The avaliable datasets of framework {self.framework_name} are:")
        print(np.array(list(self._dataset_info_dict().keys())))   

    def get_general_dataset_cleaned_graphs_and_classes(self, dataset_name: str, delete_vertexles_graphs: bool=True, delete_edgeles_graphs: bool=False, require_edge_labels: bool=True) -> Tuple[np.array, np.array]:
        """
        This function loads a general dataset (if the dataset name is known) and cleanes it by
        - deleting graphs without edges,
        - adding default vertex labels, if none are present and
        - adding default edge labels, if none are present and desired via parameter flag.
        """        
        def _delete_graphs_without_vertices_andor_edges(graphs: np.array, classes: np.array, delete_empty_vertex_set = True, delete_empty_edge_set = False) -> Tuple[np.array, np.array]:
            """
            Remove all graphs from the graphs-list, which:
            - have an empty vertex label dictionary
            """
            # Delete graphs without a vertex label dictionary.
            black_list: List[int] = list()
            class_list: np.array = classes.copy()

            for nr, g in enumerate(graphs):
                # Delete empty graphs.
                if len(g) == 0:
                    black_list += [nr]
                    continue
                # Delete graphs with no vertices.
                if delete_empty_vertex_set:
                    vertex_label_dict = g[1]
                    if len(vertex_label_dict) < 1:
                        black_list += [nr]
                        continue
                # Delete graphs with no edges.
                if delete_empty_edge_set:
                    edge_list = g[0]
                    if len(edge_list) < 1:
                        black_list += [nr]
            
            graphs     = np.delete(graphs, black_list, axis=0)
            class_list = np.delete(class_list, black_list, axis=0)
                        
            n = len(black_list)
            if VERBOSE and n > 0:
                s = f"\tDeleted {n} incomplete graphs (empty {'vertex label dictionary' if delete_empty_vertex_set else ''}{' or ' if delete_empty_edge_set and delete_empty_vertex_set else ''}{'edge set' if delete_empty_edge_set else ''})."
                LOG.debug(s)

            return graphs, class_list

        def _add_artificial_vertex_andor_edge_labels(graphs: np.array, has_vertex_labels: bool, has_edge_labels: bool) -> np.array:
            """
            Add to all general graphs from the graphs-list:
            - uniform artificial vertex labels and/or
            - uniform artificial edge labels.
            """            
            for g in graphs:
                edge_list = g[0]        
                artificial_vertex_label = 0
                artificial_edge_label = 0
                # If no vertex labels are present, create a dictionary for such vertex labels and set them all to an equal artificial label.
                if not has_vertex_labels:
                    # To do so, we need to find out how many vertices there are. Thus iterate all edges and find the highest vertex index.        
                    nr_vertices = max(max(v, w) for (v, w) in edge_list)
                    # Start the indices of the vertices at "1".
                    vertex_label_dict = dict(zip(list(range(1, nr_vertices+1)), [artificial_vertex_label]*nr_vertices))
                    # Store the vertex labels in the graph.
                    g[1] = vertex_label_dict

                # If no edge labels are present, create a dictionary for such edge labels and set them all to an equal artificial label.
                if not has_edge_labels:
                    nr_edges = len(edge_list)
                    edge_label_dict = dict(zip(edge_list, [artificial_edge_label]*nr_edges))
                    # Store the edge labels in the graph.
                    g[2] = edge_label_dict

            if VERBOSE:
                if not has_vertex_labels:
                    LOG.debug(f"\tInserted artificial vertex labels.")
                if not has_edge_labels:
                    LOG.debug(f"\tInserted artificial edge labels.")

            return graphs

        if not self.dataset_is_known(dataset_name):            
            LOG.debug(f"No dataset named {dataset_name} is known to me! If it exists, please update the dataset-dictionary!")
            return [], []
        else:
            graphs, classes = self.get_general_dataset(dataset_name)
            
            # Delete unusable graphs.
            graphs, classes = _delete_graphs_without_vertices_andor_edges(graphs, classes, delete_empty_vertex_set = delete_vertexles_graphs, delete_empty_edge_set = delete_edgeles_graphs)

            # Add vertex or edge labels if necessary. Therefore, test, what kind of data is already present.
            general_dataset_info = self.get_dataset_information(dataset_name)
            
            has_vertex_labels: bool = general_dataset_info[self.HAS_V_LABELS]
            has_edge_labels  : bool = general_dataset_info[self.HAS_E_LABELS]
            if not require_edge_labels:
                has_edge_labels = True
            
            # Add the missing data, if required.
            graphs = _add_artificial_vertex_andor_edge_labels(graphs, has_vertex_labels, has_edge_labels)

            # Delete the edge labels, if not required.
            if not require_edge_labels:
                graphs = np.delete(graphs, [2], axis=1)

            return graphs, classes

    def map_dataset_vertices_to_range(self, graphs: np.array, return_global_to_local_dict: bool = False) -> Tuple[np.array, Dict[int, int]]:
        """
        This methods takes a list of general graphs consisting of an edge set and a vertex-label dictionary.
        It iterates through all these graphs and relabels the vertex indices, such that they are unique across the whole dataset.
        It returns the relabelled list of graphs.
        The map ('graph_id', 'local_vertex_id') -> 'global_vertex_id' will be returned.
        """
        # Initialize the first global vertex id as '0' and then iterate through the dataset.
        # Often, in general databases, the indices are globalized/unique but start from '1'.
        # Starting from zero gives the benefit, that for example all labels can be stored in a vector and the index indicates the vertex id, 
        # without having the zeroth index as invalid entry. All indices will be positive integers.
        global_vertex_id: int = 0
        self._global_to_local_map: Dict[Tuple[int, int], int] = dict()
        new_graphs = list()

        for g_id, g in enumerate(graphs): # graphs: np.array[Tuple[Set, Dict, Dict]]
            # Map the vertex ids of this graph to the next gloabel indices.
            vertex_ids = list(g[1].keys())
            graphs_global_indices: Dict[Tuple[int, int], int] = range(global_vertex_id, global_vertex_id + len(vertex_ids) + 1)
            graphs_global_to_local_map = dict(zip(vertex_ids, graphs_global_indices))
            # Increase the 'global_vertex_id' such that its value is the id of the first vertex in the next graph.
            global_vertex_id = global_vertex_id + len(vertex_ids)
            
            # Rename all vertex labels in the graph. This includes two steps:
            # First: Rename the vertex labels in the edge definitions.
            new_edges = set()
            for u, v in g[0]: new_edges.add((graphs_global_to_local_map.get(u), graphs_global_to_local_map.get(v)))
            # Second: Rename the vertex labels in the vertex label dictionary.
            new_vertex_label_dict = dict()
            for u, label in g[1].items(): new_vertex_label_dict[graphs_global_to_local_map.get(u)] = label
            # Finally: Update the graph with the modified data.
            new_graph = np.array([new_edges, new_vertex_label_dict])
            new_graphs += [new_graph]

            # Merge the newly defined vertex label map the the global map.
            if return_global_to_local_dict:
                for local_id, global_id in graphs_global_to_local_map.items(): self._global_to_local_map[(g_id, local_id)] = global_id
        
        return np.array(new_graphs), self._global_to_local_map
    
    def map_graphs_vertex_indices_to_range(self, graphs: np.array) -> np.array:
        """
            This methods takes a list of (GraKel) graphs consisting of an edge set and a vertex-label dictionary.
            It iterates through all these graphs and checks, if the vertex indices in each graph are an 
            ascending list of integers, starting at zero.
            If this is not the case, it relabels the vertex indices in these graphs and returns a graph list
            with such updated indices.
            Example:
            INPUT:[
                [{(1, 2), (2, 9)}, {1: 10, 2: 20, 9: 30}],
                [{(4566, 4590), (50000, 18)}, {18: 10, 4566: 20, 4590: 30, 50000: 40}]
            ]
            OUTPUT: [
                [{(0, 1), (1, 2)}, {0: 10, 1: 20, 2: 30}],
                [{(1, 2), (3, 0)}, {0: 10, 1: 20, 2: 30, 3: 40}]
            ]
        """
        new_graphs = []

        for g in graphs:  # graphs: np.array[Tuple[Set, Dict, Dict]]
            # Get the number of vertices.
            n = len(g[1])
            # Check, if the vertex ids represent the interval [0, n].
            vertex_ids = np.sort(list(g[1].keys()))
            if set(vertex_ids) == set(range(n)):
                new_graphs += [g]
                continue
            # Otherwise...
            else:
                # Rename the vertex labels such that they do. Preserve the order among them.
                new_vertex_id = dict(zip(vertex_ids, range(n)))
                # Rename the vertex labels in the edge definitions.
                new_edges = set()
                for u, v in g[0]:
                    new_edges.add((new_vertex_id.get(u), new_vertex_id.get(v)))
                # Rename the vertex labels in the vertex label dictionary.
                new_vertex_label_dict = dict()
                for k, label in g[1].items():
                    new_vertex_label_dict[new_vertex_id[k]] = label
                # Update the graph                
                new_graph = np.array([new_edges, new_vertex_label_dict])
                new_graphs += [new_graph]

        return np.array(new_graphs)
    
    def map_vertex_labels_to_range(self, graphs: List[Tuple[Set, Dict, Dict]]) -> Tuple[List[Tuple[Set, Dict, Dict]], Dict[int, int]]:
        """
        This methods takes a list of (GraKel) graphs consisting of an edge set and a vertex-label dictionary.
        It iterates through all these graphs and relabels the vertex labels, such that they perfectly represent an interval [0, m].
        It returns the relabelled list of graphs.
        If any label had to be changed, the map 'vertex_label' -> 'new_vertex_label' will be returned. Otherwise 'None'.
        """
        # First, we have to investigate what labels do occur in the whole dataset.
        vertex_label_set = set()
        for g in graphs: vertex_label_set.update(g[1].values())

        # Check if the vertex labels are already as desired (representing an interval [0, m]).
        sorted_vertex_labels = np.sort(list(vertex_label_set))
        m = len(sorted_vertex_labels)
        labels_already_perfect: bool = False
        labels_already_perfect = (sorted_vertex_labels.min() == 0) and (sorted_vertex_labels.max() == m-1)

        if labels_already_perfect:
            return graphs, None
        else:
            vertex_labels_old_to_new = dict(zip(sorted_vertex_labels, range(m)))
            
            for g in graphs:
                # Relabel the vertices in the vertex label dictionary.
                new_vertex_label_dict = dict()
                for v, old_label in g[1].items(): new_vertex_label_dict[v] = vertex_labels_old_to_new.get(old_label)        
        
        return graphs, vertex_labels_old_to_new

    # Methods to implement.
    @abstractmethod
    def _dataset_info_dict() -> Dict[str, str]:
        pass

    @abstractmethod
    def get_dataset_information(self, dataset_name: str) -> str:
        """
        Returns a dictionary with keys:
        - self.HAS_V_LABELS    = 'has_vertex_labels'        : bool
        - self.HAS_E_LABELS    = 'has_edge_labels'          : bool
        - self.HAS_V_ATTS      = 'has_vertex_attributes'    : bool
        - self.HAS_E_ATTS      = 'has_edge_attributes'      : bool
        - self.G_DOWNLOAD      = 'download_link'            : str
        - self.G_DESCRIPTION   = 'description'              : str
        """
        pass

    @abstractmethod
    def get_general_dataset(self, dataset_name: str) -> Tuple[np.array, np.array]:
        """
        Returns list of graphs, with structure:
        0: EdgeSet                  = Set[Tuple[int, int]]
        1: Vertex Label Dictionary  = Dict[int, int]
        2: Edge Label Dictionary    = Dict[Tuple[int, int], int]

        graphs: [{EdgeSet}, {VertexId: VertexLabel}, {Edge: EdgeLabel}]
        """
        pass


class GraKelLoader(DatasetLoader):
    
    def __init__(self):        
        super().__init__()
        self.framework_name: str = "GraKel TU Dortmund"
    
    ### Implementation of the abstract methods. ###

    def _dataset_info_dict(self):
        GRAKEL_DATASETS: Dict[str, str] = {
            # https://chrsmrrs.github.io/datasets/docs/datasets/
            # Small molecules
            "AIDS": "TU Dortmund - Small molecules\n'grakel.datasets.fetch_dataset('AIDS')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            "BZR": "TU Dortmund - Small molcules\n'grakel.datasets.fetch_dataset('BZR')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "BZR_MD": "TU Dortmund - Small molcules\n'grakel.datasets.fetch_dataset('BZR_MD')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
                
            "DHFR": "TU Dortmund - Small molecules\n'grakel.datasets.fetch_dataset('DHFR')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "DHFR_MD": "TU Dortmund - Small molecules\n'grakel.datasets.fetch_dataset('DHFR_MD')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
                
            "ER_MD": "TU Dortmund - Small molecules - enzyme membership prediction\n'grakel.datasets.fetch_dataset('ER_MD')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            
            
            "MUTAG": "TU Dortmund - Small molecules\n'grakel.datasets.fetch_dataset('MUTAG')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "Mutagenicity": "TU Dortmund - Small molcules\n'grakel.datasets.fetch_dataset('Mutagenicity')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            "NCI1": "TU Dortmund - Small molecules\n'grakel.datasets.fetch_dataset('NCI1')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "NCI109": "TU Dortmund - Small molecules\n'grakel.datasets.fetch_dataset('NCI109')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            
            "Tox21_AHR": "TU Dortmund - Small molcules\n'grakel.datasets.fetch_dataset('Tox21_AHR')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "Tox21_AR-LBD": "TU Dortmund - Small molcules\n'grakel.datasets.fetch_dataset('Tox21_AR-LBD')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "Tox21_ARE": "TU Dortmund - Small molcules\n'grakel.datasets.fetch_dataset('Tox21_ARE')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
                
            # Bioinformatics
            "DD": "TU Dortmund - Bioinformatics dataset\n'grakel.datasets.fetch_dataset('DD')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            "ENZYMES": "TU Dortmund - Bioinformatics dataset - enzyme membership prediction\n'grakel.datasets.fetch_dataset('ENZYMES')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            # UNSUPPORTED
            # "KKI": "TU Dortmund - Bioinformatics dataset\n'grakel.datasets.fetch_dataset('KKI')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            # UNSUPPORTED
            # "OHSU": "TU Dortmund - Bioinformatics dataset\n'grakel.datasets.fetch_dataset('OHSU')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            
            # UNSUPPORTED - partially?
            # "Peking_1": "TU Dortmund - Bioinformatics dataset\n'grakel.datasets.fetch_dataset('Peking_1')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "PROTEINS": "TU Dortmund - Bioinformatics dataset - enzyme membership prediction\n'grakel.datasets.fetch_dataset('PROTEINS')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "PROTEINS_full": "TU Dortmund - Bioinformatics dataset - enzyme membership prediction\n'grakel.datasets.fetch_dataset('PROTEINS_full')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            # Computer vision
            "COIL-RAG": "TU Dortmund - Computer vision - segmented images of objects\n'grakel.datasets.fetch_dataset('COIL-RAG')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "COIL-DEL": "TU Dortmund - Computer vision - segmented images of objects\n'grakel.datasets.fetch_dataset('COIL-DEL')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            "Letter-high": "TU Dortmund - Computer vision - Capital letter of the roman alphabet\n'grakel.datasets.fetch_dataset('Letter-high')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "Letter-low": "TU Dortmund - Computer vision - Capital letter of the roman alphabet\n'grakel.datasets.fetch_dataset('Letter-low')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "Letter-med": "TU Dortmund - Computer vision - Capital letter of the roman alphabet\n'grakel.datasets.fetch_dataset('Letter-med')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            "MSRC_21": "TU Dortmund - Computer vision\n'grakel.datasets.fetch_dataset('MSRC_21')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "MSRC_21C": "TU Dortmund - Computer vision\n'grakel.datasets.fetch_dataset('MSRC_21C')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "MSRC_9": "TU Dortmund - Computer vision\n'grakel.datasets.fetch_dataset('MSRC_9')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            # Social networks
            "COLLAB": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('COLLAB')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            # UNSUPPORTED - partially?
            # "facebook_ct1": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('facebook_ct1')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            "IMDB-BINARY": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('IMDB-BINARY')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "IMDB-MULTI": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('IMDB-MULTI')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",       
            
            "PTC_FM": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('PTC_FM')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "PTC_FR": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('PTC_FM')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "PTC_MM": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('PTC_FM')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "PTC_MR": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('PTC_FM')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            "REDDIT-BINARY": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('REDDIT-BINARY')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            # UNSUPPORTED - partially?
            # "REDDIT-MULTI-12k": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('REDDIT-MULTI-12k')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            # UNSUPPORTED - partially?
            # "REDDIT-MULTI-5k": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('REDDIT-MULTI-5k')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            # UNSUPPORTED - partially?
            # "twitch_egos": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('twitch_egos')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",        

            # Sznthetic
            "SYNTHETIC": "TU Dortmund - Synthetic\n'grakel.datasets.fetch_dataset('SYNTHETIC')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "SYNTHETICnew": "TU Dortmund - Synthetic\n'grakel.datasets.fetch_dataset('SYNTHETICnew')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "Synthie": "TU Dortmund - Synthetic\n'grakel.datasets.fetch_dataset('Synthie')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets"
        }

        return GRAKEL_DATASETS

    def get_dataset_information(self, dataset_name: str) -> str:
        """
        Returns a dictionary with keys:
        - self.HAS_V_LABELS    = 'has_vertex_labels'        : bool
        - self.HAS_E_LABELS    = 'has_edge_labels'          : bool
        - self.HAS_V_ATTS      = 'has_vertex_attributes'    : bool
        - self.HAS_E_ATTS      = 'has_edge_attributes'      : bool
        - self.G_DOWNLOAD      = 'download_link'            : str
        - self.G_DESCRIPTION   = 'description'              : str
        """        
        grakel_info = get_dataset_info(dataset_name, default=None)

        info_dict = dict()
        info_dict[self.HAS_V_LABELS]    = grakel_info['nl']
        info_dict[self.HAS_E_LABELS]    = grakel_info['el']
        info_dict[self.HAS_V_ATTS]      = grakel_info['na']
        info_dict[self.HAS_E_ATTS]      = grakel_info['ea']
        info_dict[self.G_DOWNLOAD]      = grakel_info['link']
        info_dict[self.G_DESCRIPTION]   = self._dataset_info_dict().get(dataset_name)

        return info_dict

    def get_general_dataset(self, dataset_name: str) -> Tuple[np.array, np.array]:
        if self._general_graph_datasets.get(dataset_name) is None:
            grakel_dataset = fetch_dataset(dataset_name, verbose=False)
            graphs  = np.array(grakel_dataset.data)
            classes = np.array(grakel_dataset.target)
            self._general_graph_datasets[dataset_name] = graphs, classes

        return self._general_graph_datasets.get(dataset_name)


class OGBLoader(DatasetLoader):

    def __init__(self):        
        super().__init__()
        self.framework_name: str = "OGB"
    
    ### Local helper methods. ###

    def _is_undirected(self, edges: List[Tuple[int, int]]) -> Tuple[bool, np.array, np.array]:
        """ Test if a given list of edges denotes an undirected graph, by containing every edge twice.        
        Return:
        - is the edge list undirected, 
        - the list of (unique undirected) edges and 
        - the list of all occuring vertices.
        """
        is_undirected: bool = True
        undir_edges:    List[Tuple[int, int]] = list()        
        return_edges:   List[Tuple[int, int]] = list()
        vertices = set()

        for e in edges:
            if (e[1], e[0]) not in edges:
                is_undirected = False
            if e[0] < e[1]: undir_edges.append(e)            
            
            vertices.add(e[0])
            vertices.add(e[1])

        if not is_undirected:
            return_edges = edges
        else:
            return_edges = undir_edges

        return is_undirected, np.array(return_edges), np.array(list(vertices))

    def _map_vertex_feature_to_int_label(self, v_features_high_dim: np.array) -> np.array:
        """
        Takes a matrix of vertex features, where the features of vertex i are stored in row i.
        Concatenates this feature representation into a string-identifyer and maps it to a single integer (label).
        Returns a list of these labels.
        """
        feature_ctr = self._get_max_v_feature()
        v_features = np.array([None] * len(v_features_high_dim))

        for i, feature in enumerate(v_features_high_dim):
            feature_str = ",".join(str(f) for f in feature)
            # If this feature is unknown, add it to the dictionary.
            if self._v_feature_label_map.get(feature_str) is None:
                feature_ctr += 1
                self._v_feature_label_map[feature_str] = feature_ctr
            # At this point the feature is in the dictionary. Retrieve its assigned label.
            v_features[i] = self._v_feature_label_map[feature_str]
        
        return v_features

    def _map_edge_feature_to_int_label(self, e_features_high_dim: np.array) -> np.array:
        """
        Takes a matrix of edge features, where the features of edge i are stored in row i.
        Concatenates this feature representation into a string-identifyer and maps it to a single integer (label).
        Returns a list of these labels.
        """
        feature_ctr = self._get_max_e_feature()
        e_features = np.array([None] * len(e_features_high_dim))

        for i, feature in enumerate(e_features_high_dim):
            feature_str = ",".join(str(f) for f in feature)
            # If this feature is unknown, add it to the dictionary.
            if self._e_feature_label_map.get(feature_str) is None:
                feature_ctr += 1
                self._e_feature_label_map[feature_str] = feature_ctr
            # At this point the feature is in the dictionary. Retrieve its assigned label.
            e_features[i] = self._e_feature_label_map[feature_str]
        
        return e_features
    
    def _convert_ogb_to_grakel_graphs(self, ogb_graphs: np.array) -> List[Tuple[Set[Tuple[int, int]], Dict[int, int], Dict[int, int]]]:
        EDGE_ID     = 'edge_index'
        EDGE_FT     = 'edge_feat'
        VERTEX_FT   = 'node_feat'
        NUM_V       = 'num_nodes'

        n = ogb_graphs.shape[0]
        grakel_graphs = list()

        for g_id, g in enumerate(ogb_graphs):
            print(f"\rGraph nr: {g_id}/{n}\t[{round(g_id/n * 100, 2)}%]", end="")
            # Extract the data from the ogb graph.
            edge_cols           = g[EDGE_ID]
            edge_features       = g[EDGE_FT]
            vertex_features     = g[VERTEX_FT]
            
            directed_edges: List[Tuple[int, int]] = list(zip(edge_cols[0], edge_cols[1]))
            vertices        : np.array = np.unique(np.array([v for v in np.append(edge_cols[0], edge_cols[1])]))
            vertex_labels   : np.array = self._map_vertex_feature_to_int_label(vertex_features)
            edge_labels     : np.array = self._map_edge_feature_to_int_label(edge_features)

            # Assemble the GraKel graph equivalent.            
            edge_set     = set(directed_edges)
            v_label_dict = dict(zip(vertices, vertex_labels))
            e_label_dict = dict(zip(directed_edges, edge_labels))

            grakel_graph  = np.array([edge_set, v_label_dict, e_label_dict])
            grakel_graphs += [grakel_graph]
        
        print("\r")
        return np.array(grakel_graphs)
    
    ### Implementation of the abstract methods. ###

    def _dataset_info_dict(self):
        OGB_DATASETS: Dict[str, str] = {        
            # https://ogb.stanford.edu/docs/graphprop/
            # Graph classification
            "ogbg-molbace"      : "",
            "ogbg-molbbbp"      : "",
            "ogbg-molclintox"   : "",
            "ogbg-molmuv"       : "",
            "ogbg-molpcba"      : "\n https://ogb.stanford.edu/docs/graphprop/",
            "ogbg-molsider"     : "",
            "ogbg-moltox21"     : "",
            "ogbg-moltoxcast"   : "",
            "ogbg-molhiv"       : "OGB small scale molecular property prediction dataset 'ogbg-molhiv'\n https://ogb.stanford.edu/docs/graphprop/",
            "ogbg-molesol"      : "",
            "ogbg-molfreesolv"  : "",
            "ogbg-mollipo"      : "",
            "ogbg-molchembl"    : "",
            "ogbg-ppa"          : "\n https://ogb.stanford.edu/docs/graphprop/",            
            "ogbg-code2"        : "\n https://ogb.stanford.edu/docs/graphprop/"
        }
        return OGB_DATASETS

    def get_dataset_information(self, dataset_name: str) -> str:
        """
        Returns a dictionary with keys:
        - self.HAS_V_LABELS    = 'has_vertex_labels'        : bool
        - self.HAS_E_LABELS    = 'has_edge_labels'          : bool
        - self.HAS_V_ATTS      = 'has_vertex_attributes'    : bool
        - self.HAS_E_ATTS      = 'has_edge_attributes'      : bool
        - self.G_DOWNLOAD      = 'download_link'            : str
        - self.G_DESCRIPTION   = 'description'              : str
        """
        dataset = GraphPropPredDataset(name=dataset_name)
        
        info_dict = dict()
        info_dict[self.HAS_V_LABELS]    = dataset.meta_info['has_node_attr'] # The ND-attributes are interpreted as 1D-labels.
        info_dict[self.HAS_E_LABELS]    = dataset.meta_info['has_node_attr'] # The ND-attributes are interpreted as 1D-labels.
        info_dict[self.HAS_V_ATTS]      = dataset.meta_info['has_node_attr']
        info_dict[self.HAS_E_ATTS]      = dataset.meta_info['has_edge_attr']
        info_dict[self.G_DOWNLOAD]      = dataset.meta_info['url']
        info_dict[self.G_DESCRIPTION]   = self._dataset_info_dict().get(dataset_name)

        return info_dict

    def get_general_dataset(self, dataset_name: str) -> Tuple[np.array, np.array]:
        if self._general_graph_datasets.get(dataset_name) is None:
            ogb_dataset = GraphPropPredDataset(name=dataset_name)
            graphs  = np.array(ogb_dataset.graphs)
            classes = np.array(ogb_dataset.labels)
            graphs = self._convert_ogb_to_grakel_graphs(graphs)

            self._general_graph_datasets[dataset_name] = graphs, classes
            
        return self._general_graph_datasets.get(dataset_name)


class CleanGraKelLoader(DatasetLoader):
    # https://github.com/nd7141/graph_datasets
    
    def __init__(self):        
        super().__init__()
        self.framework_name: str = "Cleaned GraKel TU Dortmund"
    
    ### Implementation of the abstract methods. ###

    def _dataset_info_dict(self):
        ds = ['FIRSTMM_DB',
        'OHSU',
        'KKI',
        'Peking_1',
        'MUTAG',
        'MSRC_21C',
        'MSRC_9',
        'Cuneiform',
        'SYNTHETIC',
        'COX2_MD',
        'BZR_MD',
        'PTC_MM',
        'PTC_MR',
        'PTC_FM',
        'PTC_FR',
        'DHFR_MD',
        'Synthie',
        'BZR',
        'ER_MD',
        'COX2',
        'MSRC_21',
        'ENZYMES',
        'DHFR',
        'IMDB-BINARY',
        'PROTEINS',
        'DD',
        'IMDB-MULTI',
        'AIDS',
        'REDDIT-BINARY',
        'Letter-high',
        'Letter-low',
        'Letter-med',
        'Fingerprint',
        'COIL-DEL',
        'COIL-RAG',
        'NCI1',
        'NCI109',
        'FRANKENSTEIN',
        'Mutagenicity',
        'REDDIT-MULTI-5K',
        'COLLAB',
        'Tox21_ARE',
        'Tox21_aromatase',
        'Tox21_MMP',
        'Tox21_ER',
        'Tox21_HSE',
        'Tox21_AHR',
        'Tox21_PPAR-gamma',
        'Tox21_AR-LBD',
        'Tox21_p53',
        'Tox21_ER_LBD',
        'Tox21_ATAD5',
        'Tox21_AR',
        'REDDIT-MULTI-12K']
        CLEANED_GRAKEL_DATASETS: Dict[str, str] = {
            # https://chrsmrrs.github.io/datasets/docs/datasets/
            # Small molecules
            "AIDS_c": "TU Dortmund - Small molecules\n'grakel.datasets.fetch_dataset('AIDS')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/AIDS.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            "BZR_c": "TU Dortmund - Small molcules\n'grakel.datasets.fetch_dataset('BZR')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/BZR_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "BZR_MD_c": "TU Dortmund - Small molcules\n'grakel.datasets.fetch_dataset('BZR_MD')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/BZR_MD_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
                
            "DHFR_c": "TU Dortmund - Small molecules\n'grakel.datasets.fetch_dataset('DHFR')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/DHFR_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "DHFR_MD_c": "TU Dortmund - Small molecules\n'grakel.datasets.fetch_dataset('DHFR_MD')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/DHFR_MD_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
                
            "ER_MD_c": "TU Dortmund - Small molecules - enzyme membership prediction\n'grakel.datasets.fetch_dataset('ER_MD')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/ER_MD_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            
            
            "MUTAG_c": "TU Dortmund - Small molecules\n'grakel.datasets.fetch_dataset('MUTAG')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/MUTAG_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "Mutagenicity_c": "TU Dortmund - Small molcules\n'grakel.datasets.fetch_dataset('Mutagenicity')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/Mutagenicity_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            "NCI1_c": "TU Dortmund - Small molecules\n'grakel.datasets.fetch_dataset('NCI1')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/NCI1_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "NCI109_c": "TU Dortmund - Small molecules\n'grakel.datasets.fetch_dataset('NCI109')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/NCI109_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            
            "Tox21_AHR_c": "TU Dortmund - Small molcules\n'grakel.datasets.fetch_dataset('Tox21_AHR')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/Tox21_AHR_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "Tox21_AR-LBD_c": "TU Dortmund - Small molcules\n'grakel.datasets.fetch_dataset('Tox21_AR-LBD')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/Tox21_AR-LBD_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "Tox21_ARE_c": "TU Dortmund - Small molcules\n'grakel.datasets.fetch_dataset('Tox21_ARE')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/Tox21_ARE_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
                
            # Bioinformatics
            "DD_c": "TU Dortmund - Bioinformatics dataset\n'grakel.datasets.fetch_dataset('DD')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/DD_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            "ENZYMES_c": "TU Dortmund - Bioinformatics dataset - enzyme membership prediction\n'grakel.datasets.fetch_dataset('ENZYMES')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/ENZYMES_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            # UNSUPPORTED
            # "KKI_c": "TU Dortmund - Bioinformatics dataset\n'grakel.datasets.fetch_dataset('KKI')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/KKI_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            # UNSUPPORTED
            # "OHSU_c": "TU Dortmund - Bioinformatics dataset\n'grakel.datasets.fetch_dataset('OHSU')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/OHSU_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            
            # UNSUPPORTED - partially?
            # "Peking_1": "TU Dortmund - Bioinformatics dataset\n'grakel.datasets.fetch_dataset('Peking_1')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/Peking_1.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "PROTEINS_c": "TU Dortmund - Bioinformatics dataset - enzyme membership prediction\n'grakel.datasets.fetch_dataset('PROTEINS')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/PROTEINS_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "PROTEINS_full_c": "TU Dortmund - Bioinformatics dataset - enzyme membership prediction\n'grakel.datasets.fetch_dataset('PROTEINS_full')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/PROTEINS_full_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            # Computer vision
            "COIL-RAG_c": "TU Dortmund - Computer vision - segmented images of objects\n'grakel.datasets.fetch_dataset('COIL-RAG')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/COIL-RAG_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "COIL-DEL_c": "TU Dortmund - Computer vision - segmented images of objects\n'grakel.datasets.fetch_dataset('COIL-DEL')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/COIL-DEL_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            "Letter-high_c": "TU Dortmund - Computer vision - Capital letter of the roman alphabet\n'grakel.datasets.fetch_dataset('Letter-high')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/Letter-high_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "Letter-low_c": "TU Dortmund - Computer vision - Capital letter of the roman alphabet\n'grakel.datasets.fetch_dataset('Letter-low')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/Letter-low_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "Letter-med_c": "TU Dortmund - Computer vision - Capital letter of the roman alphabet\n'grakel.datasets.fetch_dataset('Letter-med')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/Letter-med_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            "MSRC_21_c": "TU Dortmund - Computer vision\n'grakel.datasets.fetch_dataset('MSRC_21')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/MSRC_21_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "MSRC_21C_c": "TU Dortmund - Computer vision\n'grakel.datasets.fetch_dataset('MSRC_21C')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/MSRC_21C_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "MSRC_9_c": "TU Dortmund - Computer vision\n'grakel.datasets.fetch_dataset('MSRC_9')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/MSRC_9_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            # Social networks
            "COLLAB_c": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('COLLAB')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/COLLAB_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            # UNSUPPORTED - partially?
            # "facebook_ct1_c": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('facebook_ct1')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/facebook_ct1_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            "IMDB-BINARY_c": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('IMDB-BINARY')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/IMDB-BINARY_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "IMDB-MULTI_c": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('IMDB-MULTI')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/IMDB-MULTI_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",       
            
            "PTC_FM_c": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('PTC_FM')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/PTC_FM_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "PTC_FR_c": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('PTC_FM')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/PTC_FR_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "PTC_MM_c": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('PTC_FM')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/PTC_MM_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "PTC_MR_c": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('PTC_FM')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/PTC_MR_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            "REDDIT-BINARY_c": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('REDDIT-BINARY')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/REDDIT-BINARY_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            # UNSUPPORTED - partially?
            # "REDDIT-MULTI-12k_c": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('REDDIT-MULTI-12k')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/REDDIT-MULTI-12k_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            # UNSUPPORTED - partially?
            # "REDDIT-MULTI-5k_c": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('REDDIT-MULTI-5k')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/REDDIT-MULTI-5k_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            # UNSUPPORTED - partially?
            # "twitch_egos_c": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('twitch_egos')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/twitch_egos_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",        

            # Sznthetic
            "SYNTHETIC_c": "TU Dortmund - Synthetic\n'grakel.datasets.fetch_dataset('SYNTHETIC')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/SYNTHETIC_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "SYNTHETICnew_c": "TU Dortmund - Synthetic\n'grakel.datasets.fetch_dataset('SYNTHETICnew')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/SYNTHETICnew_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "Synthie_c": "TU Dortmund - Synthetic\n'grakel.datasets.fetch_dataset('Synthie')'\nhttps://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/Synthie_c.zip \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets"
        }

        return CLEANED_GRAKEL_DATASETS

    def get_dataset_information(self, dataset_name: str) -> str:
        """
        Returns a dictionary with keys:
        - self.HAS_V_LABELS    = 'has_vertex_labels'        : bool
        - self.HAS_E_LABELS    = 'has_edge_labels'          : bool
        - self.HAS_V_ATTS      = 'has_vertex_attributes'    : bool
        - self.HAS_E_ATTS      = 'has_edge_attributes'      : bool
        - self.G_DOWNLOAD      = 'download_link'            : str
        - self.G_DESCRIPTION   = 'description'              : str
        """
        info_dict = dict()
        info_dict[self.HAS_V_LABELS]    = True
        info_dict[self.HAS_E_LABELS]    = True      
        info_dict[self.HAS_V_ATTS]      = False     # Ignored by '_convert_graphstruct_to_general_graph()'
        info_dict[self.HAS_E_ATTS]      = False     # Ignored by '_convert_graphstruct_to_general_graph()'
        info_dict[self.G_DOWNLOAD]      = self._dataset_info_dict().get(dataset_name)
        info_dict[self.G_DESCRIPTION]   = self._dataset_info_dict().get(dataset_name)

        return info_dict

    def _convert_graphstruct_to_general_graph(self, graphstruct: GraphStruct):        
        edge_set     = graphstruct.edges
        v_label_dict = {k: int(v) for k, v in graphstruct.node_labels.items()}
        e_label_dict = {k: int(v) for k, v in graphstruct.edge_labels.items()}
        grakel_graph  = np.array([edge_set, v_label_dict, e_label_dict])            
        return grakel_graph

    def get_general_dataset(self, dataset_name: str) -> Tuple[np.array, np.array]:
        dataset_name = dataset_name.replace("_c", "")
        if self._general_graph_datasets.get(dataset_name) is None:
            dataset = GraphDataset()
            # Extract the dataset.
            cleaned_dataset_path = "/home/fabrice/Documents/Uni/15. Semester/CleanedTUDatasets/graph_datasets/datasets"
            input_dir            = "compact"
            dataset.extract_folder(f"{cleaned_dataset_path}/{dataset_name}.zip", f"{input_dir}/")
            # Read the graphs. 
            graph_structs, classes_dict = dataset.read_dataset(dataset_name, f"{input_dir}/{dataset_name}/")
            # Convert the graph structs to a list of general graphs.
            graphs = [self._convert_graphstruct_to_general_graph(g) for g in graph_structs]
            # Convert the classes dict to an array of classes.
            sorted_keys = sorted(classes_dict.keys())
            classes: np.array = np.array([classes_dict[k] for k in sorted_keys], dtype=np.int)
            # Save the constructed dataset of general graphs.
            self._general_graph_datasets[dataset_name] = graphs, classes
            
        return self._general_graph_datasets.get(dataset_name)

def get_suitable_dataloader(dataset_name: str) -> DatasetLoader:
    known_loaders: List[DatasetLoader] = [GraKelLoader(), OGBLoader(), CleanGraKelLoader()]

    for loader in known_loaders:
        if loader.dataset_is_known(dataset_name):
            return loader

    print(f"None of the known loaders knows the dataset '{dataset_name}'!")
    print("Known datasets are:")
    for loader in known_loaders:
        loader.print_known_datasets()
        print()
    
    return None

# TESTS #

def _test_dataset_cleaning(dataset_name: str = "MUTAG"):
    print(f"Test dataset cleaning for dataset '{dataset_name}'")
    loader = get_suitable_dataloader(dataset_name)
    
    # Get not cleaned dataset.
    graphs, _ = loader.get_general_dataset(dataset_name)

    selected_test_graph_ids = [0, 50, 100]
    graphs_before_cleaning = [graphs[g_id] for g_id in selected_test_graph_ids]

    # Get cleaned dataset.
    graphs, _ = loader.get_general_dataset_cleaned_graphs_and_classes(dataset_name, delete_vertexles_graphs = True, delete_edgeles_graphs = False, require_edge_labels=False)
    
    for i, g_id in enumerate(selected_test_graph_ids):
        print(f"\n\nBEFORE CLEARNING GRAPH {g_id} (dataset '{dataset_name}'):")
        print("\n".join([f"{entry}" for entry in graphs_before_cleaning[i]]))
        print(f"\nAFTER CLEARNING GRAPH {g_id} (dataset '{dataset_name}'):")
        print("\n".join([f"{entry}" for entry in graphs[g_id]]))
        print()
    
def _test_dataset_vertices_map(dataset_name: str = "MUTAG"):
    print(f"Test dataset vertices map for dataset '{dataset_name}'")
    loader = get_suitable_dataloader(dataset_name)
    graphs, _ = loader.get_general_dataset_cleaned_graphs_and_classes(dataset_name, delete_vertexles_graphs = True, delete_edgeles_graphs = False, require_edge_labels=False)
    
    selected_test_graph_ids = [0, 50, 100]
    graphs_before_vertex_mapping = [graphs[g_id] for g_id in selected_test_graph_ids]
    
    graphs, map = loader.map_dataset_vertices_to_range(graphs, return_global_to_local_dict = True)
    
    for i, g_id in enumerate(selected_test_graph_ids):
        print(f"\n\nBEFORE VERTEX MAPPING GRAPH {g_id} (dataset '{dataset_name}'):")
        print("\n".join([f"{entry}\n" for entry in graphs_before_vertex_mapping[i]]))
        print(f"\nAFTER VERTEX MAPPING GRAPH {g_id} (dataset '{dataset_name}'):")
        print("\n".join([f"{entry}" for entry in graphs[g_id]]))
        print()
        
    print(map)

def _test_methods():
    # _test_dataset_cleaning("MUTAG")             # PASSED
    # _test_dataset_cleaning("ogbg-molhiv")       # PASSED
    # _test_dataset_cleaning("MUTAG_c")           # PASSED

    # _test_dataset_vertices_map("MUTAG")         # PASSED
    # _test_dataset_vertices_map("ogbg-molhiv")   # PASSED
    # _test_dataset_vertices_map("MUTAG_c")       # PASSED
    pass

if __name__ == "__main__":    
    # _test_methods()
    pass