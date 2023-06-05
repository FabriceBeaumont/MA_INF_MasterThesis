import numpy as np
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

class EdgeRepresentation(ABC):

    edge_labelled: bool = False

    @abstractmethod
    def exists_edge(self, v: int, w) -> bool:
        """Returns true, if there exists a directed edge (v,w) or an undirected edge {v,w}."""

    @abstractmethod
    def get_edge_label(self, v: int, w: int) -> int:
        """
        Returns 'None', if there is no such edge (v,w) or {v,w}.
        Returns (int)'-1' if no label is known, and the label otherwise.
        """
    
    @abstractmethod
    def get_neighborhood(self, v: int) -> List[int]:
        """Returns the vertex ids of neighbors of 'v'."""

    @abstractmethod
    def get_edge_representation(self) -> Any:
        """
        Returns the edge representation, in the specific format
        which the instance handles.
        """


class AdjacencyMatrix(EdgeRepresentation):

    def __init__(self, adjacency_matrix: np.ndarray):
        self.adjacency_matrix: np.ndarray = adjacency_matrix
    
    def exists_edge(self, v, w) -> bool:
        return self.get_edge_label(v, w) is not None
        
    def get_edge_label(self, v, w) -> Optional[int]: 
        if v <= self.adjacency_matrix.shape[0] and w <= self.adjacency_matrix.shape[1]:
            entry = self.adjacency_matrix[v][w]
            if entry == 0:
                return None
            else:
                return entry

    def get_neighborhood(self, v) -> List[int]:        
        return list(np.nonzero(self.adjacency_matrix[v])[0])
    
    def get_edge_representation(self) -> np.ndarray:
        return self.adjacency_matrix

    def __repr__(self) -> None:
        return f"Adjacency matrix representation:\n{self.adjacency_matrix}"

class AdjacencyList(EdgeRepresentation):

    def __init__(self, neighbor_ids_dict: Dict[int, List[int]]):
        self.neighbor_ids_dict: Dict[int, List[int]] = neighbor_ids_dict
    
    def exists_edge(self, v, w) -> bool:
        if self.neighbor_ids_dict.get(v):
            return w in self.neighbor_ids_dict.get(v)
        else:
            return False
        
    def get_edge_label(self, v, w):
        return -1
    
    def get_neighborhood(self, v):
        return self.neighbor_ids_dict.get(v)
    
    def get_edge_representation(self):
        return self.neighbor_ids_dict

    def __repr__(self):
        return f"Adjacency list representation:\n{self.neighbor_ids_dict}"
    
class LabelledAdjacencyList(EdgeRepresentation):

    def __init__(self, neighbor_ids_dict: Dict[int, List]):
        # The adjacency list could be of type 'Dict[int, List[Tuple(int, float)]]' or Dict[int, List[Tuple(int, int)]]'
        # depending on the data type of the used edge labels.
        self.neighbor_ids_dict: Dict[int, List] = neighbor_ids_dict
    
    def exists_edge(self, v, w) -> bool:
        if self.neighbor_ids_dict.get(v):
            for neighbor, label in self.neighbor_ids_dict.get(v):
                if neighbor == w:
                    return True
            return False
        else:
            return False
        
    def get_edge_label(self, v, w):
        if self.neighbor_ids_dict.get(v):
            for neighbor, label in self.neighbor_ids_dict.get(v):
                if neighbor == w:
                    return label
            return -1
        else:
            return None
    
    def get_neighborhood(self, v):
        return self.neighbor_ids_dict.get(v)

    def get_edge_representation(self):
        return self.neighbor_ids_dict

    def __repr__(self):
        return f"Labelled adjacency list representation:\n{self.neighbor_ids_dict}"


class Graph():
    """
    This class provides a simple container of graph structures which is reduced to the main components of a graph.
    Notice, that the edges can be stored in different classes. For example adjacency matrix or adjacency list.
    """
    __slots__ = "graph_id", "vertices", "vertex_labels", "edge_representation", "directed"

    def __init__(self, graph_id: int, vertices: np.array, vertex_labels: np.array, edge_representation: EdgeRepresentation, directed: bool = False):
        self.graph_id: int = graph_id
        self.vertices: np.array = vertices
        self.vertex_labels: np.array = vertex_labels
        self.edge_representation: EdgeRepresentation = edge_representation
        self.directed = directed

    def __repr__(self):
        return f"Graph id: '{self.graph_id}', |V|={self.vertices}\nlabels:\t{self.vertex_labels}\n{self.edge_representation}\n\n"

    def get_graph_id(self) -> int:
        return self.graph_id

    def get_vertices(self) -> np.array:
        return self.vertices.copy()

    def get_vertex_labels(self) -> np.array:
        return self.vertex_labels.copy()

    def get_neighborhood(self, v: int) -> List[int]:
        return self.edge_representation.get_neighborhood(v)

    def is_directed(self) -> bool:
        return self.directed
    
    def exists_edge(self, v, w) -> bool:
        if self.is_directed:
            return EdgeRepresentation.exists_edge(v, w)
        else:
            return EdgeRepresentation.exists_edge(v, w) or EdgeRepresentation.exists_edge(w, v)

class GraphDataset():
    """
    This class maintains a collection of Graph-instances from the same origin (dataset).
    It shall be used as a general datastructure, to oppose different implementations in frameworks
    like GraKel and OGB.
    """
    __slots__ = "ds_name", "ds_info", "graphs", "vertex_label_freq_dict"
    def __init__(self, dataset_name: str, information: str, graphs: List[Graph]) -> None:              
        self.ds_name: str = dataset_name
        self.ds_info: str = information
        self.graphs: List[Graph] = graphs
        # Keys = Vertex label, Value: Its frequency in the whole dataset.
        self.vertex_label_freq_dict: Dict[int, int] = self.count_vertex_label_frequencies()         
    
    def __repr__(self) -> str:
        return f"Dataset '{self.ds_name}'. Contains {self.size()} graphs."

    def count_vertex_label_frequencies(self, graph_list: List[Graph] = None) -> Dict[int, int]:
        """Get the vertex labels and their frequencies in the whole dataset."""
        if graph_list is None:
            graph_list = self.graphs

        # Keys = Vertex label, Value: Its frequency in the whole dataset.
        label_freq_dict = {}
        # Iterate over all graphs in the dataset.
        for graph in graph_list:
            vs_labels = graph.get_vertex_labels()
            # Get the labels and their frequencies.
            labels, frequencies = np.unique(vs_labels, return_counts = True)
            # Merge them to the frequencies in the whole dataset.
            for index, label in enumerate(labels):
                # If the label has already been countet, add to its frequency.
                if label_freq_dict.get(label):
                    
                    label_freq_dict[label] += frequencies[index]
                # Otherwise initialiye a new entry for the previously unknown label.
                else: 
                    label_freq_dict[label] = frequencies[index]

        return label_freq_dict

    def set_dataset_info(self, info: str)  -> None:
        self.ds_info = info

    def size(self) -> int:
        return len(self.graphs)

    def get_ds_name(self) -> str:
        return self.ds_name

    def get_ds_info(self) -> str:
        return self.ds_info

    def get_graph_list(self) -> List[Graph]:
        return self.graphs

    def get_vertex_label_frequencies(self) -> Dict[int, int]:
        return self.vertex_label_freq_dict
    
    def get_vertex_labels(self) -> List[int]:
        return self.vertex_label_freq_dict.keys()