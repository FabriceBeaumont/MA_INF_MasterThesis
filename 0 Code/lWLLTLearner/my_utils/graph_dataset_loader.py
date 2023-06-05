import numpy as np
import enum
from typing import List, Dict, Callable, Any
from abc import ABC, abstractmethod, abstractproperty

# Imports for GraKel
from grakel.datasets import fetch_dataset  # https://ysig.github.io/GraKeL/0.1a8/documentation.html

# Imports for graph imports via a pickle file
# In order to use python 3.6, pickle format 5 is not suported by default. Thus load it separately: !pip3 install pickle5
import pickle5 as pickle 

# Imports for OGB
from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl
from torch.utils.data import DataLoader
from os.path import exists
import grakel

from general_graph import Graph, GraphDataset
from general_graph import AdjacencyList, LabelledAdjacencyList, AdjacencyMatrix

DS_DIR_PATH = 'WLLTMetricLearner/GraphDatasets/'

class FrameworkLoader(ABC):
    """Basic representation of any datset loaders."""

    def __init__(self):
        self.dataset_name: str = '<Not defined!>'
        # The type of the dataset may be different in each framework/class instance
        self.dataset = None

    @abstractproperty
    def framework_name(self) -> str:  
        return NotImplementedError

    @abstractproperty
    def datasets_information_dict(self) -> Dict[str, str]:
        """A dictionary that maps known dataset names to a description string."""
        raise NotImplementedError
    
    @abstractproperty
    def convertable_formats(self) -> List[str]:
        """A list of formats, into which the loaded dataset can be converted to."""
        raise NotImplementedError

    def __repr__(self):
        return f"{self.framework_name} ({len(self.datasets_information_dict)} datasets)"

    @abstractmethod
    def load_dataset(self, dataset_name: str) -> None:
        """Load the dataset (from the web or a file)."""
        raise NotImplementedError

    @abstractmethod
    def get_dataset(self, dataset_name: str=None) -> Any:
        """Returns the loaded dataset."""
        raise NotImplementedError

    @abstractmethod
    def list_known_datasets(self) -> List[str]:
        """Returns a list of names of all datasets, the loader is able to load."""
        raise NotImplementedError
   
    @abstractmethod
    def convert_dataset_to(self, format: str) -> Any:
        """Returns that dataset, converted into another format."""        
        raise NotImplementedError

class IterableFrameworkLoader(FrameworkLoader):
    """Basic representation of any datset loaders that allow to iterate over all datasets it can handle."""
    def __init__(self):
        super().__init__()
        self.framework_ptr = 0

    def __iter__(self):
        """Return the class as iterator in order to iterate over all known datasets."""
        new_dataset_name = list(self.datasets_information_dict.keys())[self.framework_ptr]
        self.load_dataset(new_dataset_name)
        return self
    
    def __next__(self):
        """Allows to iterate over all datasets."""
        if self.framework_ptr < len(self.datasets_information_dict):
            new_dataset_name = list(self.datasets_information_dict.keys())[self.framework_ptr]
            self.framework_ptr += 1
            self.load_dataset(new_dataset_name)
            return self.dataset
        else:
            print("All known datasets were iterated over. Initialize a new class instance to start from the first known dataset.")
            return None
    

class GraKelLoader(IterableFrameworkLoader):
    """
    Dataset loader using the GraKel framework.
    https://ysig.github.io/GraKeL/0.1a8/generated/grakel.Graph.html#grakel.Graph
    """
    __slots__ = "dataset_name", "dataset", "framework_ptr"
    def __init__(self):
        super().__init__()        
        # In this instance the dataset is a tuple of an iterable of grakel graphs and their classes
        self.dataset = None

    @property
    def framework_name(self) -> str:  
        return "GraKel-Loader"

    @property
    def datasets_information_dict(self) -> Dict[str, str]:
        # Reference: https://chrsmrrs.github.io/datasets/docs/datasets/
        # Key = Dataset name, Value = Information on the dataset
        GRAKEL_DATASETS: Dict[str, str] = {
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

            "KKI": "TU Dortmund - Bioinformatics dataset\n'grakel.datasets.fetch_dataset('KKI')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            "OHSU": "TU Dortmund - Bioinformatics dataset\n'grakel.datasets.fetch_dataset('OHSU')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            "Peking_1": "TU Dortmund - Bioinformatics dataset\n'grakel.datasets.fetch_dataset('Peking_1')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
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

            "facebook_ct1": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('facebook_ct1')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            "IMDB-BINARY": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('IMDB-BINARY')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "IMDB-MULTI": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('IMDB-MULTI')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",       
            
            "PTC_FM": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('PTC_FM')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "PTC_FR": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('PTC_FM')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "PTC_MM": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('PTC_FM')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "PTC_MR": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('PTC_FM')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            "REDDIT-BINARY": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('REDDIT-BINARY')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "REDDIT-MULTI-12k": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('REDDIT-MULTI-12k')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "REDDIT-MULTI-5k": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('REDDIT-MULTI-5k')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",

            "twitch_egos": "TU Dortmund - Social networks\n'grakel.datasets.fetch_dataset('twitch_egos')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",        

            # Sznthetic
            "SYNTHETIC": "TU Dortmund - Synthetic\n'grakel.datasets.fetch_dataset('SYNTHETIC')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "SYNTHETICnew": "TU Dortmund - Synthetic\n'grakel.datasets.fetch_dataset('SYNTHETICnew')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
            "Synthie": "TU Dortmund - Synthetic\n'grakel.datasets.fetch_dataset('Synthie')'\nhttps://ysig.github.io/GraKeL/0.1a8/datasets.html \nhttps://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets"
        }    
        return GRAKEL_DATASETS

    @property
    def convertable_formats(self) -> List[str]:        
        return self._format_conversion_dict.keys()

    @property
    def _format_conversion_dict(self) -> Dict[str, Callable]:
        # Key = Convertable format description, Value = Method to get a convert the dataset
        D = {
            GRAPH_FORMATS.General_AdjList: self.convert_to_general_adj_list_graph_ds(),
            GRAPH_FORMATS.General_AdjMat: self.convert_to_general_adj_mat_graph_ds()
        }
        return D

    def load_dataset(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.dataset = fetch_dataset(self.dataset_name, verbose=False, as_graphs=True).data

    def get_dataset(self, dataset_name: str=None):
        if self.dataset_name != dataset_name:
            self.load_dataset(dataset_name)
        
        return self.dataset

    def list_known_datasets(self):
        return list(self.datasets_information_dict.keys())
      
    def convert_dataset_to(self, format: str):
        if format not in self._format_conversion_dict:
            raise Exception(f"Unknown format '{format}'! Conversion from {self.__name__} not possible!")
        else:
            return self._format_conversion_dict[format]
    
    ### Format converter functions ###

    def convert_to_general_adj_list_graph_ds(self):
        graph_list = []

        # Iterate through the dataset and convert each graph to a 'general_graph.Graph'.
        for grakel_id, grakel_graph in enumerate(self.dataset):
            graph_id = grakel_id
            vertices = np.array(list(grakel_graph.get_vertices()))
            vertex_labels = np.array(list(grakel_graph.index_node_labels.values()))

            # Construct the edge representation: Adjacency list.
            # Key = Vertex id, Value = Ids of its neighbors
            neighbor_ids_dict: Dict[int, List[int]] = {}
            for vertex_id in vertices:
                neighbor_ids_dict[vertex_id] = grakel_graph.neighbors(vertex_id)            

            # Construct the graph and add it to the list.
            graph_list.append(Graph(graph_id, vertices, vertex_labels, AdjacencyList(neighbor_ids_dict)))
        
        return GraphDataset(self.dataset_name, self.datasets_information_dict[self.dataset_name], graph_list)
    
    def convert_to_general_adj_mat_graph_ds(self):
        graph_list = []

        # Iterate through the dataset and convert each graph to a 'general_graph.Graph'.
        for grakel_id, grakel_graph in enumerate(self.dataset):
            graph_id = grakel_id
            vertices = np.array(list(grakel_graph.get_vertices()))
            vertex_labels = np.array(list(grakel_graph.index_node_labels.values()))

            # Construct the edge representation: Adjacency matrix.
            grakel_graph.change_format("adjacency")
            adjacency_mat = grakel_graph.get_adjacency_matrix()
            
            # Construct the graph and add it to the list.
            graph_list.append(Graph(graph_id, vertices, vertex_labels, AdjacencyMatrix(adjacency_mat)))
        
        return GraphDataset(self.dataset_name, self.datasets_information_dict[self.dataset_name], graph_list)

class OGBLoader(IterableFrameworkLoader):
    """
    Dataset loader using the OGB framework.
    https://docs.dgl.ai/en/latest/generated/dgl.heterograph.html
    """
    
    __slots__ = "dataset_name", "dataset", "framework_ptr"     
    def __init__(self):
        super().__init__()        
        # In this instance, the dataset is of class 'dgl.heterograph.DGLHeteroGraph' (https://docs.dgl.ai/en/latest/generated/dgl.heterograph.html)
        self.dataset = None    
    
    @property
    def framework_name(self) -> str:  
        return "OGB-Loader"

    @property
    def datasets_information_dict(self) -> Dict[str, str]:        
        # Reference: https://ogb.stanford.edu/docs/graphprop/
        # Key = Dataset name, Value = Information on the dataset
        OGB_DATASETS: Dict[str, str] = {
            # Small
            "ogbg-molhiv": "OGB small dataset (41,127 graphs) \nTask type: Binary classification \nMetric: ROC-AUC \nhttps://ogb.stanford.edu/docs/graphprop/",
            "ogbg-molpcba": "OGB small dataset (41,127 graphs) \nTask type: Binary classification \nMetric: ROC-AUC \nhttps://ogb.stanford.edu/docs/graphprop/",
            "ogbg-moltox21": "OGB small dataset \nTask type: Multi-class classification \nMetric: ROC-AUC \nhttps://ogb.stanford.edu/docs/graphprop/",
            "ogbg-molbace": "OGB small dataset \nTask type: Multi-class classification \nMetric: ROC-AUC \nhttps://ogb.stanford.edu/docs/graphprop/",
            "ogbg-molbbbp": "OGB small dataset \nTask type: Multi-class classification \nMetric: ROC-AUC \nhttps://ogb.stanford.edu/docs/graphprop/",
            "ogbg-molclintox": "OGB small dataset \nTask type: Multi-class classification \nMetric: ROC-AUC \nhttps://ogb.stanford.edu/docs/graphprop/",
            "ogbg-molmuv": "OGB small dataset \nTask type: Multi-class classification \nMetric: ROC-AUC \nhttps://ogb.stanford.edu/docs/graphprop/",
            "ogbg-molsider": "OGB small dataset \nTask type: Multi-class classification \nMetric: ROC-AUC \nhttps://ogb.stanford.edu/docs/graphprop/",
            "ogbg-moltoxcast": "OGB small dataset \nTask type: Multi-class classification \nMetric: ROC-AUC \nhttps://ogb.stanford.edu/docs/graphprop/",
            "ogbg-molesol": "OGB small dataset \nTask type: Regression \nMetric: ROC-AUC \nhttps://ogb.stanford.edu/docs/graphprop/",
            "ogbg-molfreesolv": "OGB small dataset \nTask type: Regression \nMetric: ROC-AUC \nhttps://ogb.stanford.edu/docs/graphprop/",
            "ogbg-mollipo": "OGB small dataset \nTask type: Regression \nMetric: ROC-AUC \nhttps://ogb.stanford.edu/docs/graphprop/",    
            # Medium
            "ogbg-molpcba": "OGB medium dataset (437,929 graphs) \nTask type: Binary classification \nMetric: AP \nhttps://ogb.stanford.edu/docs/graphprop/",
            "ogbg-ppa": "OGB medium dataset (158,100 graphs) \nTask type: Multi-class classification \nMetric: Accuracy \nhttps://ogb.stanford.edu/docs/graphprop/",
            # Bigger
            "ogbg-code2": "OGB medium dataset (452,741 graphs) \nTask type: Sub-token prediction \nMetric: F1 score \nhttps://ogb.stanford.edu/docs/graphprop/"
        }
        return OGB_DATASETS

    @property
    def convertable_formats(self) -> List[str]:        
        return self._format_conversion_dict.keys()

    @property
    def _format_conversion_dict(self) -> Dict[str, Callable]:
        # Key = Convertable format description, Value = Method to get a convert the dataset
        D = {
            GRAPH_FORMATS.General_AdjList: self.convert_to_general_adj_list_graph_ds(),
            GRAPH_FORMATS.NetworkX_list: self.convert_to_networkx_graph_list(),
            GRAPH_FORMATS.GraKel_list: self.convert_to_grakel_graph_list()
        }
        return D

    def load_dataset(self, dataset_name: str):        
        self.dataset_name = dataset_name
        # Graphs in the dataset are of class 'dgl.heterograph.DGLHeteroGraph' (https://docs.dgl.ai/en/latest/generated/dgl.heterograph.html)
        self.dataset = DglGraphPropPredDataset(name = dataset_name, root = DS_DIR_PATH)
    
    def get_dataset(self, dataset_name: str=None):
        if self.dataset_name != dataset_name:
            self.load_dataset(dataset_name)
        
        return self.dataset

    def list_known_datasets(self):
        return list(self.datasets_information_dict.keys())

    def convert_dataset_to(self, format: str):
        if format not in self._format_conversion_dict:
            raise Exception(f"Unknown format '{format}'! Conversion from {self.__name__} not possible!")
        else:
            return self._format_conversion_dict[format](self.dataset)
    
    ### Format converter functions ###

    def convert_to_general_adj_list_graph_ds(self) -> GraphDataset:
        # Convert the graphs to networkx, grakel and finally general graph instances
        grakel_networkx_graphs = OGBLoader.generate_grakel_graph_list(self.dataset)
        graph_list = []

        for graph_id, graph in enumerate(grakel_networkx_graphs):

            vertices = np.array(list(graph[0].keys()))
            vertex_labels = [] 
            # Construct the edge representation: Adjacency list.
            # Key = Vertex id, Value = Ids of its neighbors
            neighbor_ids_dict: Dict[int, List[int]] = {}
            for vertex_id in vertices:
                # This format stores the neighbors (keys) and the weights of the edges to them (values) in
                # a dictionary for every vertex. 
                neighbor_ids_dict[vertex_id] = list(graph[0][vertex_id].keys())
                  
            # Construct the graph and add it to the list.
            graph_list.append(Graph(graph_id, vertices, vertex_labels, AdjacencyList(neighbor_ids_dict)))            

        return GraphDataset(self.dataset_name, self.datasets_information_dict[self.dataset_name], graph_list)
    

    def convert_to_networkx_graph_list(self):
        # Use the DGL function 'to_networkx()' to convert them into NetworkX graphs
        return [g.to_networkx() for (g, _) in self.dataset]

    def convert_to_grakel_graph_list(self):
        # Use GraKels convert function to convert networkx graphs to grakel graphs
        return grakel.graph_from_networkx(self.generate_networkx_graph_list(self.dataset))

class PickleLoader(FrameworkLoader):
    """
    Dataset loader using a pickle file.
    https://igraph.org/r/doc/
    """    
    __slots__ = "dataset_name", "dataset"
    def __init__(self):
        # In this instance, the dataset is a tuple of igraphs and their classes.
        self.dataset = None  

    def __repr__(self):
        return super() + f"\n\tCurrently loaded: {self.dataset_name}"

    @property
    def framework_name(self) -> str:  
        return "Loading .pickle- and pickle.sec-files"

    @property
    def datasets_information_dict(self) -> Dict[str, str]:        
        return None

    @property
    def convertable_formats(self) -> List[str]:        
        return self._format_conversion_dict.keys()

    @property
    def _format_conversion_dict(self) -> Dict[str, Callable]:
        # Key = Convertable format description, Value = Method to get a convert the dataset
        D = {
            GRAPH_FORMATS.General_AdjList: self.convert_to_general_adj_list_graph_ds(),
            GRAPH_FORMATS.General_AdjMat: self.generate_general_adj_mat_graph_dataset()
        }
        return D
    
    def load_dataset(self, dataset_name: str):
        self.dataset_name = dataset_name   
        igraphs, classes = pickle.load(open(DS_DIR_PATH + dataset_name, 'rb'))
        self.dataset = (igraphs, classes)

    def get_dataset(self, dataset_name: str=None):
        if self.dataset_name != dataset_name:
            self.load_dataset(dataset_name)
        
        return self.dataset

    def list_known_datasets(self):
        return list(self.datasets_information_dict.keys())

    def convert_dataset_to(self, format: str):
        if format not in self._format_conversion_dict:
            raise Exception(f"Unknown format '{format}'! Conversion from {self.__name__} not possible!")
        else:
            return self._format_conversion_dict[format](self.dataset)

    # Format converter functions:

    def generate_general_adj_mat_graph_dataset(self) -> GraphDataset:
        igraphs, classes = self.dataset
        graph_list = []

        for igraph_id, igraph in enumerate(igraphs):
            graph_id = igraph_id
            vertices = np.arange(igraph.vcount())
            vertex_labels = np.array(igraph.vs['label'])
            
            # Construct the edge representation: Adjacency list.
            # Key = Vertex id, Value = Ids of its neighbors
            neighbor_ids_dict: Dict[int, List[int]] = {}    
            for vertex_id in vertices:
                neighbor_ids_dict[vertex_id] = igraph.neighbors(vertex_id)

            # Construct the graph and add it to the list.
            graph_list.append(Graph(graph_id, vertices, vertex_labels, AdjacencyList(neighbor_ids_dict)))
                
        return GraphDataset(self.dataset_name, self.datasets_information_dict[self.dataset_name], graph_list)

    def generate_general_adj_mat_graph_dataset(self):
        igraphs, classes = self.dataset
        graph_list = []

        for igraph_id, igraph in enumerate(igraphs):
            graph_id = igraph_id
            vertices = np.arange(igraph.vcount())
            vertex_labels = np.array(igraph.vs['label'])
            
            # Construct the edge representation: Adjacency matrix.
            adjacency_mat = igraph.get_adjacency()
            
            # Construct the graph and add it to the list.
            graph_list.append(Graph(graph_id, vertices, vertex_labels, AdjacencyMatrix(adjacency_mat)))
                
        return GraphDataset(self.dataset_name, self.datasets_information_dict[self.dataset_name], graph_list)


FRAMEWORK_LOADER_DICT = {
    'GraKel': GraKelLoader(),
    'OGB': OGBLoader(),
    'Pickle': PickleLoader()
}

class GRAPH_FORMATS(enum.Enum):
    General_AdjList = 0
    General_AdjMat = 1
    NetworkX_list = 2
    GraKel_list = 3
    

def get_framework_loader(dataset_name: str) -> FrameworkLoader:    
    """Returns a 'FrameworkLoader', if any such loader knows how to handle the passed 'dataset_name'."""
    # Deal with pickle files in the PickleLoader.
    if dataset_name.split('.')[-1] == "pickle" or dataset_name.split('.')[-1] == "sec":
        return PickleLoader()

    # Assume here, that the 'dataset_name' is a known dataset name and not the name of a (local) file.
    for framework_loader in FRAMEWORK_LOADER_DICT.values():
        if dataset_name in framework_loader.list_known_datasets():
            return framework_loader

    raise Exception(f"Unknown dataset or file type: '{dataset_name}'.")

def get_all_known_datasets(print_them: bool=False) -> List[str]:
    all_ds_names = []
    for framework_loader in FRAMEWORK_LOADER_DICT.values():        
        all_ds_names += framework_loader.get_known_datasets()
    
    if print_them:
        print(all_ds_names)

    return all_ds_names

def get_graph_dataset(dataset_name: str) -> GraphDataset:
    framework_loader = get_framework_loader(dataset_name)
    framework_loader.load_dataset(dataset_name)
    return framework_loader.get_dataset()

def get_general_graph_dataset(dataset_name: str, format: str =GRAPH_FORMATS.General_AdjList) -> GraphDataset:
    framework_loader = get_framework_loader(dataset_name)
    framework_loader.load_dataset(dataset_name)
    return framework_loader.convert_dataset_to(format)

def run_complete_test():    
    dataset_names = ["MUTAG", "ogbg-molbbbp"] 

    # Get a framework loader for each dataset name separately.
    for dataset_name in dataset_names:

        framework_loader = get_framework_loader(dataset_name)
        # Check the information of the actual dataset.
        print(framework_loader.datasets_information_dict.get(dataset_name))
        print(f'\n>>> Framework loader for {dataset_name} is ready.\n')

        # Test methods and definitions of the framework loader.
        print(framework_loader)
        print(framework_loader.list_known_datasets())
        print(f"Dataset {len(next(framework_loader))} in dataset {framework_loader.dataset_name}")
        print(f"Dataset {len(next(framework_loader))} in dataset {framework_loader.dataset_name}")
        print('\n\n')

def main():
    pass

if __name__ == '__main__':
    main()