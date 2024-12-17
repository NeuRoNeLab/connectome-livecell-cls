import os
import random
from typing import Tuple, Dict
import igraph as ig
import numpy as np
from utils.connectome_reader import ROLE, SYNAPSE_TYPE, SENSOR_ROLE, MOTOR_ROLE, INTERNEURON_ROLE_ORIGINAL, \
    SYNAPSE_ELECTRIC, SYNAPSE_CHEMICAL


def generate_erdos_renyi_graph(n: int,
                               p: float,
                               n_p: Dict[str, float],
                               e_p: Dict[str, float],
                               directory: str,
                               seed: int = 42) -> ig.Graph:
    random.seed(seed)
    graph = ig.Graph.Erdos_Renyi(n, p)

    random.seed(seed)
    node_labels = random.choices([SENSOR_ROLE, MOTOR_ROLE, INTERNEURON_ROLE_ORIGINAL],
                                 weights=[n_p[SENSOR_ROLE], n_p[MOTOR_ROLE], n_p[INTERNEURON_ROLE_ORIGINAL]], k=n)
    graph.vs[ROLE] = node_labels

    random.seed(seed)
    edge_types = random.choices([SYNAPSE_ELECTRIC, SYNAPSE_CHEMICAL],
                                weights=[e_p[SYNAPSE_ELECTRIC], e_p[SYNAPSE_CHEMICAL]], k=graph.ecount())
    graph.es[SYNAPSE_TYPE] = edge_types

    os.makedirs(f"{directory}", exist_ok=True)
    filename = f"{directory}/graph_{seed}.graphml"
    graph.write_graphml(filename)

    return graph


def generate_barabasi_albert_graph(n: int,
                                   m: int,
                                   n_p: Dict[str, float],
                                   e_p: Dict[str, float],
                                   directory: str,
                                   seed: int = 42) -> ig.Graph:
    random.seed(seed)
    graph = ig.Graph.Barabasi(n, m - random.randint(0, m // 4))

    random.seed(seed)
    node_labels = random.choices([SENSOR_ROLE, MOTOR_ROLE, INTERNEURON_ROLE_ORIGINAL],
                                 weights=[n_p[SENSOR_ROLE], n_p[MOTOR_ROLE], n_p[INTERNEURON_ROLE_ORIGINAL]], k=n)
    graph.vs[ROLE] = node_labels

    random.seed(seed)
    edge_types = random.choices([SYNAPSE_ELECTRIC, SYNAPSE_CHEMICAL],
                                weights=[e_p[SYNAPSE_ELECTRIC], e_p[SYNAPSE_CHEMICAL]], k=graph.ecount())
    graph.es["edge_type"] = edge_types

    os.makedirs(f"{directory}", exist_ok=True)
    filename = f"{directory}/graph_{seed}.graphml"
    graph.write_graphml(filename)

    return graph


def generate_watts_strogatz_graph(n: int,
                                  k: int,
                                  p: float,
                                  n_p: Dict[str, float],
                                  e_p: Dict[str, float],
                                  directory: str,
                                  seed: int = 42) -> ig.Graph:
    random.seed(seed)
    neigh = random.randint(k // 2, k)
    graph = ig.Graph.Watts_Strogatz(dim=1, size=neigh, nei=int(k), p=p, loops=False)

    random.seed(seed)
    node_labels = random.choices([SENSOR_ROLE, MOTOR_ROLE, INTERNEURON_ROLE_ORIGINAL],
                                 weights=[n_p[SENSOR_ROLE], n_p[MOTOR_ROLE], n_p[INTERNEURON_ROLE_ORIGINAL]], k=n)
    graph.vs[ROLE] = node_labels

    random.seed(seed)
    edge_types = random.choices([SYNAPSE_ELECTRIC, SYNAPSE_CHEMICAL],
                                weights=[e_p[SYNAPSE_ELECTRIC], e_p[SYNAPSE_CHEMICAL]], k=graph.ecount())
    graph.es["edge_type"] = edge_types

    os.makedirs(f"{directory}", exist_ok=True)
    filename = f"{directory}/graph_{seed}.graphml"
    graph.write_graphml(filename)

    return graph


def analyze_degree(connectome_path: str = 'data/connectomes/celegans.graphml') -> Tuple[float, float, int, int]:
    graph = ig.read(connectome_path)
    degrees = graph.degree()
    mean_degree = np.mean(degrees)
    median_degree = np.median(degrees)
    max_degree = np.max(degrees)
    min_degree = np.min(degrees)

    return mean_degree, median_degree, max_degree, min_degree


def analyze_graph(connectome_path: str = 'data/connectomes/celegans.graphml') -> Tuple[Dict[str, float],
                                                                                       Dict[str, float], int, int, int]:
    graph = ig.read(connectome_path)

    node_labels = graph.vs[ROLE]
    total_nodes = len(node_labels)
    label_counts = {label: node_labels.count(label) for label in set(node_labels)}
    node_percentages = {label: count / total_nodes  for label, count in label_counts.items()}

    edge_types = graph.es[SYNAPSE_TYPE]
    total_edges = len(edge_types)
    edge_type_counts = {edge_type: edge_types.count(edge_type) for edge_type in set(edge_types)}
    edge_percentages = {edge_type: count / total_edges   for edge_type, count in edge_type_counts.items()}
    num_nodes = graph.vcount()
    num_edges = graph.ecount()
    mean_degree, median_degree, max_degree, min_degree = analyze_degree(connectome_path=connectome_path)

    return node_percentages, edge_percentages, num_nodes, num_edges, int(median_degree)
