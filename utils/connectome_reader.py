################################################################################
# Copyright Bardozzo et al.
# Graph Net Spektral extention + multi-dyadic algorithms
################################################################################
from __future__ import annotations  # enable type annotation with a certain class within the same class definition
from typing import Final, Union, Optional
import igraph
from igraph import Graph
import numpy as np
import random
import itertools
import networkx as nx
import torch_geometric.data as tgd
import torch_geometric.utils as tgu
import pandas as pd
from functools import partial
from utils.utils import ALPHA, weighted_value_counts, ListDict


# Rewiring flags
NO_REWIRE: Final[int] = 1
REWIRE_NO_LOOPS_NO_MULTIPLE: Final[int] = 2
REWIRE_NO_LOOPS_MULTIPLE: Final[int] = 3
REWIRE_LOOPS_NO_MULTIPLE: Final[int] = 4
REWIRE_LOOPS_MULTIPLE: Final[int] = 5


# Misc names, flags and identifiers
U: Final[str] = 'u'  # source node
V: Final[str] = 'v'  # target node
EDGE: Final[str] = "edge"
ROLE: Final[str] = "role"  # node role, may be interneuron, sensor or motor
COLOR: Final[str] = "color"
WEIGHT: Final[str] = "weight"  # edge weight, represents the number of synapses between two neurons
SYNAPSE_TYPE: Final[str] = "synapse_type"
SYNAPSE_ELECTRIC: Final[str] = 'E'  # undirected connection
SYNAPSE_CHEMICAL: Final[str] = 'C'  # directed connection
INTERNEURON_ROLE: Final[str] = 'I'  # used name of the interneurons
INTERNEURON_ROLE_ORIGINAL: Final[str] = 'NA'  # original name of the interneurons
SENSOR_ROLE: Final[str] = 'S'
MOTOR_ROLE: Final[str] = 'M'

# Additional node attributes, not used here, but must be handled to convert the networkx to igraph
ID: Final[str] = 'ID'  # it is a string the form 'n<node_num>'
CELL_NAME: Final[str] = 'cell_name'  # it is a string
DEFAULT_CELL_NAME: Final[str] = 'new_cell'
CELL_CLASS: Final[str] = 'cell_class'  # it is a string
POS: Final[str] = 'soma_pos'  # it is a float
NEUROTRANSMITTERS: Final[str] = 'neurotransmitters'  # it is a string
NEUROTRANSMITTERS_NAMES: Final[list[str]] = sorted(["NA", "Ach", "GABA", "DA", "5-HT", "GLU"])  # Ach,F-HT omitted


def ensure_weak_connectivity(g: nx.DiGraph):
    if not nx.is_weakly_connected(g):

        # Edges to add to ensure connectivity
        added_edges = set()

        # Connected components
        components = list(nx.weakly_connected_components(g))
        """
        # Initialize union-find with weakly connected components
        nodes = g.nodes(data=False)
        uf = nx.utils.UnionFind(elements=nodes)
        remaining_components_trace = ListDict(nodes)  # traces remaining components
        for c in components:
            if log:
                print(f"Union {c}")

            # Perform union on component vertices
            uf.union(*c)

            # Remove all vertices from component trace besides one to trace that component
            remaining = len(c)
            for v in c:
                if remaining > 1:
                    remaining_components_trace.remove(v)

        # Choose edges to add until the graph is weakly connected (there is only 1 components containing all the nodes)
        n = len(nodes)
        while len(remaining_components_trace) > 1:
            # Pick two random nodes from components
            i, j = remaining_components_trace.choose_random_distinct_couple()
        """

        # For each component
        prev = None
        for c in components:

            # Create efficient data structure for choosing random node in the component
            component_list_dict = ListDict(c)

            if prev is not None:
                # Choose random nodes from current and last component and add an edge
                u = component_list_dict.choose_random()
                v = prev.choose_random()
                added_edges.add((u, v))
            prev = component_list_dict

        # Add edges
        g.add_edges_from(added_edges)


def weighted_edge_type_count(edge_df: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
    """
    For each node, count the number of synapse types (chemical or electric) and weight them by the number of edges,
    returning a dataframe with results.

    :param edge_df: the dataframe containing the edges
    :type edge_df: pd.DataFrame
    :param normalize: whether or not to normalize the weighted counts obtaining probabilities, defaults to True
    :type normalize: bool (optional)
    :return: A dataframe with the synapse type counts for each node.
    """
    if U not in edge_df or V not in edge_df:
        # Separate node in two columns
        node_columns = pd.DataFrame(edge_df[EDGE].values.tolist())
        node_columns.columns = [U, V]
    else:
        node_columns = edge_df[[U, V]]

    # Get the nodes
    nodes_u = pd.unique(node_columns[U])
    nodes_v = pd.unique(node_columns[V])

    # Add U and V columns to dataframe if required
    if U not in edge_df and V not in edge_df:
        edge_df2 = pd.concat((node_columns, edge_df), axis=1)
    elif U not in edge_df:
        edge_df2 = pd.concat((node_columns[U], edge_df), axis=1)
    elif V not in edge_df:
        edge_df2 = pd.concat((node_columns[V], edge_df), axis=1)
    else:
        edge_df2 = edge_df
    edge_df2 = edge_df2[[U, V, EDGE, SYNAPSE_TYPE, WEIGHT]]  # this copies the edge_df
    edge_df2 = edge_df2.set_index([U, V])

    # For each u and v, create a row of a new dataframe with the synapse type weighted counts
    synapse_type_counts = {}
    for u in nodes_u:
        synapse_type_counts[u] = weighted_value_counts(
            df=edge_df2.loc[u][[SYNAPSE_TYPE, WEIGHT]],
            normalize=normalize,
            alpha=ALPHA,
            categories=[SYNAPSE_CHEMICAL, SYNAPSE_ELECTRIC]
        )

    for v in nodes_v:
        if v not in synapse_type_counts:
            synapse_type_counts[v] = weighted_value_counts(
                df=edge_df2.xs(key=v, level=V)[[SYNAPSE_TYPE, WEIGHT]],
                normalize=normalize,
                alpha=ALPHA,
                categories=[SYNAPSE_CHEMICAL, SYNAPSE_ELECTRIC]
            )
        else:
            # Add the two results if we encounter an edge again
            synapse_type_counts[v] += weighted_value_counts(
                df=edge_df2.xs(key=v, level=V)[[SYNAPSE_TYPE, WEIGHT]],
                normalize=normalize,
                alpha=ALPHA,
                categories=[SYNAPSE_CHEMICAL, SYNAPSE_ELECTRIC]
            )

    # Construct the synapse_type_counts dataframe, replacing missing values with 0s
    df = pd.DataFrame(synapse_type_counts).T

    # Re-normalize results since we summed up two normalized results
    tmp = df[[SYNAPSE_CHEMICAL, SYNAPSE_ELECTRIC]].apply(lambda e: e[0] + e[1] > 1, axis=1)
    df.loc[tmp, [SYNAPSE_CHEMICAL, SYNAPSE_ELECTRIC]] /= 2

    return df


def node_mean_weight(edge: tuple[int, int], edge_df: pd.DataFrame) -> (tuple[int, int], np.float64):
    """
    Takes an edge and a dataframe of edges, and returns the mean weight of the edges incident to the edge's nodes.

    :param edge: the edge to compute the mean weight for
    :type edge: tuple[int, int]
    :param edge_df: the dataframe containing the edges
    :type edge_df: pd.DataFrame
    :return: The mean weight of the edges incident to u and v
    """
    u = edge[0]
    v = edge[1]

    # Select weights of the edges incident to u and v
    weights = edge_df.loc[edge_df[EDGE].apply(lambda e: e[0] == u or e[1] == u or e[0] == v or e[1] == v), WEIGHT]
    weights = weights.dropna()  # drop NaN weights

    # If there aren't non-NaN weights, just return 1, because it's the only connection
    if len(weights) == 0:
        return edge, np.round(1.0)

    # Otherwise, return the mean of those weights
    return edge, np.round(weights.mean())


def _sum_weights(edge: tuple[int, int], edge_df: pd.DataFrame, directed: bool = False) -> (tuple[int, int], np.float64):
    """
    If the reversed edge exists and its directed, add the weights of both the directions to get the total weight,
    otherwise just add the duplicate edge weights.

    :param edge: tuple[int, int]
    :type edge: tuple[int, int]
    :param edge_df: the dataframe containing the edges and weights
    :param directed: whether the connections are directed, defaults to False
    :type directed: bool (optional)

    :return: A tuple of the edge and the weight of the edge.
    """
    u = edge[0]
    v = edge[1]
    weight = np.float64(np.sum(edge_df.loc[[(u, v)]][WEIGHT]))  # weight

    if (v, u) in edge_df.index and not directed:
        weight += np.float64(np.sum(edge_df.loc[[(v, u)]][WEIGHT]))  # reverse weight

    return edge, weight


def fill_nan_synapse_weights(edge_df: pd.DataFrame):
    """
    For each edge in the edge dataframe, if the weight is NaN, replace it with the mean weight of the edge's nodes.

    :param edge_df: the dataframe containing the edges and weights
    :type edge_df: pd.DataFrame
    """

    node_mean_weight_ = partial(node_mean_weight, edge_df=edge_df)

    edge_df.loc[:, [EDGE, WEIGHT]] = edge_df[[EDGE, WEIGHT]].apply(
        lambda edge_weight: node_mean_weight_(edge=edge_weight[0]) if np.isnan(edge_weight[1]) else edge_weight,
        axis=1
    )


def fill_undefined_synapse_type(edge_df: pd.DataFrame):
    """
    If the synapse type is undefined, randomly assign it to either chemical or electrical.

    :param edge_df: the dataframe containing the edges
    :type edge_df: pd.DataFrame
    """
    edge_df.loc[edge_df[SYNAPSE_TYPE] == 'None', SYNAPSE_TYPE] = random.choice([SYNAPSE_ELECTRIC, SYNAPSE_CHEMICAL])


def one_hot_encode_node_features(node_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a dataframe with a column called `ROLE` and one-hot encodes it

    :param node_df: a pandas dataframe containing the node features
    :type node_df: pd.DataFrame

    :return: A dataframe with the one-hot encoded node features.
    """
    df = pd.DataFrame(pd.get_dummies(node_df[ROLE]))  # one-hot encode column
    df = df[sorted(df.columns)]  # sort columns to grant consistency between different connectomes
    """df.rename({
        f"{ROLE}_{INTERNEURON_ROLE}": INTERNEURON_ROLE,
        f"{ROLE}_{SENSOR_ROLE}": SENSOR_ROLE,
        f"{ROLE}_{MOTOR_ROLE}": MOTOR_ROLE
    })"""

    if NEUROTRANSMITTERS in node_df:
        df2 = pd.DataFrame(pd.get_dummies(node_df[NEUROTRANSMITTERS]))  # one-hot encode column
        df2 = df2[sorted(df2.columns)]  # sort columns to grant consistency between different connectomes
        """df2.rename({
            col: col.split("_")[1] for col in df2.columns
        })"""

        # Update the one-hot vector of the subcolumns of each composed columm, removing it
        columns = df2.columns
        to_drop = []
        for col in columns:
            sub_cols = col.split(",")

            # If the column is composed by two columns, set both one-hot components to 1 and remove the column
            if len(sub_cols) > 1:
                df2.loc[df2[col] == 1, sub_cols] = 1
                to_drop.append(col)

        # Drop composed columns
        df2 = df2.drop(labels=to_drop, axis=1)

        # Concat the result dataframe with the role dataframe
        df = pd.concat((df, df2), axis=1)

    if POS in node_df:
        # Add POS feature if required
        df = pd.concat((df, node_df[POS]), axis=1)

    return df


class ConnectomeReader(object):
    def __init__(self, name_file: str):
        """
        This class reads in a connectome file and returns a graph.

        :param name_file: the name of the file that contains the connectome graph
        :type name_file: str
        """
        self.__name_file: str = name_file
        self.__graph_el: Union[int, igraph.Graph] = 0
        self.__edge_df: Optional[pd.DataFrame] = None
        self.__node_df: Optional[pd.DataFrame] = None
        self.__synapse_type_node_probs: Optional[pd.DataFrame] = None
        self.__synapse_type_probs: Optional[pd.DataFrame] = None
        self.__nxgraph: Optional[Union[nx.Graph, nx.DiGraph]] = None

    def read(self, sym_flag=NO_REWIRE, pp=0.1, seed=1):
        """
        The function reads a graphml file and rewires the edges of the graph according to the parameters passed to the
        function.

        The function takes three parameters:

        1. sym_flag: This is an integer that determines the type of rewiring.
        2. pp: This is a float that determines the probability of rewiring.
        3. seed: This is an integer that determines the seed for the random number generator.

        The function returns a graph object.

        :param sym_flag: , defaults to 1, which means no rewiring (optional)
        :param pp: the probability of rewiring an edge
        :param seed: the seed for the random number generator, defaults to 1 (optional)
        :return: The graph_el object is being returned.
        """
        self.__graph_el = Graph.Read_GraphML(self.name_file)

        if sym_flag == REWIRE_NO_LOOPS_NO_MULTIPLE:
            random.seed(seed)
            self.graph_el.rewire_edges(pp, loops=False, multiple=False)
        elif sym_flag == REWIRE_NO_LOOPS_MULTIPLE:
            random.seed(seed)
            self.graph_el.rewire_edges(pp, loops=False, multiple=True)
        elif sym_flag == REWIRE_LOOPS_NO_MULTIPLE:
            random.seed(seed)
            self.graph_el.rewire_edges(pp, loops=True, multiple=False)
        elif sym_flag == REWIRE_LOOPS_MULTIPLE:
            random.seed(seed)
            self.graph_el.rewire_edges(pp, loops=True, multiple=True)

        return self.graph_el

    @property
    def name_file(self) -> str:
        return self.__name_file

    @name_file.setter
    def name_file(self, name_file: str):
        self.__name_file = name_file

    @property
    def graph_el(self) -> Union[igraph.Graph, int]:
        return self.__graph_el

    def make_it_undirected(self):
        self.__graph_el = self.graph_el.as_undirected()

    def get_it_undirected(self):
        return self.graph_el.as_undirected()

    def get_edge_list(self):
        l1 = []
        l2 = []
        for el in self.graph_el.get_edgelist():
            l1.append(el[0])
            l2.append(el[1])

        return l1, l2

    def get_paths(self, length_effect=3):
        l1 = []
        l1_func = []
        l2 = []
        l2_func = []
        l3 = []

        for el in [p for p in itertools.product(np.asarray(self.graph_el.vs().indices), repeat=2)]:
            ll = self.graph_el.get_all_shortest_paths(el[0], el[1], mode=None)
            if ll:
                if len(ll[0]) == length_effect:
                    for el2 in ll:
                        if len(el2) == length_effect:
                            l3.append(el2)
                            l1.append(el[0])
                            l2.append(el[1])
                            el_f1 = self.graph_el.vs()[el[0]][ROLE]
                            if el_f1 == INTERNEURON_ROLE_ORIGINAL:
                                el_f1 = INTERNEURON_ROLE
                            l1_func.append(el_f1)
                            el_f2 = self.graph_el.vs()[el[1]][ROLE]
                            if el_f2 == INTERNEURON_ROLE_ORIGINAL:
                                el_f2 = INTERNEURON_ROLE
                            l2_func.append(el_f2)

        return l1, l2, l3, l1_func, l2_func

    def get_triplet_list(self):
        l1 = []
        l2 = []
        l3 = []
        for el in [p for p in itertools.product(np.asarray(self.graph_el.vs().indices), repeat=2)]:
            ll = self.graph_el.get_all_shortest_paths(el[0], el[1], mode=None)
            if ll:
                if len(ll[0]) == 3:
                    for el2 in ll:
                        if len(el2) == 3:
                            l3.append(el2)
                            l1.append(el[0])
                            l2.append(el[1])

        # generate dict edges
        l0 = []
        for i in range(0, len(l2), 1):
            l0.append([l1[i], l2[i]])

        ll_0 = list([set(x) for x in l0])
        edges_dict = {index: list(value) for index, value in enumerate(ll_0)}

        # generate dict triplets
        ll = list([set(x) for x in l3])
        target_dict = {index: list(value) for index, value in enumerate(ll)}

        # print("--------")
        # print(l1[14600])
        # print(l2[14600])
        # print(edges_dict.get(14600))   #[a, b]
        # print(target_dict.get(14600))  #[
        # print("--------")

        # input()

        return edges_dict, target_dict, l1, l2

    def get_role_list(self):
        edge_list1, edge_list2 = self.get_edge_list()
        role_list_l1 = []
        role_list_l2 = []

        for el in edge_list1:
            el1 = self.graph_el.vs()[el][ROLE]
            if el1 == INTERNEURON_ROLE_ORIGINAL:
                el1 = INTERNEURON_ROLE
            role_list_l1.append(el1)

        for el in edge_list2:
            el2 = self.graph_el.vs()[el][ROLE]
            if el2 == INTERNEURON_ROLE_ORIGINAL:
                el2 = INTERNEURON_ROLE
            role_list_l2.append(el2)

        return role_list_l1, role_list_l2

    def get_syn_list(self):
        edge_list = self.graph_el.get_edgelist()
        syn_list = []

        for el in edge_list:
            gg_el = self.graph_el.es[self.graph_el.get_eid(el[0], el[1])]
            el1 = gg_el[SYNAPSE_TYPE]
            syn_list.append(el1)

        return syn_list

    def get_syn_dict(self) -> dict[tuple[int, int], dict]:
        """
        This function takes a graph and returns a dictionary of dictionaries, where the keys are tuples of the form
        (source, target) and the values are dictionaries of the form {weight: weight, synapse_type: synapse_type}

        :return: A dictionary of dictionaries. The keys are edges of the form (int, int) and the values are
            dictionaries representing the edge features, namely weight and synapse type.
        """

        if self.graph_el == 0:
            raise ValueError("Graph not read yet")

        edge_list = self.graph_el.get_edgelist()
        syn_dict = {}

        for el in edge_list:
            gg_el = self.graph_el.es[self.graph_el.get_eid(el[0], el[1])]
            syn_type = gg_el[SYNAPSE_TYPE]
            weight = gg_el[WEIGHT]
            syn_dict[el] = {WEIGHT: weight, SYNAPSE_TYPE: syn_type}

        return syn_dict

    def get_node_role_dict(self) -> dict[int, str]:
        """
        Takes a graph and returns a dictionary where the keys are the node IDs and the values are the node roles.

        :return: A dictionary of node IDs and their corresponding roles.
        """

        if self.graph_el == 0:
            raise ValueError("Graph not read yet")

        node_role_dict = {}

        # For each node, get the corresponding role and put it in the dictionary
        for edge in self.graph_el.get_edgelist():
            u, v = edge
            # print(f"{u}: {self.graph_el.vs()[u][ROLE]}")
            # print(f"{v}: {self.graph_el.vs()[v][ROLE]}")
            if u not in node_role_dict:
                role = INTERNEURON_ROLE if self.graph_el.vs()[u][ROLE] == INTERNEURON_ROLE_ORIGINAL else \
                    self.graph_el.vs()[u][ROLE]
                node_role_dict[u] = role
            if v not in node_role_dict:
                role = INTERNEURON_ROLE if self.graph_el.vs()[v][ROLE] == INTERNEURON_ROLE_ORIGINAL else \
                    self.graph_el.vs()[v][ROLE]
                node_role_dict[v] = role

        return node_role_dict

    def get_weight_edge(self):
        edge_list = self.graph_el.get_edgelist()
        syn_list = []

        for el in edge_list:
            gg_el = self.graph_el.es[self.graph_el.get_eid(el[0], el[1])]
            el1 = gg_el[WEIGHT]
            syn_list.append(el1)

        return syn_list

    def get_synapse_type_probs(self, normalize: bool = True, cached: bool = True) -> pd.Series:
        """
        Returns a series of synapse types and corresponding weighted probabilities.

        :param normalize: whether to normalize the frequency count to get probabilities, defaults to True
        :type normalize: bool (optional)
        :param cached: whether use the cached node dataframe (may not correspond to the given parameters), defaults to
            True
        :type cached: bool (optional)

        :return: A series of synapse types and corresponding weighted probabilities.
        :rtype: pd.Series
        """
        if self.__synapse_type_probs is None or not cached:
            # Get the edge dataframe
            edge_df = self.get_edge_dataframe()

            # Calculate the weighted probabilities
            self.__synapse_type_probs = weighted_value_counts(
                edge_df,
                normalize=normalize,
                categories=[SYNAPSE_CHEMICAL, SYNAPSE_ELECTRIC]
            )
        return self.__synapse_type_probs

    def get_synapse_type_node_probs(self, normalize: bool = True, cached: bool = True) -> pd.DataFrame:
        """
        For each node, count the number of synapse types (chemical or electric) and weight them by the number of edges,
        returning a dataframe with results.


        :param normalize: whether or not to normalize the weighted counts obtaining probabilities, defaults to True
        :type normalize: bool (optional)
        :param cached: whether use the cached node dataframe (may not correspond to the given parameters), defaults to
            True
        :type cached: bool (optional)

        :return: A dataframe with the synapse type counts for each node.
        :rtype: pd.DataFrame
        """
        if self.__synapse_type_node_probs is None or not cached:
            # Get the edge dataframe
            edge_df = self.get_edge_dataframe()

            # Calculate the weighted node probabilities
            self.__synapse_type_node_probs = weighted_edge_type_count(edge_df, normalize=normalize)
        return self.__synapse_type_node_probs

    def get_node_dataframe(self, cached: bool = True, additional_features: Optional[list] = None) -> pd.DataFrame:
        """
        Returns a dataframe of node features of the connectome graph.

        :param cached: whether use the cached node dataframe (may not correspond to the given parameters), default to
            True
        :type cached: bool (optional)
        :param additional_features: additional features provided, these may include NEUROTRANSMITTERS, POS or both.
        :type additional_features: list[str]

        :return: A dataframe with the node id as the index and the role as the column.
        :rtype: pd.DataFrame
        """
        if self.graph_el and (self.__node_df is None or not cached):
            """# Get node roles
            node_roles = self.get_node_role_dict()

            # Create dataframe for node features
            node_df = pd.DataFrame(node_roles, index=[0]).T
            node_df.columns = [ROLE]"""

            if additional_features is None:
                additional_features = []

            # Get features
            features = [ROLE, *additional_features]

            # Get full node dataframe and select features
            node_df = self.graph_el.get_vertex_dataframe()
            node_df = node_df[features]

            # Replace INTERNEURON_ROLE_ORIGINAL with INTERNEURON_ROLE
            node_df.loc[node_df[ROLE] == INTERNEURON_ROLE_ORIGINAL, ROLE] = INTERNEURON_ROLE

            # Set node dataframe attribute
            self.__node_df = node_df

        elif not self.graph_el:
            raise ValueError("read() must be called before using the running graph.")

        return self.__node_df

    def get_edge_dataframe(self, fill_undefined_types: bool = True, fill_nan_weights: bool = True,
                           cached: bool = True) -> pd.DataFrame:
        """
        Returns a dataframe containing the synapse data.


        :param fill_undefined_types: If True, then any undefined synapse types will be set to a random choice between
            electric and chemical, defaults to False

        :type fill_undefined_types: bool (optional)

        :param fill_nan_weights: If True, then any NaN weights will be filled with the mean weight of the edge's nodes,
            defaults to False
        :type fill_nan_weights: bool (optional)
        :param cached: whether use the cached edge dataframe (may not correspond to the given parameters), defaults to
            True
        :type cached: bool (optional)

        :return: A dataframe with the edge, weight, and synapse type.
        :rtype: pd.DataFrame
        """

        """
        # Get synapse info
        syn_dict = self.get_syn_dict()

        # Create synapse dataframe
        edge_df = pd.DataFrame(columns=[EDGE, WEIGHT, SYNAPSE_TYPE])
        for edge in syn_dict:
            edge_info = syn_dict[edge]
            edge_info.update({EDGE: edge})
            edge_df.loc[len(edge_df)] = edge_info
        """

        if self.graph_el and (self.__edge_df is None or not cached):

            # Get edge dataframe and rename columns
            edge_df = self.graph_el.get_edge_dataframe()
            edge_df = edge_df.rename({'source': U, 'target': V}, axis=1)

            # Create edge column and add it to dataframe
            edge_col = edge_df.apply(lambda row: (row[U], row[V]), axis=1).rename(EDGE)
            edge_df[EDGE] = edge_col
            edge_df = edge_df[[U, V, EDGE, SYNAPSE_TYPE, WEIGHT]]

            # Set undefined connections to random between electric and chemical if required
            if fill_undefined_types:
                fill_undefined_synapse_type(edge_df)

            # Fill NaN edge weights with mean weight of the edge's nodes if required
            if fill_nan_weights:
                fill_nan_synapse_weights(edge_df)

            # Set edge dataframe attribute
            self.__edge_df = edge_df

        elif not self.graph_el:
            raise ValueError("read() must be called before using the running graph.")

        return self.__edge_df

    @staticmethod
    def __add_node_feat(node_row, g: Union[nx.Graph, nx.DiGraph], one_hot: bool = False,
                        additional_features: Optional[list] = None):

        if additional_features is None:
            additional_features = []

        if not isinstance(node_row, pd.Series):
            node_row = node_row.to_frame()

        # Add role features
        if one_hot:
            features = {
                ROLE: np.array(node_row[sorted([INTERNEURON_ROLE, MOTOR_ROLE, SENSOR_ROLE])])
            }
        else:
            features = {
                ROLE: np.array(node_row[ROLE])
            }

        # Add additional features
        for feat in additional_features:
            if feat == NEUROTRANSMITTERS and one_hot:
                features[NEUROTRANSMITTERS] = np.array(node_row[NEUROTRANSMITTERS_NAMES])
            elif feat == NEUROTRANSMITTERS and not one_hot:
                features[NEUROTRANSMITTERS] = np.array(node_row[NEUROTRANSMITTERS])
            elif feat == POS:
                features[POS] = node_row[POS]

        # Add node to the graph
        g.add_node(node_row.name, **features)

    def to_networkx(self, one_hot_node_features: bool = True, directed: bool = False,
                    cached: bool = True, additional_features: Optional[list] = None) -> Union[nx.Graph, nx.DiGraph]:
        """
        Converts the running `iGraph` object and returns a `networkx` graph object.

        The `iGraph` object is a class that we've defined in the `igraph` package. It's a instance that represents
        a connectome network.

        The `networkx` graph object is a class that's defined in the `networkx` package. It's a class that represents a
        graph.

        :param one_hot_node_features: bool = True, defaults to True
        :type one_hot_node_features: bool (optional)
        :param directed: Whether the graph is directed or not, defaults to False
        :type directed: bool (optional)
        :param cached: whether use the cached networkx graph (may not correspond to the given parameters), defaults to
            True
        :type cached: bool (optional)
        :param additional_features: additional features provided, these may include NEUROTRANSMITTERS, POS or both.
        :type additional_features: list[str]

        :return: A networkx graph object representing the connectome
        """

        if self.__nxgraph is None or not cached:

            # Create NetworkX graph
            nxg = nx.DiGraph() if directed else nx.Graph()

            # Create dataframe for edge and node features, setting undefined connections to random between electric and
            # chemical and filling NaN edge weights with mean weight of the edge's nodes
            node_df = self.get_node_dataframe(cached=cached, additional_features=additional_features).copy()
            edge_df = self.get_edge_dataframe(fill_undefined_types=True, fill_nan_weights=True, cached=cached).copy()

            # One-hot encode node features if required and add nodes to the graph
            if one_hot_node_features:
                node_df = one_hot_encode_node_features(node_df)
                node_df.apply(
                    lambda node_row: self.__add_node_feat(node_row, g=nxg, one_hot=True,
                                                          additional_features=additional_features),
                    axis=1
                )
            else:
                node_df.apply(
                    lambda node_row: self.__add_node_feat(node_row, g=nxg, one_hot=False,
                                                          additional_features=additional_features),
                    axis=1
                )

            # If undirected, sum up the (u, v) connections to the (v, u) connections, for all u and v, otherwise if
            # directed, sum up just the duplicates connection weights (u, v)
            sum_weights = partial(_sum_weights, edge_df=edge_df.copy().set_index(EDGE), directed=directed)
            tmp = pd.DataFrame(edge_df[[EDGE, WEIGHT]].apply(lambda ew: sum_weights(edge=ew[0]), axis=1).to_list())
            edge_df[WEIGHT] = tmp[1]
            """# A bit inefficient, but other solutions seem not to work
            for i in range(0, len(edge_df)):
                edge = edge_df.loc[i, EDGE]
                _, weight = sum_weights(edge=edge)
                edge_df.loc[i, WEIGHT] = weight"""

            # Add edges
            edge_df.apply(
                lambda e: nxg.add_edge(e[EDGE][0], e[EDGE][1], **{WEIGHT: e[WEIGHT], SYNAPSE_TYPE: e[SYNAPSE_TYPE]}),
                axis=1
            )

            # Set the nxgraph attribute
            self.__nxgraph = nxg

        return self.__nxgraph

    @classmethod
    def __set_additional_attributes(cls, node: Union[int, tuple], graph: Union[nx.Graph, nx.DiGraph],
                                    original_connectome: igraph.Graph):
        """
        If the node is not in the original connectome, then set the additional attributes to a random node in the
        original connectome. Otherwise, set the additional attributes to the node in the original connectome.

        :param node: the node
        :type node: int
        :param graph: The graph to be modified
        :type graph: Union[nx.Graph, nx.DiGraph]
        :param original_connectome: The original connectome graph
        :type original_connectome: igraph.Graph
        """

        if isinstance(node, tuple):
            node: int = node[0]

        set_random_flag: bool = False
        node_obj: Optional[igraph.Vertex] = None

        if not set_random_flag:
            try:
                node_obj = original_connectome.vs.find(str(node))
            except (ValueError, IndexError):
                try:
                    node_obj = original_connectome.vs.find(node)
                except (ValueError, IndexError):
                    set_random_flag = True

        # If node is not in the original connectome
        if set_random_flag:

            # Get a random node object from connectome graph
            random_node_index = random.randint(0, original_connectome.vcount())
            node_obj = original_connectome.vs.find(random_node_index)

            # Set the additional attributes according to the selected random cell
            cell_name = f"{DEFAULT_CELL_NAME}{node}"  # default name for new neurons
            cell_class = node_obj[CELL_CLASS]
            soma_pos = node_obj[POS]
            neurotransmitters = node_obj[NEUROTRANSMITTERS]

        # Otherwise set the additional attributes according to the original connectome
        else:
            cell_name = node_obj[CELL_NAME]
            cell_class = node_obj[CELL_CLASS]
            soma_pos = node_obj[POS]
            neurotransmitters = node_obj[NEUROTRANSMITTERS]

        # Set additional node attributes
        graph.nodes[node][ID] = f"n{node}"
        graph.nodes[node][CELL_NAME] = cell_name
        graph.nodes[node][CELL_CLASS] = cell_class
        graph.nodes[node][POS] = soma_pos
        graph.nodes[node][NEUROTRANSMITTERS] = neurotransmitters

    @classmethod
    def from_networkx(cls, graph: Union[nx.Graph, nx.DiGraph], original_connectome: Union[str, ConnectomeReader],
                      path: Optional[str] = None, keep_features: bool = True) -> igraph.Graph:

        # Read original connectome if required
        if type(original_connectome) == str:
            original_connectome_reader = ConnectomeReader(original_connectome)
            original_connectome_reader.read()
        elif isinstance(original_connectome, ConnectomeReader):
            original_connectome_reader = original_connectome
        else:
            raise ValueError(f"original_connectome parameter must be a path or a ConnectomeReader, "
                             f"{type(original_connectome)} given.")

        # TODO: maybe store (serialize) these to make computation faster
        # Get the original connectome nx.Graph object and node/edge info
        original_connectome_igraph = original_connectome_reader.get_running_graph()
        original_connectome = original_connectome_reader.to_networkx(directed=True, one_hot_node_features=False)
        original_connectome_node_df = original_connectome_reader.get_node_dataframe()
        original_connectome_edge_df = original_connectome_reader.get_edge_dataframe()
        synapse_type_node_probs = weighted_edge_type_count(original_connectome_edge_df, normalize=True)
        synapse_type_probs = weighted_value_counts(
            original_connectome_edge_df,
            normalize=True,
            alpha=ALPHA,
            categories=[SYNAPSE_CHEMICAL]
        )

        # Make the graph directed to ensure consistency
        graph = graph.to_directed()

        # Ensure node connectivity
        ensure_weak_connectivity(graph)

        # For each node, assign corresponding role and attributes as a string
        roles = sorted([INTERNEURON_ROLE, SENSOR_ROLE, MOTOR_ROLE])
        nodes = graph.nodes(data=True)
        for node in nodes:
            features = node[1]
            node_id = node[0]

            # Get the role from the given graph if possible and if keep_features is True
            if ROLE in features and keep_features:
                role_one_hot = features[ROLE]
                role = roles[np.argmax(role_one_hot)]  # get str role corresponding to one-hot encoding

            # Otherwise if the original connectome contains the node, get its role and set it
            elif original_connectome.has_node(node_id):
                role = original_connectome_node_df.loc[node_id, ROLE]

            # Otherwise current node is a new node, then get a random node from the original connectome and set its role
            else:
                random_index = random.randint(0, len(original_connectome_node_df))
                role = original_connectome_node_df.iloc[random_index, ROLE]

            # Set the node role
            if role == INTERNEURON_ROLE:
                role = INTERNEURON_ROLE_ORIGINAL  # replace with original interneuron role name
            features[ROLE] = role

            # Set additional attributes to grant consistency
            cls.__set_additional_attributes(node=node, graph=graph, original_connectome=original_connectome_igraph)

        # For each edge, set the corresponding weight and synapse type
        edges = graph.edges
        for u, v in edges:

            # If keep_features is True and the weight is available in the given graph, get it
            if keep_features and WEIGHT in graph[u][v]:
                weight = graph[u][v][WEIGHT]

            # Otherwise, get it from the original connectome if the edge exists in it, adding some gaussian noise
            elif original_connectome.has_edge(u, v):
                weight = np.max([
                    1.0,
                    np.round(original_connectome[u][v][WEIGHT] + np.clip(np.random.normal(0, 5), a_min=-5, a_max=5))
                ])

            # Otherwise, mean the weights of the edges incident to u and v from the original connectome, if both exist
            elif original_connectome.has_node(u) and original_connectome.has_node(v):
                _, weight = node_mean_weight(edge=(u, v), edge_df=original_connectome_edge_df)

            # Otherwise, just put it at random
            else:
                weight = np.round(np.clip(np.random.normal(0, 10), a_min=1, a_max=10))

            graph[u][v][WEIGHT] = weight  # set the weight

            # If keep_features is True and the synapse type is available in the given graph, get it
            if keep_features and SYNAPSE_TYPE in graph[u][v]:
                synapse_type = graph[u][v][SYNAPSE_TYPE]

            # Otherwise, get it from the original connectome if the edge exists in it
            elif original_connectome.has_edge(u, v):
                synapse_type = original_connectome[u][v][SYNAPSE_TYPE]

            # Otherwise, take a synapse type a random one incident to u or v in the original connectome
            # based on the synapse type weighted frequency for that node, if at least one of the node exist
            elif original_connectome.has_node(u):
                synapse_type = np.random.choice(
                    a=[SYNAPSE_CHEMICAL, SYNAPSE_ELECTRIC],
                    p=[synapse_type_node_probs.loc[u, SYNAPSE_CHEMICAL],
                       synapse_type_node_probs.loc[u, SYNAPSE_ELECTRIC]]
                )
            elif original_connectome.has_node(v):
                synapse_type = np.random.choice(
                    a=[SYNAPSE_CHEMICAL, SYNAPSE_ELECTRIC],
                    p=[synapse_type_node_probs.loc[v, SYNAPSE_CHEMICAL],
                       synapse_type_node_probs.loc[v, SYNAPSE_ELECTRIC]]
                )

            # Otherwise, take the type from a random synapse from the original connectome, based on absolute weighted
            # frequency
            else:
                synapse_type = np.random.choice(
                    a=[SYNAPSE_CHEMICAL, SYNAPSE_ELECTRIC],
                    p=[synapse_type_probs.loc[SYNAPSE_CHEMICAL],
                       synapse_type_probs.loc[SYNAPSE_ELECTRIC]]
                )

            graph[u][v][SYNAPSE_TYPE] = synapse_type

        # Construct node list to add and corresponding attribute dictionary
        node_list = []
        node_attr_dict = {}

        # For each node
        for node in nodes:
            node_features = node[1]
            node_id = node[0]

            # Add node id to node list
            node_list.append(f"{node_id}")

            # For each feature
            for feature in node_features:

                # Add the feature to the dict
                if feature not in node_attr_dict:
                    node_attr_dict[feature] = [node_features[feature]]
                else:
                    node_attr_dict[feature].append(node_features[feature])

        # Construct edge list to add and corresponding attribute dictionary
        added = set()  # temporary structure to trace the added edges
        edge_list = []
        edge_attr_dict = {
            WEIGHT: [],
            SYNAPSE_TYPE: []
        }

        # For each edge (u, v)
        for edge in edges:
            u = edge[0]
            v = edge[1]
            e = (f"{u}", f"{v}")
            rev = (f"{v}", f"{u}")

            weight = graph[u][v][WEIGHT]
            synapse_type = graph[u][v][SYNAPSE_TYPE]

            # If connection is undirected (electrical), add both edges to the list if they were not added yet
            if synapse_type == SYNAPSE_ELECTRIC and e not in added and rev not in added:
                # Add (u, v) and (v, u) to the edge_list
                edge_list.append(e)
                edge_list.append(rev)

                # Add attributes for both edges
                edge_attr_dict[WEIGHT].append(weight)
                edge_attr_dict[WEIGHT].append(weight)
                edge_attr_dict[SYNAPSE_TYPE].append(synapse_type)
                edge_attr_dict[SYNAPSE_TYPE].append(synapse_type)

                # Add edges to trace structure
                added.add(e)
                added.add(rev)

            # Otherwise if connection is directed (chemical) and it was not already added
            if synapse_type == SYNAPSE_CHEMICAL and e not in added and rev not in added:

                # If original connectome has (u, v)
                if original_connectome.has_edge(u, v):
                    # Add it to the edge_list and trace structure
                    edge_list.append(e)
                    added.add(e)

                    # Add edge attributes
                    edge_attr_dict[WEIGHT].append(weight)
                    edge_attr_dict[SYNAPSE_TYPE].append(synapse_type)

                # If original connectome has (v, u)
                if original_connectome.has_edge(v, u):
                    # Add it to the edge_list and trace structure
                    edge_list.append(rev)
                    added.add(rev)

                    # Add edge attributes
                    edge_attr_dict[WEIGHT].append(weight)
                    edge_attr_dict[SYNAPSE_TYPE].append(synapse_type)

                # Otherwise, if it is a new edge, choose randomly which to add
                if not original_connectome.has_edge(u, v) and not original_connectome.has_edge(v, u):

                    # Generate random number: 0 means (u, v), 1 means (v, u) and 2 means both
                    rnd = np.random.randint(0, 3)

                    if rnd == 0 or rnd == 2:
                        # Add (u, v) to the edge_list and trace structure
                        edge_list.append(e)
                        added.add(e)

                        # Add edge attributes
                        edge_attr_dict[WEIGHT].append(weight)
                        edge_attr_dict[SYNAPSE_TYPE].append(synapse_type)
                    if rnd == 1 or rnd == 2:
                        # Add (v, u) to the edge_list and trace structure
                        edge_list.append(rev)
                        added.add(rev)

                        # Add edge attributes
                        edge_attr_dict[WEIGHT].append(weight)
                        edge_attr_dict[SYNAPSE_TYPE].append(synapse_type)

        # Construct direct graph using iGraph
        ig = igraph.Graph(directed=True)
        ig.add_vertices(node_list, attributes=node_attr_dict)
        ig.add_edges(edge_list, attributes=edge_attr_dict)

        # If path is given, write the graph to GraphML file
        if path is not None:
            # Write graphml file
            ig.write_graphml(path)
        return ig

    @classmethod
    def from_torch_geometric(cls,
                             graph: tgd.Data,
                             original_connectome: Union[str, ConnectomeReader],
                             path: Optional[str] = None,
                             undirected: bool = True,
                             keep_features: bool = True) -> (igraph.Graph, Union[nx.Graph, nx.DiGraph]):
        """
        Convert a PyTorch Geometric graph to a ConnectomeReader/iGraph object and a NetworkX graph object.

        :param graph: the PyG graph object (must be a single graph, not a batch with size > 1).
        :type graph: tgd.Data
        :param original_connectome: The original connectome file that was used to generate the graph
        :type original_connectome: Union[str, ConnectomeReader]
        :param path: The path to the connectome graphml file to create
        :type path: Optional[str]
        :param undirected: If True, the graph will be converted to an undirected graph by treating each edge as an
            undirected edge with weight equal to the sum of the weights of the corresponding directed edges,
            defaults to True
        :type undirected: bool (optional)
        :param keep_features: Whether to keep the node/edge features from the PyG object, defaults to True
        :type keep_features: bool (optional)

        :return: A ConnectomeReader/iGraph object and a NetworkX graph object
        """

        # Select node/edge features from PyG object (probably not strictly necessary)
        node_features = [ROLE] if ROLE in graph else []
        edge_features = []
        if WEIGHT in graph and keep_features:
            edge_features.append(WEIGHT)
        if SYNAPSE_TYPE in graph and keep_features:
            edge_features.append(SYNAPSE_TYPE)
            graph[SYNAPSE_TYPE] = graph[SYNAPSE_TYPE] if len(graph[SYNAPSE_TYPE]) > 1 else graph[SYNAPSE_TYPE][0]

        # Convert to NetworkX
        nxg = tgu.to_networkx(
            data=graph,
            to_undirected=undirected,
            remove_self_loops=True,
            node_attrs=node_features,
            edge_attrs=edge_features
        )

        # Convert to iGraph/Connectome reader
        ig = cls.from_networkx(
            graph=nxg,
            original_connectome=original_connectome,
            path=path,
            keep_features=keep_features
        )

        return ig, nxg

    def print_edge_prop(self, string_el):
        for i in self.graph_el.es():
            print(str(i[string_el]))

    def print_node_prop(self, string_el):
        for i in self.graph_el.vs():
            print(str(i[string_el]))

    def get_running_graph(self):
        if self.graph_el != 0:
            return self.graph_el
        else:
            return None

    def put_color_by_cell_class(self):
        list_class = []
        for i in self.graph_el.vs():
            if i[ROLE] == SENSOR_ROLE:
                list_class.append(0)
            elif i[ROLE] == MOTOR_ROLE:
                list_class.append(1)
            else:
                list_class.append(2)

        self.graph_el.vs[COLOR] = np.asarray(list_class)
        return self.graph_el, list_class

# summary(elegans_graph.put_color_by_cell_class())

# To characterize the direct impact that one neuron can have on
# another, we quantify the strength of connections by the
# multiplicity, m ij , between neurons i and j, which is the number
# of synaptic contacts (here gap junctions) connecting i to j. The
# degree treats synaptic connections as binary, whereas the
# multiplicity, also called edge weight, quantifies the number of contacts.
# elegans_graph.print_edge_prop(WEIGHT)
# elegans_graph.print_node_prop(COLOR)
