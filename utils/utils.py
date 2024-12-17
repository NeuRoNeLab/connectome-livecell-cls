import random
from functools import wraps
from time import time
from typing import Union, List, Final, Iterable, Optional, Any, Dict, Set
import numpy as np
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.constants import VAL_SIZE, TEST_SIZE, RANDOM_SEED

ALPHA: Final[float] = 1.0  # laplacian smoothing alpha
_X_MIN: Final[int] = 0
_X_MAX: Final[int] = 2
_Y_MIN: Final[int] = 0
_Y_MAX: Final[int] = 2


def train_test_validation_split(dataset: Union[pd.DataFrame, List[str]], val_size: float = VAL_SIZE,
                                test_size: float = TEST_SIZE, random_seed: int = RANDOM_SEED) -> \
        tuple[Union[pd.DataFrame, List[str]], Union[pd.DataFrame, List[str]], Union[pd.DataFrame, List[str]]]:
    """
    Splits a dataframe or a list dataset into train, validation and test sets.

    :param dataset: the dataframe to split
    :type dataset: pd.DataFrame
    :param val_size: the ratio of the validation set to the entire dataset
    :type val_size: float
    :param test_size: the ratio of the test set to the entire dataset
    :type test_size: float
    :param random_seed: The random seed to use for the split
    :type random_seed: int
    :return: A tuple of three dataframes.
    """

    if type(dataset) == list:
        df = pd.DataFrame(dataset)
    else:
        df = dataset

    df_train, df_val = train_test_split(df, test_size=val_size, random_state=random_seed)
    df_train, df_test = train_test_split(df_train, test_size=val_size / (1 - test_size), random_state=random_seed)

    if type(dataset) == list:
        return df_train[0].to_list(), df_val[0].to_list(), df_test[0].to_list()
    else:
        return df_train, df_val, df_test


def weighted_value_counts(df: pd.DataFrame, normalize: bool = False, alpha: float = ALPHA,
                          categories: Optional[Iterable] = None):
    """
    Takes a dataframe with two columns, the first one being the categorical variable and the second one being the
    weight, and returns a series with the weighted value counts of the categorical variable, applying normalization and
    Laplacian smoothing if required.

    :param df: the dataframe to be used
    :type df: pd.DataFrame
    :param normalize: If True, the object returned will contain the relative frequencies of the unique values, defaults
        to False
    :type normalize: bool (optional)
    :param alpha: Laplacian smoothing parameter, applied only if normalization is True, defaults to 1
    :type alpha: float (optional)
    :param categories: additional categories to be added (they may not be included in the dataframe)
    :type categories: Optional[Iterable] (optional)

    :return: A series with the weighted counts of the values in the first column of the dataframe.
    """

    # Get columns
    catg_col = df.columns[0]
    weight_col = df.columns[1]

    # Aggregate categorical column, summing weights
    tmp = df[[catg_col, weight_col]].groupby(catg_col).agg({weight_col: 'sum'}).sort_values(weight_col, ascending=False)

    # Create series with results
    s = pd.Series(index=tmp.index, data=tmp[weight_col], name=catg_col)

    # Set to 0 the counts of the missing categories
    if categories is not None:
        for categorical_value in categories:
            if categorical_value not in s:
                s.loc[categorical_value] = 0.0

    # Normalize values if required, applying Laplacian smoothing if alpha > 0
    if normalize:
        # Add alpha anyway, even if it is 0 it wont have any effect
        s = (s + alpha) / (df[weight_col].sum() + len(s) * alpha)

    return s


def plot_graph(g, weighted: bool = False):
    if len(nx.get_node_attributes(g, "pos", )) == 0:
        pos = {i: (random.gauss(_X_MIN, _X_MAX), random.gauss(_Y_MIN, _Y_MAX)) for i in g.nodes}
        nx.set_node_attributes(g, pos, "pos")

    edge_x = []
    edge_y = []
    for edge in g.edges():
        x0, y0 = g.nodes[edge[0]]['pos']
        x1, y1 = g.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in g.nodes():
        x, y = g.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title=f'Node # of {"Weighted" if weighted else ""} Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        )
    )

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(g.adjacency()):
        features = g.nodes[node].copy()  # get node features
        del features['pos']  # remove position
        text = f'node {node}, {features if len(features) > 0 else ""}'
        if not weighted:
            text += f', # of connections: ' + str(len(adjacencies[1]))
            node_adjacencies.append(len(adjacencies[1]))
        else:
            weighted_edges_sum = np.sum([adjacencies[1][i]['weight'] for i in adjacencies[1]])
            text += f', # of weighted connections: {weighted_edges_sum}'
            node_adjacencies.append(weighted_edges_sum)

        node_text.append(text)

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    # noinspection PyTypeChecker
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Graph plot',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.show()


class ListDict(object):
    def __init__(self, initial_values: Optional[Iterable] = None):
        """
        This class represent a list of items paired with a dictionary, which maps each item (that must be hash-able) to
        the corresponding position, hallowing to do operations like delete, retrieve and insert in O(1) time, as well as
        random selection.

        :param initial_values: values to initialize the data structure with
        :type initial_values: Optional[Iterable]
        """
        self.__item_to_position = {}
        self.__items = []

        if initial_values is not None:
            self.extend(initial_values)

    def extend(self, items: Iterable):
        """
        Add a list of items to the list-dict.

        :param items: Iterable containing items to add to the list-dict
        :type items: Iterable
        """
        for item in items:
            self.add(item)

    def add(self, item):
        """
        Adds an element to the list-dict in O(1) time.

        :param item: The item to add to the list
        :return: The position of the item in the list.
        """
        if item in self.__item_to_position:
            return
        self.__items.append(item)
        self.__item_to_position[item] = len(self.__items) - 1

    def remove(self, item):
        """
        Removes the item from the list in O(1) time.

        :param item: The item to remove from the set
        """
        position = self.__item_to_position.pop(item)
        last_item = self.__items.pop()
        if position != len(self.__items):
            self.__items[position] = last_item
            self.__item_to_position[last_item] = position

    def remove_multiple(self, items: Iterable):
        """
        Removes a list of items from the list-dict.

        :param items: Iterable containing items to remove to the list-dict
        :type items: Iterable
        """
        for item in items:
            self.remove(item)

    def choose_random(self, sample_size: int = 1):
        """
        Returns a random item from the list-dict of items in O(k) time, where k is the required sample size.

        :param sample_size: sample size.

        :return: A list with sample_size random items from the list-dict if sample_size > 1, a single random item with
            if sample_size = 1.
        """
        random_items = []
        for _ in range(0, sample_size):
            random_items.append(random.choice(self.__items))
        if sample_size == 1:
            return random_items[0]
        else:
            return random_items

    def choose_random_distinct_couple(self) -> tuple[Any, Any]:
        """
        Returns 2 distinct random items from the list-dict of items in O(1) time (unless the list has only 1 element).

        :return: A tuple with two items.
        """
        if len(self.__items) < 2:
            raise ValueError(f"Cannot choose more than 2 distinct items in a list-dict with {len(self.__items)} items")
        i = random.randint(0, len(self.__items) - 1)
        random_item0 = self.__items[i]
        random_item1 = self.__items[(i + 1) % len(self.__items)]

        return random_item0, random_item1

    def position(self, item) -> int:
        """
        It returns the position of the item in the list-dict.

        :param item: The item to find the position of
        :return: The position of the item in the list-dict.
        """
        return self.__item_to_position[item]

    def to_list(self) -> list:
        """
        Returns a list containing all the items in the list-dict.

        :return: the contents of the list-dict as a list.
        :rtype: list
        """
        return list(self.__items)

    def __getitem__(self, index):
        """
        Returns the item at the given index.

        :param index: The index of the item you want to retrieve
        :return: The item at the given index.
        """
        return self.__items[index]

    def __contains__(self, item):
        """
        If the item is in the dictionary, return True. Otherwise, return False. This runs in O(1) time.

        :param item: The item to check for membership
        :return: The position of the item in the list.
        """
        return item in self.__item_to_position

    def __iter__(self):
        """
        Returns an iterator object that can be used to iterate over the items in the list-dict.
        :return: The iter() function is being called on the list of items.
        """
        return iter(self.__items)

    def __len__(self):
        """
        The function returns the length of the list-dict of items.
        :return: The length of the list-dict.
        """
        return len(self.__items)

    def __str__(self):
        return str(self.__items) + "\n" + str(self.__item_to_position)


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te - ts))
        return result

    return wrap


@timing
def timed_model_call(model, *args, **kwargs):
    return model(*args, **kwargs)


class SerializableConfig(dict):

    def __getattr__(self, item):
        if item in self:
            return self.__getitem__(item)
        else:
            return super().__getattribute__(item)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)


def _dict_to_dot_strings_rec(current_key: str, value: Any, leave_as_dict_keys: Set[str] = frozenset([])) -> List[str]:
    """
    Recursively converts a dictionary into a list of dot-separated strings,
    where each string represents a path from the root to a leaf in the dictionary.

    Args:
        current_key (str): The current key in the dictionary traversal.
        value (Any): The current value in the dictionary traversal.

    Returns:
        List[str]: A list of strings, each representing a root-to-leaf path
        in the dictionary, with keys separated by dots and the final leaf
        value appended.

    Examples:
        >>> _dict_to_dot_strings_rec('root', {'a': 1, 'b': {'c': 2, 'd': "aaa"}})
        >>> ['root.a 1', 'root.b.c 2', 'root.b.d aaa']
    """
    # Base case of the DFS
    if not isinstance(value, dict) or current_key in leave_as_dict_keys:
        return [f"{current_key} {value}"]

    # Get all the keys
    child_keys = list(value.keys())

    parsed_key_strings = []
    # For each child key
    for child_key in child_keys:

        # Recursively call the function on the child key
        child_parsed_key_strings = _dict_to_dot_strings_rec(
            current_key=child_key,
            value=value[child_key],
            leave_as_dict_keys=leave_as_dict_keys
        )

        # For each child parsed subkey
        for child_parsed_key_string in child_parsed_key_strings:
            # Add the current key as a prefix with "." as separator to all the returned key representing root-leaf paths
            parsed_key_string = f"{current_key}.{child_parsed_key_string}"

            # Add the new string "
            parsed_key_strings.append(parsed_key_string)

    return parsed_key_strings


def dict_to_dot_strings(dict_: Dict[str, Any],
                        global_key: str = "",
                        leave_as_dict_keys: Optional[Set[str]] = None) -> str:
    """
    Converts a dictionary into a single string with dot-separated paths
    from the root to each leaf. Each path is separated by a space.

    Args:
        dict_ (Dict[str, Any]): The dictionary to convert.
        global_key (str, optional): The global key prefix to start the paths
        from. Defaults to an empty string.

    Returns:
        str: A single string containing all root-to-leaf paths in the
        dictionary, with keys separated by dots and paths separated by spaces.

    Examples:
        >>> dict_to_dot_strings({'a': 1, 'b': {'c': 2, 'd': 'aaa'}})
        >>> 'a 1 b.c 2 b.d aaa'

        >>> dict_to_dot_strings({'a': 1, 'b': {'c': 2, 'd': 'aaa'}}, global_key='root')
        >>> 'root.a 1 root.b.c 2 root.b.d aaa'
    """
    leave_as_dict_keys = leave_as_dict_keys if leave_as_dict_keys is not None else set()

    dot_strings = _dict_to_dot_strings_rec(current_key=global_key, value=dict_, leave_as_dict_keys=leave_as_dict_keys)
    concatenated_dot_strings = " ".join(dot_strings)
    return concatenated_dot_strings
