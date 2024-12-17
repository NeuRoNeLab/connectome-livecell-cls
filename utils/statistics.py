from multiprocessing.connection import Connection
from typing import Union, Final, Optional, List, Set, Sequence, Mapping
import numpy as np
import pandas as pd
import scipy.spatial.distance as distance
import scipy.stats as stats
import networkx as nx
from scipy.stats import gaussian_kde
from tqdm import tqdm
from utils.connectome_reader import ConnectomeReader
from utils.constants import ORIGINAL_CONNECTOME
from multiprocessing import Process as Thread, Pipe


# Stats
JACCARD: Final[str] = "jaccard"
SHANNON: Final[str] = "jensen_shannon"
WASSERSTEIN: Final[str] = "wasserstein"
BHATTA: Final[str] = "bhatta"
HAMMING: Final[str] = "hamming"
KULSINSKI: Final[str] = "kulsinski"
LK_NORM: Final[str] = "lknorm"
AVG_DEGREE: Final[str] = "avg_degree"
MEDIAN_DEGREE: Final[str] = "median_degree"
EDGE_NUM: Final[str] = "edge_num"
SIGMA: Final[str] = "sigma"
OMEGA: Final[str] = "omega"
GRAPH_EDIT: Final[str] = "graph_edit"
FILENAME: Final[str] = "filename"
_STATS: Final[Set[str]] = {
    JACCARD,
    WASSERSTEIN,
    SHANNON,
    BHATTA,
    HAMMING,
    KULSINSKI,
    LK_NORM,
    AVG_DEGREE,
    MEDIAN_DEGREE,
    EDGE_NUM,
    SIGMA,
    OMEGA,
    GRAPH_EDIT
}
DEFAULT_STATS: Final[frozenset[str]] = frozenset({
    JACCARD,
    SHANNON,
    WASSERSTEIN,
    BHATTA, HAMMING,
    KULSINSKI,
    LK_NORM,
    AVG_DEGREE,
    MEDIAN_DEGREE,
    EDGE_NUM
})

# Params
N_ITER_DEFAULT: Final[int] = 50
P_DEFAULT: Final[int] = 2
N_BINS_DEFAULT: Final[int] = 10
N_STEPS_BHATTA_CONTINUOUS: Final[int] = 200
CONTINUOUS: Final[str] = "continuous"
NOISELESS: Final[str] = "noiseless"
HIST: Final[str] = "hist"
AUTOHIST: Final[str] = "autohist"


def __get_density(x, cov_factor=0.1):
    # Produces a continuous density function for the data in 'x'.
    # Some benefit may be gained from adjusting the cov_factor.
    density = gaussian_kde(x)
    density.covariance_factor = lambda: cov_factor
    density._compute_covariance()
    return density


def jensen_shannon(adj0: np.ndarray, adj1: np.ndarray) -> float:
    adj0 = adj0.flatten()
    adj1 = adj1.flatten()
    return distance.jensenshannon(adj0, adj1, base=2)


def bhatta_distance(x1: np.ndarray, x2: np.ndarray, method: str = CONTINUOUS):
    """
    Computes the Bhattacharyya distance between x1 and x2, which should be 1D numpy arrays representing the same
    feature in two separate classes.
    """

    # Flatten the arrays
    x1 = x1.flatten()
    x2 = x2.flatten()

    # Combine x1 and x2, we'll use it later:
    combined_x = np.concatenate((x1, x2))

    if method == NOISELESS:
        # This method works well when the feature is qualitative (rather than quantitative). Each unique value is
        # treated as an individual bin.
        u_x = np.unique(combined_x)
        a1 = len(x1) * (max(combined_x) - min(combined_x)) / len(u_x)
        a2 = len(x2) * (max(combined_x) - min(combined_x)) / len(u_x)
        bht = 0
        for x in u_x:
            p1 = (x1 == x).sum() / a1
            p2 = (x2 == x).sum() / a2
            bht += np.sqrt(p1 * p2) * (max(combined_x) - min(combined_x)) / len(u_x)

    elif method == HIST:
        # Cluster the values into a hardcoded number of bins (This is sensitive to N_BINS)
        # Bin the values:
        h1 = np.histogram(x1, bins=N_BINS_DEFAULT, range=(min(combined_x), max(combined_x)), density=True)[0]
        h2 = np.histogram(x2, bins=N_BINS_DEFAULT, range=(min(combined_x), max(combined_x)), density=True)[0]
        # Calculate coefficient from bin densities:
        bht = 0
        for i in range(N_BINS_DEFAULT):
            p1 = h1[i]
            p2 = h2[i]
            bht += np.sqrt(p1 * p2) * (max(combined_x) - min(combined_x)) / N_BINS_DEFAULT

    elif method == AUTOHIST:
        # Cluster the values into bins automatically set by np.histogram:
        # Create bins from the combined sets:
        # bins = np.histogram(cX, bins='fd')[1]
        bins = np.histogram(combined_x, bins='doane')[1]  # Seems to work best
        # bins = np.histogram(cX, bins='auto')[1]

        h1 = np.histogram(x1, bins=bins, density=True)[0]
        h2 = np.histogram(x2, bins=bins, density=True)[0]

        # Calculate coefficient from bin densities:
        bht = 0
        for i in range(len(h1)):
            p1 = h1[i]
            p2 = h2[i]
            bht += np.sqrt(p1 * p2) * (max(combined_x) - min(combined_x)) / len(h1)

    elif method == CONTINUOUS:
        # Use a continuous density function to calculate the coefficient (most consistent, but also slightly slow)
        # Get density functions:
        d1 = __get_density(x1)
        d2 = __get_density(x2)
        # Calc coefficients
        xs = np.linspace(min(combined_x), max(combined_x), N_STEPS_BHATTA_CONTINUOUS)
        bht = 0
        for x in xs:
            p1 = d1(x)
            p2 = d2(x)
            bht += np.sqrt(p1 * p2) * (max(combined_x) - min(combined_x)) / N_STEPS_BHATTA_CONTINUOUS

    else:
        raise ValueError("The value of the 'method' parameter does not match any known method")

    # Lastly, convert the coefficient into distance:
    if bht == 0:
        return np.Inf
    else:
        return float(-np.log(bht))


def wasserstein_distance(adj0: np.ndarray, adj1: np.ndarray) -> float:
    adj0 = adj0.flatten()
    adj1 = adj1.flatten()
    return stats.wasserstein_distance(u_values=adj0, v_values=adj1, u_weights=None, v_weights=None)


def hamming_distance(adj0: np.ndarray, adj1: np.ndarray) -> float:
    adj0 = adj0.flatten()
    adj1 = adj1.flatten()
    return distance.hamming(adj0, adj1)


def jaccard_distance(adj0: np.ndarray, adj1: np.ndarray) -> float:
    adj0 = adj0.flatten()
    adj1 = adj1.flatten()
    # noinspection PyTypeChecker
    return distance.jaccard(adj0, adj1)


def kulsinski_distance(adj0: np.ndarray, adj1: np.ndarray) -> float:
    adj0 = adj0.flatten()
    adj1 = adj1.flatten()
    # noinspection PyTypeChecker
    return distance.kulsinski(adj0, adj1)


def lk_norm_distance(adj0: np.ndarray, adj1: np.ndarray, k: int = 2) -> float:
    adj0 = adj0.flatten()
    adj1 = adj1.flatten()
    return distance.minkowski(adj0, adj1, p=k)


def average_degree(g0: Union[np.ndarray, nx.Graph]) -> float:
    if isinstance(g0, np.ndarray):
        degrees = np.sum(g0, axis=1)
    else:
        degrees = np.sum(nx.to_numpy_array(g0), axis=1)
    return np.sum(degrees) / len(degrees)


def median_degree(g0: Union[np.ndarray, nx.Graph]) -> float:
    if isinstance(g0, np.ndarray):
        degrees = np.sum(g0, axis=1)
    else:
        degrees = np.sum(nx.to_numpy_array(g0), axis=1)
    return float(np.median(degrees))


def edge_number(g0: Union[nx.DiGraph, np.ndarray]) -> int:
    if isinstance(g0, nx.Graph):
        return nx.number_of_edges(g0)
    else:
        return int(np.sum(g0))


def small_world_sigma(g: Union[np.ndarray, nx.DiGraph]) -> float:
    if isinstance(g, np.ndarray):
        g = nx.from_numpy_matrix(g, create_using=nx.DiGraph())
    return nx.sigma(g.to_undirected())


def small_world_omega(g: Union[np.ndarray, nx.DiGraph]) -> float:
    if isinstance(g, np.ndarray):
        g = nx.from_numpy_matrix(g, create_using=nx.DiGraph())
    return nx.omega(g.to_undirected())


def approx_graph_edit_distance(g0: Union[np.ndarray, nx.DiGraph], g1: Union[np.ndarray, nx.DiGraph],
                               n_iter: int = N_ITER_DEFAULT) -> float:
    """Computes approximate the graph edit distance between two graphs. NP-hard and very costly to compute."""

    if isinstance(g0, np.ndarray):
        g0 = nx.from_numpy_matrix(g0, create_using=nx.DiGraph())

    if isinstance(g1, np.ndarray):
        g1 = nx.from_numpy_matrix(g1, create_using=nx.DiGraph())

    ged = np.Inf
    for i in range(0, n_iter):
        try:
            ged = next(nx.optimize_graph_edit_distance(g0, g1))
        except StopIteration:
            return ged

    return ged


def stats_dict(g0: Union[nx.DiGraph, tuple[nx.DiGraph, np.ndarray]],
               g1: Optional[Union[nx.DiGraph, tuple[nx.DiGraph, np.ndarray]]] = None,
               stats_to_compute: set[str] = DEFAULT_STATS,
               p: Optional[int] = P_DEFAULT,
               n_iter: Optional[int] = N_ITER_DEFAULT) -> dict[str, float]:
    """
    Takes in a graph and a set of statistics to compute, and returns a dictionary of the computed statistics.

    :param g0: The graph to compare to the original connectome.
    :type g0: Union[nx.DiGraph, tuple[nx.DiGraph, np.ndarray]]
    :param g1: The original connectome graph.
    :type g1: Optional[Union[nx.DiGraph, tuple[nx.DiGraph, np.ndarray]]]
    :param stats_to_compute: a set of strings that are the names of the statistics you want to compute
    :type stats_to_compute: set[str]
    :param p: The p-norm to use for the Lk-norm distance
    :type p: Optional[int]
    :param n_iter: number of iterations to run the graph edit distance algorithm
    :type n_iter: Optional[int]
    :return: A dictionary of the statistics computed for the graph.
    """
    for stat in stats_to_compute:
        if stat not in _STATS:
            raise ValueError(f"Stat {stat} not in {_STATS}")

    # Current graph
    if isinstance(g0, tuple):
        adj0 = g0[1]
        g0 = g0[0]
    else:
        adj0 = nx.to_numpy_array(g0)

    # Original connectome graph
    if g1 is not None and isinstance(g1, tuple):
        adj1 = g1[1]
        g1 = g1[0]
    elif g1 is not None:
        adj1 = nx.to_numpy_array(g1)
    else:
        adj1 = None

    # Create binary counterparts
    adj0_bin = np.clip(adj0, 0, 1)
    adj1_bin = None
    if adj1 is not None:
        adj1_bin = np.clip(adj1, 0, 1)

    stats_dct = {}
    if JACCARD in stats_to_compute:
        if g1 is not None:
            stats_dct[JACCARD] = jaccard_distance(adj0_bin, adj1_bin)
        else:
            stats_dct[JACCARD] = 0.0
    if SHANNON in stats_to_compute:
        if g1 is not None:
            stats_dct[SHANNON] = jensen_shannon(adj0, adj1)
        else:
            stats_dct[SHANNON] = 0.0
    if WASSERSTEIN in stats_to_compute:
        if g1 is not None:
            stats_dct[WASSERSTEIN] = wasserstein_distance(adj0, adj1)
        else:
            stats_dct[WASSERSTEIN] = 0.0
    if HAMMING in stats_to_compute:
        if g1 is not None:
            stats_dct[HAMMING] = hamming_distance(adj0_bin, adj1_bin)
        else:
            stats_dct[HAMMING] = 0.0
    if KULSINSKI in stats_to_compute:
        if g1 is not None:
            stats_dct[KULSINSKI] = kulsinski_distance(adj0_bin, adj1_bin)
        else:
            stats_dct[KULSINSKI] = 0.0
    if LK_NORM in stats_to_compute:
        if g1 is not None:
            stats_dct[LK_NORM] = lk_norm_distance(adj0, adj1, k=p)
        else:
            stats_dct[LK_NORM] = 0.0
    if BHATTA in stats_to_compute:
        if g1 is not None:
            stats_dct[BHATTA] = bhatta_distance(adj0, adj1, method=CONTINUOUS)
        else:
            stats_dct[BHATTA] = 0.0
    if AVG_DEGREE in stats_to_compute:
        stats_dct[AVG_DEGREE] = average_degree(adj0)
    if MEDIAN_DEGREE in stats_to_compute:
        stats_dct[MEDIAN_DEGREE] = median_degree(adj0)
    if EDGE_NUM in stats_to_compute:
        stats_dct[EDGE_NUM] = edge_number(g0)
    if SIGMA in stats_to_compute:
        stats_dct[SIGMA] = small_world_sigma(g0)
    if OMEGA in stats_to_compute:
        stats_dct[OMEGA] = small_world_omega(g0)
    if GRAPH_EDIT in stats_to_compute:
        if g1 is not None:
            stats_dct[GRAPH_EDIT] = approx_graph_edit_distance(g0, g1, n_iter=n_iter)
        else:
            stats_dct[GRAPH_EDIT] = 0.0

    return stats_dct


def _stats_dataframe(filenames: List[str],
                     original_connectome_path: str = ORIGINAL_CONNECTOME,
                     stats_to_compute: set[str] = DEFAULT_STATS,
                     p: Optional[int] = P_DEFAULT,
                     n_iter: Optional[int] = N_ITER_DEFAULT,
                     weighted: bool = False,
                     use_tqdm: bool = True,
                     results_connection: Connection = None) -> pd.DataFrame:
    """
    Takes a list of graphml filenames, reads each graphml file, computes the given statistics for each graphml file, and
    returns a dataframe containing the statistics for each graphml file

    :param filenames: paths to the connectomes.
    :type filenames: List[str]
    :param original_connectome_path: path to the original connectome.
    :type original_connectome_path: str
    :param stats_to_compute: the stats and distances to compute as a set.
    :type stats_to_compute: set[str]
    :param p: p-parameter for the Minkowski (L_p) norm distance, ignored if not required.
    :type p: int
    :param n_iter: maximum number of approximation iterations fr the graph edit distance.
    :type n_iter: int
    :param weighted: whether to use edge weights in the distance computations.
    :type weighted: bool
    :param use_tqdm: whether to use tqdm to show a progress bar for the metric calculation.
    :type use_tqdm: bool
    :param results_connection: pipe connection for the results.
    :type results_connection: Connection
    :return: A dataframe with the stats for each graphml file.
    """
    # Read the original connectome and get its adjacency matrix
    cr_original = ConnectomeReader(original_connectome_path)
    cr_original.read()
    g_original = cr_original.to_networkx(directed=True)
    attribute = "weight" if weighted else None
    adj_original = np.array(cr_original.get_running_graph().get_adjacency(attribute=attribute).data)
    # adj_original = nx.to_numpy_array(g_original)

    # For each given graphml filename
    stats_dicts: list[dict[str, Union[float, str]]] = []
    iterable = tqdm(filenames, desc="Generating stats dataframes...") if use_tqdm else filenames
    for filename in iterable:
        # Read the graphml and convert to NetworkX format
        cr = ConnectomeReader(filename)
        cr.read()
        adj = np.array(cr.get_running_graph().get_adjacency(attribute=attribute).data)
        g = cr.to_networkx(directed=True)

        # Get the stats as a dictionary, add the filename to it and put it into the stats dict list
        stats_dct = stats_dict(
            g0=(g, adj),
            g1=(g_original, adj_original),
            stats_to_compute=stats_to_compute,
            p=p,
            n_iter=n_iter
        )
        stats_dct[FILENAME] = filename
        stats_dicts.append(stats_dct)

    # Convert the list to a dataframe
    df = pd.DataFrame.from_records(data=stats_dicts)

    # Write result dataframe to Pipe object
    if results_connection is not None:
        results_connection.send(df)
        results_connection.close()

    return df


def stats_dataframe(filenames: List[str],
                    original_connectome_path: str = ORIGINAL_CONNECTOME,
                    stats_to_compute: Union[set[str], frozenset[str]] = DEFAULT_STATS,
                    p: Optional[int] = P_DEFAULT,
                    n_iter: Optional[int] = N_ITER_DEFAULT,
                    weighted: bool = False,
                    use_tqdm: bool = True,
                    n_jobs: int = 1) -> pd.DataFrame:
    """
    Computes statistics for the given connectomes comparing them to the original one using multiprocessing.

    :param filenames: paths to the connectomes.
    :type filenames: List[str]
    :param original_connectome_path: path to the original connectome.
    :type original_connectome_path: str
    :param stats_to_compute: the stats and distances to compute as a set.
    :type stats_to_compute: set[str]
    :param p: p-parameter for the Minkowski (L_p) norm distance, ignored if not required.
    :type p: int
    :param n_iter: maximum number of approximation iterations fr the graph edit distance.
    :type n_iter: int
    :param weighted: whether to use edge weights in the distance computations.
    :type weighted: bool
    :param use_tqdm: whether to use tqdm to show a progress bar for the metric calculation.
    :type use_tqdm: bool
    :param n_jobs: maximum number of jobs to run in parallel to compute the required stats.
    :type n_jobs: int

    :return: a dataframe containing a column for each required statistic.
    :rtype: pd.DataFrame
    """
    if n_jobs <= 0:
        raise ValueError("n_jobs must be strictly positive")

    if n_jobs > 1:
        n_filenames_per_job = len(filenames) // n_jobs
        threads = []
        results = [None for _ in range(0, n_jobs)]
        pipes = [Pipe() for _ in range(0, n_jobs)]

        # Create jobs from 1 to n_jobs-1 splitting
        for i in range(0, n_jobs - 1):
            target_filenames = filenames[i * n_filenames_per_job:i * n_filenames_per_job + n_filenames_per_job]
            pipe_write_end_connection = pipes[i][1]
            thread = Thread(
                target=_stats_dataframe,
                args=(target_filenames, original_connectome_path, stats_to_compute, p, n_iter, weighted, use_tqdm,
                      pipe_write_end_connection),
                name=f"grid_search_thread{i}"
            )
            threads.append(thread)

        # Create last job
        target_filenames = filenames[(n_jobs - 1) * n_filenames_per_job:]
        pipe_write_end_connection = pipes[n_jobs - 1][1]
        last_thread = Thread(
            target=_stats_dataframe,
            args=(target_filenames, original_connectome_path, stats_to_compute, p, n_iter, weighted, use_tqdm,
                  pipe_write_end_connection),
            name=f"grid_search_thread{n_jobs - 1}"
        )
        threads.append(last_thread)

        # Execute threads
        for i in range(0, n_jobs):
            threads[i].start()

        # Read results through pipes
        for i in range(0, n_jobs):
            pipe_read_end_connection = pipes[i][0]
            results[i] = pipe_read_end_connection.recv()

        # Wait end of execution
        for i in range(n_jobs):
            threads[i].join()

        # Combine results
        final_result = results[0]
        for i in range(1, n_jobs):
            final_result = pd.concat([final_result, results[i]], axis=0, ignore_index=True)

        return final_result

    else:
        return _stats_dataframe(filenames=filenames, original_connectome_path=original_connectome_path,
                                stats_to_compute=stats_to_compute, p=p, n_iter=n_iter, use_tqdm=use_tqdm,
                                results_connection=None)


def quantile_select_subset(stats_path: str,
                           column: str = HAMMING,
                           quantiles: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
                           elements_per_quantile: Sequence[int] = (2, 1, 1, 2),
                           return_dataframe: bool = False) -> Union[pd.DataFrame, Mapping[float, List[str]]]:
    if len(quantiles) - 1 != len(elements_per_quantile):
        raise ValueError(f"Quantiles should be elements_per_quantile + 1, {len(quantiles)} and "
                         f"{len(elements_per_quantile)} given.")

    df = pd.read_csv(stats_path, index_col=False)
    quantile_thresholds = list(df[column].quantile(q=quantiles))
    quantile_values = {}

    # For each quantile threshold
    for i, quantile_threshold in enumerate(quantile_thresholds):
        if i < len(quantile_thresholds) - 1:
            # Get the elements in that quantile
            tmp = df[(df[column] >= quantile_threshold) & (df[column] < quantile_thresholds[i + 1])]
            # Sample from them
            quantile_values[quantile_threshold] = tmp.sample(n=elements_per_quantile[i])

    if return_dataframe:
        concatenated = pd.concat([quantile_values[p] for p in quantile_values])
        return concatenated
    return {p: list(quantile_values[p][FILENAME]) for p in quantile_values}
