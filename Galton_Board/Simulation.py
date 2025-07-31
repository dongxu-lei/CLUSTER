# Importing essential libraries
import pymc as pm  # PyMC: probabilistic programming for Bayesian statistical modelling
import numpy as np  # NumPy: fundamental package for numerical computations in Python
import xarray as xr  # xarray: N-dimensional labelled arrays and datasets
from tqdm import tqdm  # tqdm: progress bar for iterables
from matplotlib import pyplot as plt  # matplotlib: plotting library
import pickle  # pickle: Python object serialisation
import os
import sys

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

# Change the working directory to the script's directory
os.chdir(script_dir)

# # Snowmake Test
# if __name__ == "__main__":
#     test_data = np.random.randn(10)
#     with open("Galton_Dataset.pickle", "wb") as file:
#         pickle.dump(test_data, file)

# exit(0)

# Function to simulate stick-breaking process for Dirichlet process realisations
def stick_breaking(
    alpha: float, rng: np.random.Generator, size: int = 100
) -> np.ndarray:
    """
    Implements the stick-breaking construction for generating a Dirichlet process realisation.

    Parameters:
        alpha (float): Concentration parameter; must be strictly positive.
        rng (np.random.Generator): NumPy random number generator.
        size (int): Number of components (default: 100).

    Returns:
        np.ndarray: Vector of weights approximately summing to unity.
    """
    if alpha <= 0:
        raise ValueError("Alpha must be positive")

    # Generate Beta-distributed samples
    betas = rng.beta(1, alpha, size=size)

    # Initialise arrays in log-space for numerical stability
    log_weights = np.zeros(size)
    log_remaining = 0.0  # log(1.0)

    for i in range(size):
        log_beta = np.log(betas[i])
        log_one_minus_beta = np.log1p(-betas[i])  # Numerically stable log(1 - beta)

        if i == 0:
            log_weights[i] = log_beta
        else:
            log_weights[i] = log_beta + log_remaining

        log_remaining += log_one_minus_beta  # Update log of remaining stick

    # Convert from log-space and normalise
    weights = np.exp(log_weights)
    weights /= np.sum(weights)

    return weights


# Function to generate samples from the Dirichlet distribution
def dirichlet(
    alpha: np.ndarray, rng: np.random.Generator, size: tuple[int, ...]
) -> np.ndarray:
    """
    Generates random vectors from a Dirichlet distribution.

    Parameters:
        alpha (np.ndarray): Vector of concentration parameters.
        rng (np.random.Generator): NumPy random generator.
        size (tuple[int, ...]): Desired output shape.

    Returns:
        np.ndarray: Dirichlet samples with the final axis summing to one.
    """
    gamma_samples: np.ndarray = rng.gamma(shape=alpha, scale=1.0, size=size)
    return gamma_samples / gamma_samples.sum(axis=-1, keepdims=True)


# Function to generate synthetic data resembling a Galton board
def generate_data(
    r: float,
    c: float,
    n_data: int,
    n_cell: int,
    n_funnel: int,
    rng: np.random.Generator,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Simulates data according to a probabilistic Galton board framework.

    Parameters:
        r (float): Parameter influencing randomness.
        c (float): Parameter influencing clustering.
        n_data (int): Number of data points.
        n_cell (int): Number of target categories or bins.
        n_funnel (int): Number of intermediate distributions.
        rng (np.random.Generator): Random number generator.

    Returns:
        Tuple of xarray.DataArray: Lid openness matrix `U` and observed outcomes `X`.
    """
    # Funnel weights drawn via stick-breaking
    W: xr.DataArray = xr.DataArray(
        stick_breaking(alpha=1 / c, rng=rng, size=n_funnel),
        dims="Funnel",
        coords={"Funnel": np.arange(n_funnel)},
    )

    # Per-funnel cell probabilities from Dirichlet
    P: xr.DataArray = xr.DataArray(
        rng.dirichlet(alpha=np.ones(n_cell), size=n_funnel),
        dims=("Funnel", "Cell"),
        coords={"Funnel": np.arange(n_funnel), "Cell": np.arange(n_cell)},
    )

    # Uniform openness of each lid
    U: xr.DataArray = xr.DataArray(
        rng.uniform(low=0, high=1, size=(n_data, n_cell)),
        dims=("Data", "Cell"),
        coords={"Data": np.arange(n_data), "Cell": np.arange(n_cell)},
    )

    # Mixture proportions normalised across cells
    M: xr.DataArray = (U * P) / (U * P).sum(dim="Cell")

    # Dirichlet-distributed allocation across funnel × cell
    X_: xr.DataArray = xr.DataArray(
        dirichlet(
            alpha=1 + M.transpose("Data", "Funnel", "Cell") / r,
            rng=rng,
            size=(n_data, n_funnel, n_cell),
        ),
        dims=("Data", "Funnel", "Cell"),
        coords={
            "Data": np.arange(n_data),
            "Funnel": np.arange(n_funnel),
            "Cell": np.arange(n_cell),
        },
    )

    # Aggregate over funnels
    X: xr.DataArray = (X_ * W).sum(dim="Funnel")

    return U, X


# Identify array values closest to specified percentiles
def find_closest_percentiles(array, percentiles):
    """
    Finds values within an array closest to specified percentiles.

    Parameters:
        array (np.ndarray): Sorted numerical array.
        percentiles (List[float]): Desired percentiles (0-100 scale).

    Returns:
        tuple[np.ndarray, np.ndarray]: Closest values and their respective indices.
    """
    array = np.sort(array)
    min_val, max_val = array[0], array[-1]
    targets = min_val + (max_val - min_val) * np.array(percentiles) / 100
    indices = np.searchsorted(array, targets, side="left")
    indices[indices == len(array)] = len(array) - 1

    for i in range(len(indices)):
        if indices[i] > 0 and np.abs(array[indices[i] - 1] - targets[i]) < np.abs(
            array[indices[i]] - targets[i]
        ):
            indices[i] -= 1

    closest_values = array[indices]
    return closest_values, indices


# --- Main script starts here ---

# Simulation constants
c_param: float = 0.1
n_data: int = 2000
n_cell: int = 5
n_funnel: int = 100

# Generate baseline data using very low r (quasi-deterministic)
rng: np.random.Generator = np.random.default_rng(42)
_, X_0 = generate_data(
    r=1e-9, c=c_param, n_data=n_data, n_cell=n_cell, n_funnel=n_funnel, rng=rng
)

# Scan over r values to observe impact on data
r_list = np.logspace(-6, 0, 1001, endpoint=True)
err_list = []
for r in tqdm(r_list):
    rng = np.random.default_rng(42)
    U, X = generate_data(
        r=r, c=c_param, n_data=n_data, n_cell=n_cell, n_funnel=n_funnel, rng=rng
    )
    err_list.append(np.mean(abs(X - X_0)))  # Mean absolute error from baseline

# Select representative r values based on percentile distance from baseline
percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
values, indices = find_closest_percentiles(err_list, percentiles)
r_dataset = r_list[indices]

# Repeat the above for parameter c (affecting clustering)
c_list = np.logspace(-8, 0, 1001, endpoint=True)
w_0 = np.sort(stick_breaking(alpha=1 / 1e-10, rng=rng, size=n_funnel))[::-1]
err_list = []
for c in tqdm(c_list):
    rng = np.random.default_rng(42)
    w = np.sort(stick_breaking(alpha=1 / c, rng=rng, size=n_funnel))[::-1]
    err_list.append(np.mean(abs(w - w_0)))

percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
values, indices = find_closest_percentiles(err_list, percentiles)
c_dataset = c_list[indices]

# Generate final dataset for all r × c combinations
dataset = {f"r_{i}": {} for i in range(10)}
for r_idx, r in tqdm(enumerate(r_dataset)):
    for c_idx, c in enumerate(c_dataset):
        dataset[f"r_{r_idx}"][f"c_{c_idx}"] = {}
        rng = np.random.default_rng(42)
        U, X = generate_data(
            r=r, c=c, n_data=n_data, n_cell=n_cell, n_funnel=n_funnel, rng=rng
        )
        dataset[f"r_{r_idx}"][f"c_{c_idx}"]["U"] = U.copy()
        dataset[f"r_{r_idx}"][f"c_{c_idx}"]["X"] = X.copy()

# Persist the full dataset to disk
with open("Galton_Dataset.pickle", "wb") as file:
    pickle.dump(dataset, file)
