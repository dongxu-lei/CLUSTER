"""
Author: Dongxu Lei
Date: 22/03/2025
Description:
This script performs posterior inference based on the Galton board-inspired dataset.
"""

# --- Library Imports ---

import numpy as np  # Numerical computations and array operations
from tqdm import tqdm  # Progress bar utility
import arviz as az  # Diagnostics and visualisation of Bayesian inference
import pymc as pm  # Probabilistic programming framework for Bayesian models
from pytensor import tensor as pt  # Symbolic tensor operations (backend for PyMC)
import pandas as pd  # Data manipulation and I/O
import pickle  # Object serialisation (used here for dataset loading)
import os
import sys

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

# Change the working directory to the script's directory
os.chdir(script_dir)

# # Snakemake Test
# if __name__ == "__main__":
#     with open("Galton_Dataset.pickle", "rb") as file:
#         pass

#     for r_idx in range(1, 11):
#         for c_idx in range(1, 11):
#             filename = f"./r_{r_idx}_c_{c_idx}.nc"
#             with open(filename, 'w') as file:
#                 pass

# exit(0)

# --- Bayesian Model Definition ---


def CLUSTER(
    n_funnel: int, n_cell: int, U_data: np.ndarray, X_data: np.ndarray
) -> pm.Model:
    """
    Defines a hierarchical Bayesian model to perform posterior inference
    over funnel-based latent structures within the Galton board dataset.

    Parameters:
        n_funnel (int): Number of latent funnel components.
        n_cell (int): Number of output categories (cells).
        U_data (np.ndarray): Openness values for each cell per observation.
        X_data (np.ndarray): Observed data samples (cell probabilities).

    Returns:
        pm.Model: A fully specified PyMC model.
    """
    data_length: int = len(U_data)

    # Broadcast U_data to shape (Data, Funnel, Cell)
    U_broad = np.broadcast_to(
        U_data.values[:, None, :], (data_length, n_funnel, n_cell)
    )

    # Coordinate labels for use in the model
    coords = {
        "Funnel": np.arange(n_funnel),
        "Data": np.arange(data_length),
        "Cell": np.arange(n_cell),
    }

    # Context manager defines the PyMC model
    with pm.Model(coords=coords) as model:

        # Hyperprior on concentration parameter for stick-breaking
        alpha = pm.Gamma("alpha", alpha=1, beta=1)

        # Stick-breaking weights for the mixture components
        w = pm.StickBreakingWeights("w", alpha=alpha, K=n_funnel - 1)

        # Per-funnel categorical distributions over cells
        L = pm.Dirichlet("L", a=np.ones((n_funnel, n_cell)), dims=["Funnel", "Cell"])

        # Register observed U values as mutable input data
        U = pm.MutableData("U", U_broad)

        # Modulate L by U to produce funnel-weighted likelihoods
        L_hat = L * U
        L_hat /= L_hat.sum(axis=-1, keepdims=True)  # Normalise across cells

        # Broadcast stick-breaking weights across data
        w_broad = pt.extra_ops.broadcast_to(w, (data_length, n_funnel))

        # Compute mixture over funnels (batch-wise tensor dot product)
        p = pt.batched_tensordot(L_hat, w_broad, axes=[[1], [1]])
        p /= p.sum(axis=1, keepdims=True)  # Ensure probabilities sum to 1

        # Concentration parameter for final Dirichlet observation model
        c = pm.Gamma("c", alpha=1, beta=1)

        # Observed data modelled via Dirichlet with mean p and concentration c
        obs = pm.Dirichlet(
            "obs",
            a=p * c,
            dims=["Data", "Cell"],
            observed=X_data.values,
        )

    return model


# --- Sampling Procedure ---


def sample(U_data: np.ndarray, X_data: np.ndarray, n_funnel: int) -> None:
    """
    Performs posterior sampling on the Galton board model.

    Parameters:
        U_data (np.ndarray): Lid openness values.
        X_data (np.ndarray): Observed data vectors.
        n_funnel (int): Number of latent funnel components.

    Returns:
        trace: Posterior samples for the model parameters.
    """
    with CLUSTER(
        n_funnel=n_funnel, n_cell=U_data.shape[1], U_data=U_data, X_data=X_data
    ):

        # Perform MCMC sampling using NUTS with ADVI initialisation
        trace = pm.sample(
            draws=1000,  # Number of posterior samples
            tune=1000,  # Number of tuning (warm-up) steps
            init="advi",  # Use ADVI to initialise
            target_accept=0.99,  # High target acceptance to ensure robust convergence
            nuts_sampler="nutpie",  # Use Nutpie
            chains=4,  # Number of parallel chains
        )

    return trace


# --- Main Execution Routine ---


def main():
    """
    Orchestrates the posterior inference over a grid of r and c parameter combinations.
    Loads data, samples posterior distributions, and saves results to NetCDF format.
    """
    # Load pre-generated synthetic data
    with open("Galton_Dataset.pickle", "rb") as file:
        dataset = pd.read_pickle(file)
        # The dataset is a dict of dict objects. The first level of indexing is for the r parameter, which ranges from r_0 to r_9. The second level is for the c parameter, which ranges from c_0 to c_9. The third level is for 'U' or 'X', which stands for the BS duty cycles and the observed loads. The dataset for a specific parameter setting is a NumPy ndarray object with the shape (n_data, n_BS).

    # Iterate over all parameter combinations
    for r_idx in range(10):
        for c_idx in range(10):

            print(f"-----Sampling for r_{r_idx}_c_{c_idx}-----")

            U = dataset[f"r_{r_idx}"][f"c_{c_idx}"]["U"]
            X = dataset[f"r_{r_idx}"][f"c_{c_idx}"]["X"]

            # Conduct sampling only on the first 1500 observations
            trace = sample(U_data=U[:1500], X_data=X[:1500], n_funnel=10)

            # Save posterior trace to file (NetCDF format)
            trace.to_netcdf(f"./r_{r_idx}_c_{c_idx}.nc")


# Execute script if run as main programme
if __name__ == "__main__":
    main()
