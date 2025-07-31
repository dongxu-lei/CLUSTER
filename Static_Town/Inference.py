# Import necessary libraries
import numpy as np  # For numerical computations and data manipulation
import pymc as pm  # For probabilistic programming and Bayesian inference
from pytensor import (
    tensor as pt,
)  # Symbolic tensor manipulation, used by PyMC internally
import os
import sys

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

# Change the working directory to the script's directory
os.chdir(script_dir)

# # Snakemake Test
# if __name__ == "__main__":
#     U = np.load(f"U_day.npy")
#     X = np.load(f"X_day.npy")
#     with open("day.nc", "wb") as file:
#         pass

# exit(0)


# Specify the time setting of the data; must be either "day" or "night"
time = "day"

# Load the preprocessed data arrays from storage
# U represents the duty cycles of base stations (binary matrix)
# X denotes the base station usage counts (observed data)

# Both U and X are NumPy ndarray objects, with the shape (n_data, n_BS)
U = np.load(f"U_{time}.npy")
X = np.load(f"X_{time}.npy")

# Define the number of data observations and modelled latent clusters
n_data = 10000
n_cluster, n_BS = (
    10,
    13,
)  # The model considers up to 10 latent user clusters and 13 base stations

# Broadcast U to match the dimensions required for cluster-wise modelling
# Shape: (n_data, n_cluster, n_BS)
U_broad = np.broadcast_to(U[:, None, :], (n_data, n_cluster, n_BS))

# Define named coordinate dimensions for use in the PyMC model
# These provide dimension labels for readability and diagnostic clarity
coords = {
    "cluster": np.arange(n_cluster),
    "obs_id": np.arange(n_data),
    "BS": np.arange(n_BS),
}

# Begin the context for defining the probabilistic model
with pm.Model(coords=coords) as model:

    # Define a Gamma prior for the concentration parameter (alpha) of the stick-breaking process
    alpha = pm.Gamma("alpha", alpha=10, beta=1)

    # Define stick-breaking weights for the latent clusters using the Dirichlet Process construction
    # This models the prior probability of cluster membership
    w = pm.StickBreakingWeights("w", alpha=alpha, K=n_cluster - 1)

    # Define the Dirichlet-distributed latent preference vectors (L) over base stations for each cluster
    # Each cluster has its own distribution over the 13 base stations
    L = pm.Dirichlet("L", a=np.ones((n_cluster, n_BS)), dims=["cluster", "BS"])

    # Introduce the broadcasted binary availability mask as mutable data for dynamic substitution if needed
    U = pm.MutableData("U", U_broad)

    # Mask the latent cluster preference vectors using the availability matrix
    L_hat = L * U

    # Renormalise the masked preferences so they sum to one across base stations
    L_hat /= L_hat.sum(axis=-1, keepdims=True)

    # Broadcast the stick-breaking weights across all observations
    w_broad = pt.extra_ops.broadcast_to(w, (n_data, n_cluster))

    # Compute the expected base station proportions by marginalising over the cluster assignments
    # This effectively performs a weighted mixture of cluster-specific preferences
    p = pt.batched_tensordot(L_hat, w_broad, axes=[[1], [1]])

    # Ensure the probabilities are properly normalised
    p /= p.sum(axis=1, keepdims=True)

    # Define a hyperparameter for the concentration of the Dirichlet likelihood (effectively a precision)
    c = pm.Gamma("c", alpha=1000, beta=1)

    # Specify the likelihood of the observed data
    obs = pm.Dirichlet(
        "obs", a=p * c, dims=["obs_id", "BS"], observed=X / X.sum(axis=1, keepdims=True)
    )

# Sample from the posterior distribution using the No-U-Turn Sampler (NUTS)
# Draws include 2000 samples per chain after 1000 tuning steps
# ADVI initialisation is used for automatic variational inference to aid convergence
with model:
    trace = pm.sample(draws=2000, tune=1000, init="advi", target_accept=0.99, chains=4)

# Persist the sampling results to a NetCDF file for later analysis
trace.to_netcdf(f"./{time}.nc")
