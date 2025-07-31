# Importing necessary libraries for numerical operations, probabilistic programming, tensor manipulation,
# typing annotations, object serialisation, and data splitting.
import numpy as np
import pymc as pm
from pytensor import tensor as pt
from typing import Any
import pickle
from sklearn.model_selection import train_test_split

import os
import sys

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

# Change the working directory to the script's directory
os.chdir(script_dir)

# # Snakemake Test
# if __name__ == "__main__":
#     with open("Cixi.pkl", "rb") as file:
#         pass
#     if not os.path.exists("Cixi_MBNP"):
#         os.makedirs("Cixi_MBNP")
#     with open("./Cixi_MBNP/trace.nc", "wb") as file:
#         pass

# exit(0)

def load_data() -> tuple[np.ndarray, ...]:
    """
    Load the dataset from a serialised pickle file and perform basic preprocessing.

    Returns:
        A tuple of NumPy arrays containing the training and testing sets of
        duty cycle data (U) and base station load data (X).
    """

    # Load the dictionary data from the pickle file
    with open("Cixi.pkl", "rb") as file:
        data = pickle.load(file)

    # Extract the duty cycle ('U') and BS load ('X') arrays from the dictionary.
    # Both are expected to be of shape (n_data, n_BS).
    U = data["U"]
    X = data["X"]

    # Replace zero values in U with a small positive constant for numerical stability
    U[U == 0] = 1e-6
    # Add a small positive constant to all elements in X to avoid zero entries
    X += 1e-6

    # Split the dataset into training and testing subsets
    U_train, U_test, X_train, X_test = train_test_split(
        U, X, test_size=0.1, random_state=42
    )

    return U_train, U_test, X_train, X_test


def M_BNP(n_truncate_cluster: int, U_data: np.ndarray, X_data: np.ndarray) -> pm.Model:
    """
    Construct a Bayesian Nonparametric (BNP) model using a stick-breaking Dirichlet process.

    Args:
        n_truncate_cluster: The truncation level for the number of clusters.
        U_data: Duty cycle data.
        X_data: Base station load data.

    Returns:
        A PyMC probabilistic model object.
    """

    data_length: int = len(U_data)
    n_BS: int = U_data.shape[1]

    # Broadcast U_data to align with the shape expected in clustering calculations
    U_broad = np.broadcast_to(
        U_data[:, None, :], (data_length, n_truncate_cluster, n_BS)
    )

    # Define model coordinates for interpretability
    coords = {
        "Cluster": np.arange(n_truncate_cluster),
        "Obs": np.arange(data_length),
        "BS": np.arange(n_BS),
    }

    with pm.Model(coords=coords) as model:
        # Concentration parameter of the stick-breaking process
        alpha = pm.Gamma("alpha", 1, 1)

        # Stick-breaking weights defining the cluster proportions
        w = pm.StickBreakingWeights("w", alpha=alpha, K=n_truncate_cluster - 1)

        # Cluster-specific Dirichlet-distributed latent variables for BS load allocation
        L = pm.Dirichlet(
            "L", np.ones((n_truncate_cluster, n_BS)), dims=["Cluster", "BS"]
        )

        # Declare mutable input data for the model
        U = pm.MutableData("U", U_broad)

        # Compute the element-wise product of L and U and normalise across the BS axis
        L_hat = L * U
        L_hat /= L_hat.sum(axis=-1, keepdims=True)

        # Broadcast cluster weights to match dimensions
        w_broad = pt.extra_ops.broadcast_to(w, (data_length, n_truncate_cluster))

        # Calculate the weighted sum over clusters to obtain the predictive distribution p
        p = pt.batched_tensordot(L_hat, w_broad, axes=[[1], [1]])
        p /= p.sum(axis=1, keepdims=True)  # Normalise the probability vector

        # Concentration parameter for the Dirichlet observation model
        c = pm.Gamma("c", 1000, 1)

        # Observation model: Dirichlet-distributed random variables with parameters scaled by p and c
        obs = pm.Dirichlet(
            "obs",
            p * c,
            dims=["Obs", "BS"],
            observed=X_data / X_data.sum(axis=1, keepdims=True),  # Normalised observations
        )

    return model


def sample(U_train: np.ndarray, X_train: np.ndarray, n_truncate_cluster: int):
    """
    Perform posterior sampling using the PyMC model.

    Args:
        U_train: Training data for duty cycles.
        X_train: Training data for BS loads.
        n_truncate_cluster: Truncation level for BNP model.

    Returns:
        Trace object containing posterior samples.
    """

    # Instantiate and sample from the model
    with M_BNP(
        n_truncate_cluster=n_truncate_cluster,
        U_data=U_train,
        X_data=X_train,
    ):
        trace = pm.sample(
            draws=5000,        # Total number of samples to draw
            tune=2000,         # Number of tuning steps
            init="advi",       # Initialisation method using variational inference
            target_accept=0.99,# Target acceptance probability for the NUTS sampler
            nuts_sampler="nutpie",  # Use 'nutpie' backend for NUTS
            chains=4           # Number of MCMC chains
        )

    return trace


def main() -> None:
    """
    Main execution function: loads data, fits the BNP model, and stores posterior samples.
    """

    n_truncate_cluster = 20  # Define truncation level

    # Load preprocessed training and testing data
    U_train, U_test, X_train, X_test = load_data()
    
    # Create output directory if it doesn't exist
    if not os.path.exists("Cixi_MBNP"):
        os.makedirs("Cixi_MBNP")
        
    # Sample from the model using training data
    trace = sample(U_train, X_train, n_truncate_cluster=n_truncate_cluster)

    # Save posterior samples to disk in NetCDF format
    trace.to_netcdf(f"./Cixi_MBNP/trace.nc")


# Entry point of the script
if __name__ == "__main__":
    main()
