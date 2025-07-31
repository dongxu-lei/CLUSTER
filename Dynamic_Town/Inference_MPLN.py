import numpy as np
import pymc as pm
from pytensor import tensor as pt
from typing import Any
import pickle
import os  # Required for directory creation
from sklearn.model_selection import train_test_split
import os
import sys

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

# Change the working directory to the script's directory
os.chdir(script_dir)

# # Snakemake Test
# if __name__ == "__main__":
#     with open("Dynamic_Town.pickle", "rb") as file:
#         pass
    
#     if not os.path.exists("M_PLN"):
#         os.makedirs("M_PLN")
#     with open("./M_PLN/10.nc", "wb") as file:
#         pass

# exit(0)

def load_data(time: int) -> tuple[np.ndarray, ...]:
    """
    Loads and prepares data from a serialised pickle file and splits it into training and testing datasets.

    Parameters:
    - time (int): The starting index indicating the k-th 15-minute interval of the day (range: 0 to 95).

    Returns:
    - A tuple of NumPy arrays containing U_train, U_test, X_train, and X_test.
    """

    with open("Dynamic_Town.pickle", "rb") as file:
        data = pickle.load(file)

    # Extracts data from the specified interval up to the end of the day.
    # The size of the data is of the shape (n_data, n_BS)
    U = data["U"][time : 24 * 4 :]
    X = data["X"][time : 24 * 4 :]

    # Mitigates numerical instability by ensuring no zero values are present.
    U[U == 0] = 1e-6
    X += 1e-6

    # Splits data into training and testing subsets (90% training, 10% testing).
    U_train, U_test, X_train, X_test = train_test_split(
        U, X, test_size=0.1, random_state=42
    )

    return U_train, U_test, X_train, X_test


def M_PLN(n_preset_cluster: int, U_data: np.ndarray, X_data: np.ndarray) -> pm.Model:
    """
    Constructs M_PLN using a fixed number of clusters.

    Parameters:
    - n_preset_cluster (int): The predefined number of clusters for the model.
    - U_data (np.ndarray): Input BS duty cycles.
    - X_data (np.ndarray): Observed BS loads.

    Returns:
    - An instance of a PyMC probabilistic model.
    """

    data_length: int = len(U_data)
    n_BS: int = U_data.shape[1]

    # Broadcasts U_data to match model dimensions (Obs, Cluster, BS).
    U_broad = np.broadcast_to(U_data[:, None, :], (data_length, n_preset_cluster, n_BS))

    # Defines coordinate labels for interpretability in the model trace.
    coords = {
        "Cluster": np.arange(n_preset_cluster),
        "Obs": np.arange(data_length),
        "BS": np.arange(n_BS),
    }

    with pm.Model(coords=coords) as model:
        # Mixture weights across clusters (drawn from a symmetric Dirichlet distribution).
        w = pm.Dirichlet("w", np.ones(n_preset_cluster), dims="Cluster")

        # Cluster-specific CPs over BS dimensions.
        L = pm.Dirichlet("L", np.ones((n_preset_cluster, n_BS)), dims=["Cluster", "BS"])

        # Registers the broadcasted latent input as mutable shared data.
        U = pm.MutableData("U", U_broad)

        # Element-wise multiplication and normalisation.
        L_hat = L * U
        L_hat /= L_hat.sum(axis=-1, keepdims=True)

        # Broadcasts cluster weights over all observations.
        w_broad = pt.extra_ops.broadcast_to(w, (data_length, n_preset_cluster))

        # Weighted sum of CPs to derive final mixture probabilities.
        p = pt.batched_tensordot(L_hat, w_broad, axes=[[1], [1]])
        p /= p.sum(axis=1, keepdims=True)

        # Precision (concentration) parameter for the Dirichlet likelihood.
        c = pm.Gamma("c", 1000, 1)

        # Observed data is modelled via a Dirichlet distribution conditioned on p and c.
        obs = pm.Dirichlet(
            "obs",
            p * c,
            dims=["Obs", "BS"],
            observed=X_data / X_data.sum(axis=1, keepdims=True),
        )

    return model


def sample(U_train: np.ndarray, X_train: np.ndarray, n_preset_cluster: int):
    """
    Performs posterior sampling from the defined parametric latent network model.

    Parameters:
    - U_train (np.ndarray): Training data for U.
    - X_train (np.ndarray): Training data for X.
    - n_preset_cluster (int): Number of clusters (fixed, not inferred) for model definition.

    Returns:
    - PyMC trace object containing posterior samples.
    """

    with M_PLN(
        n_preset_cluster=n_preset_cluster,
        U_data=U_train,
        X_data=X_train,
    ):

        trace = pm.sample(
            draws=5000,  # Total number of samples to draw from the posterior.
            tune=2000,  # Number of tuning steps for the NUTS sampler.
            init="advi",  # Initialisation via Automatic Differentiation Variational Inference.
            target_accept=0.99,  # High target acceptance probability for improved convergence.
            nuts_sampler="nutpie",  # Specifies the use of the nutpie backend.
            chains=4,  # Number of Markov chains to run concurrently.
        )

    return trace


def main() -> None:
    """
    Coordinates the loading of data, execution of the Bayesian model, and persistence of inference results.
    """

    # The 'time' variable specifies the 15-minute interval index (0 ≤ time < 96).
    time = 10
    n_preset_cluster = 20

    # Loads the dataset and partitions it into training/testing subsets.
    U_train, U_test, X_train, X_test = load_data(time=time)

    # Creates directory for saving output if it does not exist.
    if not os.path.exists("M_PLN"):
        os.makedirs("M_PLN")

    # Performs inference and saves the posterior samples to file.
    trace = sample(U_train, X_train, n_preset_cluster=n_preset_cluster)
    trace.to_netcdf(f"./M_PLN/{time}.nc")


# Script entry point.
if __name__ == "__main__":
    main()
