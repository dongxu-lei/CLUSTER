# Import essential libraries for numerical computation, probabilistic modelling, tensor algebra,
# type annotation, data deserialisation, and dataset splitting.
import numpy as np
import pymc as pm
from pytensor import tensor as pt
from typing import Any
import pickle
import os
from sklearn.model_selection import train_test_split
import os
import sys

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

# Change the working directory to the script's directory
os.chdir(script_dir)

# # Snakemake Test
# if __name__ == "__main__":
#     with open("Liuzhou.pkl", "rb") as file:
#         pass
#     # Ensure output directory exists before saving results
#     if not os.path.exists("Liuzhou_MPLN"):
#         os.makedirs("Liuzhou_MPLN")
#     with open("./Liuzhou_MPLN/Countryside_0.nc", "wb") as file:
#         pass

# exit(0)


def load_data(region: str, time: str) -> tuple[np.ndarray, ...]:
    """
    Load and preprocess data for a specified region and time interval from a serialised dictionary.

    Args:
        region: A string identifier for a geographic region within Liuzhou
                (e.g., "Countryside", "Outskirts", "Downtown", "Residential", "Industrial").
        time: A string indicating a time slot in a daily schedule (values from '0' to '7').

    Returns:
        A tuple containing the training and testing sets for both the duty cycle data (U) and
        base station load data (X), all as NumPy arrays.
    """

    # Open and load the nested dictionary object from the pickle file
    with open("Liuzhou.pkl", "rb") as file:
        data = pickle.load(file)

    # Extract sub-dictionaries using hierarchical keys: region → time → measurement type
    U = data[region][time]["U"]  # Duty cycle matrix
    X = data[region][time]["X"]  # Base station load matrix

    # Ensure numerical stability by avoiding zero entries
    U[U == 0] = 1e-6
    X += 1e-6

    # Partition the dataset into training and testing subsets (10% for testing)
    U_train, U_test, X_train, X_test = train_test_split(
        U, X, test_size=0.1, random_state=42
    )

    return U_train, U_test, X_train, X_test


def M_PLN(n_preset_cluster: int, U_data: np.ndarray, X_data: np.ndarray) -> pm.Model:
    """
    Construct M_PLN with a fixed number of clusters.

    Args:
        n_preset_cluster: The predefined number of latent clusters.
        U_data: Matrix of duty cycle values (training subset).
        X_data: Matrix of base station load values (training subset).

    Returns:
        A PyMC probabilistic model instance.
    """

    data_length: int = len(U_data)  # Number of observations
    n_BS: int = U_data.shape[1]  # Number of base stations

    # Broadcast U_data for compatibility with latent cluster dimensions
    U_broad = np.broadcast_to(U_data[:, None, :], (data_length, n_preset_cluster, n_BS))

    # Define named axes for interpretability and dimensional consistency
    coords = {
        "Cluster": np.arange(n_preset_cluster),
        "Obs": np.arange(data_length),
        "BS": np.arange(n_BS),
    }

    with pm.Model(coords=coords) as model:
        # Cluster proportions drawn from a symmetric Dirichlet prior
        w = pm.Dirichlet("w", np.ones(n_preset_cluster), dims="Cluster")

        # Cluster-specific base station allocation vectors, also Dirichlet-distributed
        L = pm.Dirichlet("L", np.ones((n_preset_cluster, n_BS)), dims=["Cluster", "BS"])

        # Declare mutable input tensor representing duty cycle data
        U = pm.MutableData("U", U_broad)

        # Apply element-wise multiplication and normalise across the BS axis
        L_hat = L * U
        L_hat /= L_hat.sum(axis=-1, keepdims=True)

        # Expand weight vector to align with observation dimensions
        w_broad = pt.extra_ops.broadcast_to(w, (data_length, n_preset_cluster))

        # Calculate the convex combination of cluster-specific profiles
        p = pt.batched_tensordot(L_hat, w_broad, axes=[[1], [1]])
        p /= p.sum(
            axis=1, keepdims=True
        )  # Ensure that output probabilities are normalised

        # Concentration parameter for the Dirichlet likelihood
        c = pm.Gamma("c", 1000, 1)

        # Observation model: Dirichlet distribution conditioned on the weighted latent structure
        obs = pm.Dirichlet(
            "obs",
            p * c,
            dims=["Obs", "BS"],
            observed=X_data
            / X_data.sum(axis=1, keepdims=True),  # Normalise empirical data
        )

    return model


def sample(U_train: np.ndarray, X_train: np.ndarray, n_preset_cluster: int):
    """
    Sample from the posterior distribution of the PLN model using Markov Chain Monte Carlo (MCMC).

    Args:
        U_train: Duty cycle training matrix.
        X_train: Base station load training matrix.
        n_preset_cluster: Number of latent clusters.

    Returns:
        A PyMC trace object containing posterior samples from the model.
    """

    with M_PLN(
        n_preset_cluster=n_preset_cluster,
        U_data=U_train,
        X_data=X_train,
    ):

        trace = pm.sample(
            draws=5000,  # Number of samples to draw from the posterior
            tune=2000,  # Number of tuning iterations
            init="advi",  # Use Automatic Differentiation Variational Inference for initialisation
            target_accept=0.99,  # Target acceptance rate for NUTS sampler
            nuts_sampler="nutpie",  # Specify the NUTS backend
            chains=4,  # Use four parallel MCMC chains
        )

    return trace


def main() -> None:
    """
    Main function: loads the data, constructs the model, performs inference, and stores the results.
    """

    region = "Countryside"  # Select region of interest
    time = "0"  # Specify time interval
    n_preset_cluster = 20  # Define the number of latent clusters

    # Load preprocessed data for the specified region and time interval
    U_train, U_test, X_train, X_test = load_data(region=region, time=time)

    # Ensure output directory exists before saving results
    if not os.path.exists("Liuzhou_MPLN"):
        os.makedirs("Liuzhou_MPLN")

    # Sample from the model and persist results in NetCDF format
    trace = sample(U_train, X_train, n_preset_cluster=n_preset_cluster)
    trace.to_netcdf(f"./Liuzhou_MPLN/{region}_{time}.nc")


# Entry point for the script when executed as the main module
if __name__ == "__main__":
    main()
