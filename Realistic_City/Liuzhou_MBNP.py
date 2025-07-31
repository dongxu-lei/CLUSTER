# Importing necessary libraries
import numpy as np  # Numerical operations
import pymc as pm  # Probabilistic programming and Bayesian modelling
from pytensor import tensor as pt  # Tensor operations using PyTensor (formerly Aesara)
from typing import Any  # For type annotations
import pickle  # Serialisation and deserialisation of Python objects
import os  # Interaction with the operating system
import sys
from sklearn.model_selection import train_test_split  # For data splitting



# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

# Change the working directory to the script's directory
os.chdir(script_dir)

# # Snakemake Test
# if __name__ == "__main__":
#     with open("Liuzhou.pkl", "rb") as file:
#         pass
    
#     # Ensure output directory exists
#     if not os.path.exists("Liuzhou_MBNP"):
#         os.makedirs("Liuzhou_MBNP")

#     with open("./Liuzhou_MBNP/Countryside_0.nc", "wb") as file:
#         pass
    
# exit(0)

def load_data(region: str, time: str) -> tuple[np.ndarray, ...]:
    """
    Loads the environmental data for a specified region and time.

    Parameters:
        region (str): The name of the region under analysis.
        time (str): The temporal slice of the data.

    Returns:
        tuple: Split data arrays (U_train, U_test, X_train, X_test).
    """

    # Load the pickled dataset
    with open("Liuzhou.pkl", "rb") as file:
        data = pickle.load(file)

    # Extract U and X matrices from the nested dictionary
    # The dataset is a dict of dict objects. The first level of indexing specifies the region of the city of Liuzhou. Allowable entries include ("Countryside", "Outskirts", "Downtown", "Residential", "Industrial"). The second level specifies the time period of a day. Valid entries can be from '0' to '7'. The third level specifies whether we are to extract the duty cycle ("U") or BS loads ("X").
    U = data[region][time]["U"]
    X = data[region][time]["X"]

    # Replace zero values in U with a small constant to ensure numerical stability
    U[U == 0] = 1e-6
    X += 1e-6  # Avoid zero entries in X for stability in probabilistic modelling

    # Partition the data into training and testing sets
    U_train, U_test, X_train, X_test = train_test_split(
        U, X, test_size=0.1, random_state=42
    )

    return U_train, U_test, X_train, X_test


def M_BNP(n_truncate_cluster: int, U_data: np.ndarray, X_data: np.ndarray) -> pm.Model:
    """
    Constructs a M_BNP using a truncated stick-breaking process.

    Parameters:
        n_truncate_cluster (int): Truncation level for the Dirichlet Process (i.e., number of clusters).
        U_data (np.ndarray): BS duty cycle.
        X_data (np.ndarray): BS load.

    Returns:
        pm.Model: A compiled PyMC probabilistic model.
    """

    data_length: int = len(U_data)  # Number of observations
    n_BS: int = U_data.shape[1]  # Number of BSs

    # Broadcast U_data to match the dimensionality required for modelling clusters
    U_broad = np.broadcast_to(
        U_data[:, None, :], (data_length, n_truncate_cluster, n_BS)
    )

    # Define model coordinates for indexing and clarity
    coords = {
        "Cluster": np.arange(n_truncate_cluster),
        "Obs": np.arange(data_length),
        "BS": np.arange(n_BS),
    }

    # Context manager for PyMC model construction
    with pm.Model(coords=coords) as model:
        # Hyperparameter for stick-breaking process
        alpha = pm.Gamma("alpha", 1, 1)

        # Stick-breaking weights for the Dirichlet Process
        w = pm.StickBreakingWeights("w", alpha=alpha, K=n_truncate_cluster - 1)

        # Cluster-specific CPs
        L = pm.Dirichlet(
            "L", np.ones((n_truncate_cluster, n_BS)), dims=["Cluster", "BS"]
        )

        # Register U_data as mutable data for potential updating
        U = pm.MutableData("U", U_broad)

        # Element-wise multiplication of L and U to compute L_hat
        L_hat = L * U
        L_hat /= L_hat.sum(axis=-1, keepdims=True)  # Normalisation

        # Broadcast cluster weights and compute the weighted mixture
        w_broad = pt.extra_ops.broadcast_to(w, (data_length, n_truncate_cluster))
        p = pt.batched_tensordot(L_hat, w_broad, axes=[[1], [1]])
        p /= p.sum(axis=1, keepdims=True)  # Normalise the resulting probabilities

        # Precision parameter for the Dirichlet likelihood
        c = pm.Gamma("c", 1000, 1)

        # Observed Dirichlet-distributed outcomes
        obs = pm.Dirichlet(
            "obs",
            p * c,
            dims=["Obs", "BS"],
            observed=X_data / X_data.sum(axis=1, keepdims=True),
        )

    return model


def sample(U_train: np.ndarray, X_train: np.ndarray, n_truncate_cluster: int):
    """
    Samples from the posterior distribution of the Bayesian Nonparametric model.

    Parameters:
        U_train (np.ndarray): Training data matrix for BS duty cycles.
        X_train (np.ndarray): Training data matrix for observed Bs loads.
        n_truncate_cluster (int): Number of clusters for the stick-breaking approximation.

    Returns:
        trace: A PyMC trace object containing posterior samples.
    """

    with M_BNP(
        n_truncate_cluster=n_truncate_cluster,
        U_data=U_train,
        X_data=X_train,
    ):

        # Perform posterior sampling using NUTS with ADVI initialisation
        trace = pm.sample(
            draws=5000,
            tune=2000,
            init="advi",
            target_accept=0.99,
            nuts_sampler="nutpie",
            chains=4,
        )

    return trace


def main() -> None:
    """
    Main execution function for data loading, model training, and saving the posterior trace.
    """

    # Specify the region and time window for data extraction
    region = "Countryside"  # Options include: "Countryside", "Outskirts", "Downtown", "Residential", "Industrial"
    time = "0"  # Valid entries range from "0" to "7"
    n_truncate_cluster = (
        20  # Predefined number of clusters for the truncated stick-breaking process
    )

    # Load and preprocess the data
    U_train, U_test, X_train, X_test = load_data(region=region, time=time)

    # Ensure output directory exists
    if not os.path.exists("Liuzhou_MBNP"):
        os.makedirs("Liuzhou_MBNP")

    # Fit the model and sample from the posterior
    trace = sample(U_train, X_train, n_truncate_cluster=n_truncate_cluster)

    # Persist the posterior trace to NetCDF format
    trace.to_netcdf(f"./Liuzhou_MBNP/{region}_{time}.nc")


# Execute the script if run as the main module
if __name__ == "__main__":
    main()
