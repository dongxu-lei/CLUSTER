import numpy as np 
import pymc as pm
from pytensor import tensor as pt
from typing import Any
import pickle
import os  # Required for directory management
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
    
#     if not os.path.exists("M_BNP"):
#         os.makedirs("M_BNP")
#     with open("./M_BNP/10.nc", "wb") as file:
#         pass

# exit(0)

def load_data(time: int) -> tuple[np.ndarray, ...]:
    """
    Loads preprocessed data from a pickle file and partitions it into training and testing sets.

    Parameters:
    - time (int): The starting index representing a 15-minute interval of the day.

    Returns:
    - A tuple comprising training and testing datasets for both U and X variables.
    """

    with open("Dynamic_Town.pickle", "rb") as file:
        data = pickle.load(file)

    # Extracts data starting from the specified time index up to the end of the day (24*4 = 96 intervals).
    # The size of the data is of the shape (n_data, n_BS)
    U = data["U"][time : :24 * 4 ]
    X = data["X"][time : :24 * 4 ]

    # Ensures numerical stability by avoiding zero values.
    U[U == 0] = 1e-6
    X += 1e-6

    # Partitioning data into training and testing sets (90% training, 10% testing).
    U_train, U_test, X_train, X_test = train_test_split(
        U, X, test_size=0.1, random_state=42
    )

    return U_train, U_test, X_train, X_test


def M_BNP(n_truncate_cluster: int, U_data: np.ndarray, X_data: np.ndarray) -> pm.Model:
    """
    Constructs M_BNP using Dirichlet Processes and Stick-Breaking construction.

    Parameters:
    - n_truncate_cluster (int): The number of clusters for truncated approximation of the Dirichlet Process.
    - U_data (np.ndarray): Input BS duty cycles.
    - X_data (np.ndarray): Observed BS loads.

    Returns:
    - A PyMC probabilistic model instance.
    """

    data_length: int = len(U_data)
    n_BS: int = U_data.shape[1]

    # Broadcasts U_data to match model dimensions (Obs, Cluster, BS).
    U_broad = np.broadcast_to(
        U_data[:, None, :], (data_length, n_truncate_cluster, n_BS)
    )

    # Specifies coordinate dimensions for interpretability in model traces.
    coords = {
        "Cluster": np.arange(n_truncate_cluster),
        "Obs": np.arange(data_length),
        "BS": np.arange(n_BS),
    }

    with pm.Model(coords=coords) as model:
        # Concentration parameter for the Dirichlet Process.
        alpha = pm.Gamma("alpha", 1, 1)

        # Stick-breaking weights for the truncated Dirichlet Process.
        w = pm.StickBreakingWeights("w", alpha=alpha, K=n_truncate_cluster - 1)

        # Latent Dirichlet-distributed CPs for each cluster.
        L = pm.Dirichlet(
            "L", np.ones((n_truncate_cluster, n_BS)), dims=["Cluster", "BS"]
        )

        # Register broadcasted U data for use in the model.
        U = pm.MutableData("U", U_broad)

        # Element-wise multiplication and normalisation.
        L_hat = L * U
        L_hat /= L_hat.sum(axis=-1, keepdims=True)

        # Broadcast cluster weights across observations.
        w_broad = pt.extra_ops.broadcast_to(w, (data_length, n_truncate_cluster))

        # Weighted sum of CPs to derive final mixture probabilities.
        p = pt.batched_tensordot(L_hat, w_broad, axes=[[1], [1]])
        p /= p.sum(axis=1, keepdims=True)

        # Precision parameter for Dirichlet observations.
        c = pm.Gamma("c", 1000, 1)

        # Observed data modelled via Dirichlet likelihood, scaled by concentration parameter.
        obs = pm.Dirichlet(
            "obs",
            p * c,
            dims=["Obs", "BS"],
            observed=X_data / X_data.sum(axis=1, keepdims=True),
        )

    return model


def sample(U_train: np.ndarray, X_train: np.ndarray, n_truncate_cluster: int):
    """
    Executes Bayesian sampling on the defined model using the No-U-Turn Sampler (NUTS).

    Parameters:
    - U_train (np.ndarray): Training data for U.
    - X_train (np.ndarray): Training data for X.
    - n_truncate_cluster (int): Number of clusters in the truncated Dirichlet Process.

    Returns:
    - Posterior samples (trace) from the fitted model.
    """

    with M_BNP(
        n_truncate_cluster=n_truncate_cluster,
        U_data=U_train,
        X_data=X_train,
    ):

        trace = pm.sample(
            draws=5000,          # Number of posterior samples.
            tune=2000,           # Number of tuning steps.
            init="advi",         # Initialisation strategy using Automatic Differentiation Variational Inference.
            target_accept=0.99,  # High acceptance rate to ensure stable sampling.
            nuts_sampler="nutpie",  # Fast NUTS sampler using NumPy back-end.
            chains=4,            # Number of MCMC chains to run in parallel.
        )

    return trace


def main() -> None:
    """
    Main function for orchestrating data preparation, model training, and saving results.
    """

    # The 'time' variable refers to the index of the 15-minute interval (0 to 95).
    time = 10
    n_truncate_cluster = 20

    # Data loading and preprocessing.
    U_train, U_test, X_train, X_test = load_data(time=time)

    # Ensure that the directory exists for saving model results.
    if not os.path.exists("M_BNP"):
        os.makedirs("M_BNP")

    # Perform sampling and save posterior trace to NetCDF format.
    trace = sample(U_train, X_train, n_truncate_cluster=n_truncate_cluster)
    trace.to_netcdf(f"./M_BNP/{time}.nc")


# Entry point for script execution.
if __name__ == "__main__":
    main()
