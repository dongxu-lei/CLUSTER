# Import necessary libraries
import numpy as np  # Numerical operations on arrays
import pymc as pm  # Probabilistic programming for Bayesian modelling
from pytensor import tensor as pt  # Tensor operations for symbolic computation
import pandas as pd  # Data manipulation and analysis
from sklearn.model_selection import (
    train_test_split,
)  # Utility for splitting datasets into training and testing subsets
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
#     # Create output directory if it doesn't exist
#     if not os.path.exists("Liuzhou_TCM"):
#         os.makedirs("Liuzhou_TCM")
#     with open("./Liuzhou_TCM/Countryside_trace.nc", "wb") as file:
#         pass
# exit(0)


# Define function to load and preprocess data for a specified region
def load_data(region):
    # Initialise lists to store training and testing data
    U_data_list, X_data_list, U_test_list, X_test_list = [], [], [], []

    # Load pickled data file containing measurements for different regions and time intervals
    # Extract sub-dictionaries using hierarchical keys: region → time → measurement type
    with open("Liuzhou.pkl", "rb") as file:
        data = pd.read_pickle(file)

    # Iterate over eight discrete time intervals
    for time in range(8):
        U = data[region][f"{time}"][
            "U"
        ]  # Retrieve duty cycle (U) for a specific region and time
        X = data[region][f"{time}"]["X"]  # Retrieve BS load data (X) for the same

        U[U == 0] = 1e-6  # Replace zero values in U to avoid division or log errors
        X += 1e-6  # Add small value to X for numerical stability

        # Split both U and X into training (90%) and testing (10%) sets
        U_data, U_test, X_data, X_test = train_test_split(
            U, X, test_size=0.1, random_state=42
        )

        # Store truncated and full training/testing datasets
        U_data_list.append(U_data.copy()[:170])
        X_data_list.append(X_data.copy()[:170])
        U_test_list.append(U_test.copy())
        X_test_list.append(X_test.copy())

    # Return lists containing processed data for all time intervals
    return U_data_list, X_data_list, U_test_list, X_test_list


def M_BNP(n_truncated_cluster, U_data_list, X_data_list):
    conc = 1e5  # Concentration parameter for the Dirichlet distribution
    data_length, n_BS = U_data_list[0].shape  # Determine data dimensions

    # Define model coordinate dimensions
    coords = {
        "Cluster": np.arange(n_truncated_cluster),
        "Obs": np.arange(data_length),
        "BS": np.arange(n_BS),
    }

    w = {}
    w_broad = {}
    L = {}
    L_hat = {}
    U = {}
    U_broad = {}
    p = {}

    # Construct a hierarchical Bayesian model using PyMC
    with pm.Model(coords=coords) as model:
        # Hyperparameters and prior distributions
        s = pm.Beta("s", alpha=1, beta=1)  # Mixture weight between Dirichlet components
        c = pm.Gamma(
            "c", 1000, 1
        )  # Scaling factor for observed Dirichlet distributions
        alpha = pm.Gamma(
            "alpha", alpha=1, beta=1
        )  # Concentration parameter for stick-breaking weights

        # Define initial stick-breaking weights and base distribution over clusters
        w[0] = pm.StickBreakingWeights("w_0", alpha=alpha, K=n_truncated_cluster - 1)
        L[0] = pm.Dirichlet(
            "L_0", np.ones((n_truncated_cluster, n_BS)), dims=["Cluster", "BS"]
        )

        # Broadcast usage data for compatibility with cluster dimensions
        U_broad[0] = np.broadcast_to(
            U_data_list[0][:, None, :], (data_length, n_truncated_cluster, n_BS)
        )
        U[0] = pm.MutableData(
            "U_0", U_broad[0]
        )  # Define mutable input data for PyMC model

        L_hat[0] = L[0] * U[0]
        L_hat[0] /= L_hat[0].sum(axis=-1, keepdims=True)

        w_broad[0] = pt.extra_ops.broadcast_to(w[0], (data_length, n_truncated_cluster))
        p[0] = pt.batched_tensordot(L_hat[0], w_broad[0], axes=[[1], [1]])
        p[0] /= p[0].sum(axis=1, keepdims=True)

        # Observed data modelled using a Dirichlet distribution
        pm.Dirichlet(
            "obs_0",
            p[0] * c,
            dims=["Obs", "BS"],
            observed=X_data_list[0] / X_data_list[0].sum(axis=1, keepdims=True),
        )

        # Iteratively define model structure for subsequent time intervals (1 to 7)
        for i in range(1, len(U_data_list)):
            w[i] = pm.StickBreakingWeights(
                f"w_{i}", alpha=alpha, K=n_truncated_cluster - 1
            )
            # Introduces temporal correlation to CPs
            L[i] = pm.Mixture(
                f"L_{i}",
                w=[s, 1 - s],
                comp_dists=[
                    pm.Dirichlet.dist(a=L[i - 1] * conc),
                    pm.Dirichlet.dist(a=np.ones((n_truncated_cluster, n_BS))),
                ],
            )
            U_broad[i] = np.broadcast_to(
                U_data_list[i][:, None, :], (data_length, n_truncated_cluster, n_BS)
            )
            U[i] = pm.MutableData(f"U_{i}", U_broad[i])
            L_hat[i] = L[i] * U[i]
            L_hat[i] /= L_hat[i].sum(axis=-1, keepdims=True)
            w_broad[i] = pt.extra_ops.broadcast_to(
                w[i], (data_length, n_truncated_cluster)
            )
            p[i] = pt.batched_tensordot(L_hat[i], w_broad[i], axes=[[1], [1]])
            p[i] /= p[i].sum(axis=1, keepdims=True)
            pm.Dirichlet(
                f"obs_{i}",
                p[i] * c,
                dims=["Obs", "BS"],
                observed=X_data_list[i] / X_data_list[i].sum(axis=1, keepdims=True),
            )
    return model


# Main script execution block
if __name__ == "__main__":
    # Specify the region of interest
    region = "Countryside"  # Options include: "Countryside", "Outskirts", "Downtown", "Residential", "Industrial"

    U_data_list, X_data_list, _, _ = load_data(
        region
    )  # Load data for the specified region

    n_truncated_cluster = (
        20  # Define number of clusters for truncated stick-breaking process
    )

    model = M_BNP(n_truncated_cluster, U_data_list, X_data_list)

    # Perform posterior sampling using NUTS algorithm with ADVI initialisation
    with model:
        trace = pm.sample(
            draws=10000,
            tune=1000,
            init="advi",
            target_accept=0.99,
            nuts_sampler="blackjax",
            chains=8,
        )

    # Create output directory if it doesn't exist
    if not os.path.exists("Liuzhou_TCM"):
        os.makedirs("Liuzhou_TCM")
    # Save the posterior trace to NetCDF format for subsequent analysis
    trace.to_netcdf(f"./Liuzhou_TCM/{region}_trace.nc")
