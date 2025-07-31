# Import necessary libraries
import numpy as np  # Provides support for numerical operations and array manipulation
from tqdm import (
    tqdm,
)  # Enables progress bars for loops, aiding in tracking execution progress
import pymc as pm  # A probabilistic programming library for Bayesian statistical modelling
from pytensor import tensor as pt  # Tensor operations used in symbolic computation
import os
import sys

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

# Change the working directory to the script's directory
os.chdir(script_dir)

# # Snowmake Test
# if __name__ == "__main__":
#     test_data = np.random.randn(10)
#     np.save(file=f"U_day", arr=test_data)
#     np.save(file=f"X_day", arr=test_data)

# exit(0)

# Define the coordinates of the base stations (BS)
loc_BS = np.array(
    (
        (139, 85),
        (269, 99),
        (140, 187),
        (599, 79),
        (700, 106),
        (722, 206),
        (485, 543),
        (710, 533),
        (722, 371),
        (225, 377),
        (374, 339),
        (471, 237),
        (526, 357),
    )
)

# Define the number of users in each of the four clusters
# These clusters may be associated with different neighbourhoods or areas
# Cluster identifiers: Blue (100), Yellow (200), Green (300), Red (400)
N_cluster = np.array((100, 200, 300, 400))

# Define the mean locations (centroids) of the clusters during nighttime
loc_cluster_night = np.array(((197, 130), (197, 130), (675, 502), (670, 121)))

# Define the covariance matrices for the clusters at night
# These matrices determine the spread and orientation of user distributions
cov_cluster_night = np.array(
    (
        ((200, 0), (0, 150)),
        ((200, 0), (0, 150)),
        ((200, -100), (-100, 200)),
        ((200, 150), (150, 200)),
    )
)

# Define the mean locations of the clusters during daytime
loc_cluster_day = np.array(((197, 130), (380, 350), (380, 350), (670, 121)))

# Define the covariance matrices for the clusters in daytime settings
cov_cluster_day = np.array(
    (
        ((200, 0), (0, 150)),
        ((750, -250), (-250, 300)),
        ((750, -250), (-250, 300)),
        ((200, 150), (150, 200)),
    )
)

# Define whether the simulation pertains to daytime or nighttime conditions. The value can be 'day' or 'night'
time = "day"

# Instantiate a random number generator for reproducibility
rng = np.random.default_rng(0)

# Specify the number of data samples to generate
n_data = 10000

# Initialise arrays to store BS duty cycles (U) and BS loads (X)
U = np.zeros((n_data, len(loc_BS)))
X = np.zeros((n_data, len(loc_BS)))

# Main simulation loop over the specified number of samples
for i in tqdm(range(n_data)):
    # Generate user locations for each cluster based on time of day
    # Uses the appropriate mean and covariance matrix to simulate multivariate normal distributions
    user_loc = np.vstack(
        [
            rng.multivariate_normal(
                mean=eval(f"loc_cluster_{time}")[j],
                cov=10 * eval(f"cov_cluster_{time}")[j],
                size=N_cluster[j],
            )
            for j in range(4)
        ]
    )

    # Compute the squared Euclidean distance from each user to each base station
    dist = user_loc[:, None] - loc_BS
    dist = dist[..., 0] ** 2 + dist[..., 1] ** 2

    # Define connection masks for different neighbourhoods depending on the time of day
    if time == "night":
        # Neighbourhood 1: Randomly activate at least one of three base stations
        while True:
            u_n_1 = rng.choice([0, 1], size=3)
            if u_n_1.sum() > 0:
                break

        # Neighbourhood 2
        while True:
            u_n_2 = rng.choice([0, 1], size=3)
            if u_n_2.sum() > 0:
                break

        # Neighbourhood 3
        while True:
            u_n_3 = rng.choice([0, 1], size=3)
            if u_n_3.sum() > 0:
                break

        # Downtown area: all base stations inactive during nighttime
        u_d = np.zeros(4)

    if time == "day":
        # Neighbourhood 1
        while True:
            u_n_1 = rng.choice([0, 1], size=3)
            if u_n_1.sum() > 0:
                break

        # Neighbourhood 2
        while True:
            u_n_2 = rng.choice([0, 1], size=3)
            if u_n_2.sum() > 0:
                break

        # Neighbourhood 3: no active stations during daytime
        u_n_3 = np.zeros(3)

        # Downtown: at least one station must be active
        while True:
            u_d = rng.choice([0, 1], size=4)
            if u_d.sum() > 0:
                break

    # Concatenate all neighbourhood selections into a single binary vector
    u = np.hstack((u_n_1, u_n_2, u_n_3, u_d))

    # Set the distance to inactive base stations to infinity to exclude them
    dist[:, ~u.astype(bool)] = np.inf

    # Identify the nearest active base station for each user
    connection = dist.argmin(axis=-1)

    # Initialise the count of connections to each base station
    x = np.zeros(len(loc_BS), dtype=int)

    # Count the number of users connected to each base station
    for j, n in zip(*np.unique(connection, return_counts=True)):
        x[j] = n

    # Store connection mask and base station load for the current iteration
    U[i] = u
    X[i] = x

# Avoid division by zero or logarithmic singularities in downstream applications
X[X == 0] += 1e-6
U += 1e-6

# Persist the results to disk for future analysis
np.save(file=f"U_{time}", arr=U)
np.save(file=f"X_{time}", arr=X)
