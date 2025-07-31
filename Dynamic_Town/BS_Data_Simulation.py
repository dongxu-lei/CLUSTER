# This script transforms the xy trajectory data of each user during a given interval to the x and u form.

import numpy as np
from numpy.random import Generator
import pickle
import os
import sys

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

# Change the working directory to the script's directory
os.chdir(script_dir)

# # Snakemake Test
# if __name__ == "__main__":
#     with open("xy_data.npy", "rb") as file:
#         pass

#     test_data = np.random.randn(10)
#     with open("Dynamic_Town.pickle", "wb") as file:
#         pickle.dump(test_data, file)

# exit(0)

# Set number of base stations and initialise random generator
n_base_station: int = 15
rng: Generator = np.random.default_rng(0)

print("calculating distances...")

# Load user trajectory data: shape (Time, User, 2)
users_loc: np.ndarray = np.load("./xy_data.npy")

# Generate random base station coordinates
base_stations_loc: np.ndarray = np.empty((n_base_station, 2))
base_stations_loc[:, 0] = rng.uniform(305, 1585, n_base_station)  # x-coordinates
base_stations_loc[:, 1] = rng.uniform(244, 1268, n_base_station)  # y-coordinates

# Expand dimensions for broadcasting in distance calculation
users_loc_expanded = np.expand_dims(users_loc, axis=2)  # (T, U, 1, 2)
base_stations_loc_expanded = np.expand_dims(
    np.expand_dims(base_stations_loc, axis=0), axis=1
)  # (1, 1, B, 2)

# Compute Euclidean distances from users to all base stations
distances = np.linalg.norm(users_loc_expanded - base_stations_loc_expanded, axis=-1)

print("calculating station status...")

# Setup parameters for temporal and spatial dimensions
time_steps = distances.shape[0]
station_number = distances.shape[2]
min_guarantee_number = 2

# Ensure at least 2 base stations are active per time step
min_guarantee_indices = rng.random((time_steps, station_number)).argsort(1)[
    :, :min_guarantee_number
]

# Randomly assign on/off status to each base station
random_station_status = rng.choice(a=[False, True], size=(time_steps, 15), p=[0.5, 0.5])
random_station_status[np.arange(time_steps)[:, None], min_guarantee_indices] = (
    True  # enforce minimum active
)

# Compute inverse distances and mask by station status (inactive stations get zero weight)
distances_inverse = 1 / (distances + 1e-9)
distances_inverse_masked = distances_inverse * random_station_status[:, None, :]

# Assign each user to the nearest active base station
connected_station = np.argmax(distances_inverse_masked, axis=2)  # (T, U)

# Aggregate over fixed periods (e.g., 15 time steps)
period_length = 15
aggregated_station_stauts = random_station_status.reshape(
    -1, period_length, *random_station_status.shape[1:]
)
aggregated_connected_station = connected_station.reshape(
    -1, period_length, *connected_station.shape[1:]
)

# Compute average activation time for each station across each period
average_station_status = np.average(aggregated_station_stauts, axis=1)

# Compute average user load per station across each period
station_average_user = (
    np.sum(
        aggregated_connected_station[..., None] == np.arange(station_number),
        axis=(1, 2),
    )
    / period_length
)

print("saving...")

# Save results in a dictionary for downstream use
# The dataset is of the shape of (n_data, n_BS)
dataset = {}
dataset["U"] = average_station_status  # Station availability over time
dataset["X"] = station_average_user  # Station load over time

# Serialize and store the dataset
with open("Dynamic_Town.pickle", "wb") as file:
    pickle.dump(dataset, file)

print("done.")
