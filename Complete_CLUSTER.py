from argparse import ArgumentParser
import numpy as np
from numpy.lib.npyio import NpzFile
import pymc as pm
from pytensor import tensor as pt
from pymc.sampling.jax import sample_blackjax_nuts, sample_numpyro_nuts
from typing import Any


def load_data(input: str) -> tuple[np.ndarray, ...]:
    """Load and partition synthetic data. 

    Parameters
    ----------
    input : str
        history data containing "status" and "load".

    Returns
    -------
    tuple[np.ndarray, ...]
        Partitioned data, with the sequence of (status_train, status_test, load_train, load_test)
    """

    data: NpzFile = np.load(f"{input}.npz")
    status: np.ndarray = data["status"]
    load: np.ndarray = data["load"]

    assert len(status) == len(
        load), "Status and load data must be of the same length."
    T: int = len(status) // 2

    # Avoid null status for numerical stability
    status[status == 0] = 1e-6
    status /= status.sum(axis=1, keepdims=True)

    status_train: np.ndarray = status[:T, ...]
    status_test: np.ndarray = status[T:, ...]
    load_train: np.ndarray = load[:T, ...]
    load_test: np.ndarray = load[T:, ...]
    
    load_train += 1e-3

    return status_train, status_test, load_train, load_test


def Complete_CLUSTER(n_model_component: int, n_node: int,
                     status_train: np.ndarray,
                     load_train: np.ndarray) -> pm.Model:

    data_length: int = len(status_train)
    status_broad = np.broadcast_to(status_train[:, None, :],
                                   (data_length, n_model_component, n_node))

    coords = {
        "component": np.arange(n_model_component),
        "obs_id": np.arange(data_length),
        "node": np.arange(n_node)
    }

    stick_breaking = lambda β: β * pt.concatenate(
        [[1], pt.extra_ops.cumprod(1 - β)[:-1]])

    with pm.Model(coords=coords) as model:
        α = pm.Gamma("α", 5, 1.0)
        β = pm.Beta("β", 1.0, α, dims="component")
        w = pm.Deterministic("w", stick_breaking(β), dims="component")

        L = pm.Dirichlet("L",
                         np.ones((n_model_component, n_node)),
                         dims=["component", "node"])

        status = pm.MutableData("node_status", status_broad)

        L_hat = L * status
        L_hat /= L_hat.sum(axis=-1, keepdims=True)

        w_broad = pt.extra_ops.broadcast_to(w,
                                            (data_length, n_model_component))
        p = pt.batched_tensordot(L_hat, w_broad, axes=[[1], [1]])
        p /= p.sum(axis=1, keepdims=True)

        c = pm.Uniform("c", lower=10, upper=100)

        obs = pm.Dirichlet("obs",
                           p * c,
                           dims=["obs_id", "node"],
                           observed=load_train /
                           load_train.sum(axis=1, keepdims=True))

    return model


def sample(status_train: np.ndarray, load_train: np.ndarray,
           n_model_component: int, output: str) -> None:
    """Sample the posterior distribution using MCMC methods.

    Parameters
    ----------
    status_train : np.ndarray
        Training set of the status data.
    load_train : np.ndarray
        Training set of the load data.
    n_node : int
        Number of nodes in the network.
    output : str
        Name of the output trace file.
    """

    T: int = len(status_train)

    with Complete_CLUSTER(n_model_component=n_model_component,
                          n_node=status_train.shape[1],
                          status_train=status_train,
                          load_train=load_train):
        trace = sample_blackjax_nuts(draws=2000,
                                     tune=1000,
                                     init="advi",
                                     target_accept=0.99,
                                     postprocessing_backend="gpu",
                                     chains=8)

    trace.to_netcdf(f"{output}_W{n_model_component}_new.nc")


def parse_args() -> dict[str, Any]:
    args_dict: dict[str, Any] = {}
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("W", help="number of model_components", type=int)
    parser.add_argument("-i",
                        "--input",
                        help="filename of the input data file",
                        type=str,
                        default="DPMM_model_synthetic")

    args: Namespace = parser.parse_args()

    assert args.W > 0, "W must be a postive integer"
    args_dict["W"] = args.W
    args_dict["input"] = args.input

    return args_dict


def main() -> None:
    args = parse_args()

    status_train, status_test, load_train, load_test = load_data(args["input"])
    sample(status_train,
           load_train,
           n_model_component=args["W"],
           output=args["input"])


if __name__ == "__main__":
    main()