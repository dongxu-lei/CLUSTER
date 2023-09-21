from argparse import ArgumentParser
import numpy as np
import pymc as pm
from pytensor import tensor as pt
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


def Naive_CLUSTER(n_model_component: int, n_node: int,
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


    stick_breaking = lambda β: pt.exp(
        pt.log(β) + pt.concatenate([[0],
                                    pt.extra_ops.cumsum(pt.log(1 - β))[:-1]]))

    with pm.Model(coords=coords) as model:
        w = pm.Dirichlet("w", np.ones(n_model_component), dims="component")
        L = pm.Dirichlet("T",
                         np.ones((n_model_component, n_node)),
                         dims=["component", "node"])

        status = pm.MutableData("node_status", status_broad)

        L_hat = L * status
        L_hat /= L_hat.sum(axis=-1, keepdims=True)

        w_broad = pt.extra_ops.broadcast_to(w,
                                            (data_length, n_model_component))
        p = pt.batched_tensordot(L_hat, w_broad, axes=[[1], [1]])
        p /= p.sum(axis=1, keepdims=True)

        c = pm.Gamma("c", 1000, 1)

        obs = pm.Dirichlet("obs",
                           p * c,
                           dims=["obs_id", "node"],
                           observed=load_train /
                           load_train.sum(axis=1, keepdims=True))

    return model


def sample(status_train: np.ndarray, load_train: np.ndarray,
           n_model_component: int, sampler: str) -> None:
    """Sample the posterior distribution using MCMC methods.

    Parameters
    ----------
    status_train : np.ndarray
        Training set of the status data.
    load_train : np.ndarray
        Training set of the load data.
    n_model_component : int
        Number of preset model components in the network.
    sampler: str
        Which sampler to run.
    """

    with Naive_CLUSTER(n_model_component=n_model_component,
                       n_node=status_train.shape[1],
                       status_train=status_train,
                       load_train=load_train):

        trace = pm.sample(
            draws=5000,
            tune=2000,
            init="advi",
            target_accept=0.99,
            nuts_sampler=sampler,
            nuts_sampler_kwargs={"postprocessing_backend": "cpu"},
            chains=8 if sampler in ("blackjax", "numpyro") else 4)

    trace.to_netcdf(
        f"./Naive_Trace/sim_data_N10_T2000_M100_S0.5_W{n_model_component}.nc")


def parse_args() -> dict[str, Any]:
    args_dict: dict[str, Any] = {}
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("-c",
                        "--component",
                        help="number of preset model components",
                        type=int)
    parser.add_argument("-s",
                        "--sampler",
                        help="which NUTS implementation to run.",
                        type=str,
                        default="blackjax")

    args: Namespace = parser.parse_args()

    args_dict["sampler"] = args.sampler
    args_dict["n_model_component"] = args.component

    return args_dict


def main() -> None:
    args = parse_args()

    status_train, status_test, load_train, load_test = load_data(
        "/sim_data_N10_T2000_M100_S0.5"
    )
    sample(status_train,
           load_train,
           n_model_component=args["n_model_component"],
           sampler=args["sampler"])


if __name__ == "__main__":
    main()