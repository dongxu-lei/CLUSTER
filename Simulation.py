from argparse import ArgumentParser, Namespace

import numpy as np
from numpy.random import Generator

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

import cv2
from pathlib import Path
from shutil import rmtree

from typing import Any, Final
from tqdm import tqdm
from pprint import pprint


class User:
    Δt: Final[float] = 5e-3

    def __init__(
        self,
        x_0: np.ndarray,
        μ: np.ndarray,
        θ: float,
        σ: float,
        rng: Generator,
    ) -> None:
        self.μ: np.ndarray = μ
        self.θ: float = θ
        self.σ: float = σ

        self.x: np.ndarray = x_0

        self.std_asy: float = np.sqrt(σ**2 / (2 * θ) *
                                      (1 - np.exp(-2 * θ * self.Δt)))
        self.rng: Generator = rng

    def __call__(self) -> np.ndarray:

        self.x = self.μ + (self.x - self.μ) * np.exp(
            -self.θ *
            self.Δt) + self.std_asy * self.rng.standard_normal(size=2)

        return self.x


class Network:

    def __init__(self,
                 N: int,
                 T: int,
                 M: int,
                 S: float,
                 seed: int,
                 period: int = 60) -> None:
        self.N: int = N
        self.T: int = T
        self.M: int = M
        self.S: float = S
        self.rng: Generator = np.random.default_rng(seed=seed)
        self.period: int = period

        self.units: np.ndarray = self.rng.random(size=(self.N, 2))
        self.instant_status: np.ndarray = np.full(self.N, True, dtype=bool)
        self.status_data: np.ndarray = np.empty((self.T, self.N))
        self.instant_load: np.ndarray = np.zeros(self.N, dtype=int)
        self.load_data: np.ndarray = np.empty((self.T, self.N))

        self.user_loc: np.ndarray = np.zeros((self.M, 2))
        self.user_alloc: dict[str, int] = {}
        self.users: dict[str, User] = self._generate_users()

    def _generate_users(self):
        users: dict[str, User] = {}
        for j in range(self.M):
            x_0: np.ndarray = self.rng.random((1, 2))
            σ: float = np.abs(
                self.rng.normal(loc=0, scale=self.S / np.sqrt(self.N)))
            users[f"{j}"] = User(x_0=x_0, μ=x_0, θ=1, σ=σ, rng=self.rng)

            self.user_loc[j] = x_0

        return users

    def _nearest_alloc(self) -> None:
        self.instant_load = np.zeros(self.N, dtype=int)
        for idx, user in self.users.items():
            x: np.ndarray = user.x
            valid_units: np.ndarray = self.units.copy()
            valid_units[~self.instant_status] = np.inf

            dist: np.ndarray = np.linalg.norm(x - valid_units, ord=2, axis=1)
            self.user_alloc[idx] = dist.argmin()
            self.instant_load[self.user_alloc[idx]] += 1

    def _evolve(self) -> None:
        for j in range(self.M):
            self.user_loc[j] = self.users[f"{j}"]()
        self._nearest_alloc()

    def simulate(self, visualize: bool, fps: int) -> None:

        if visualize:
            fig: Figure = plt.figure(figsize=(10, 10), constrained_layout=True)
            ax: Axes = fig.add_subplot()

            tmp_path: Path = Path("tmp")
            if tmp_path.exists():
                rmtree(tmp_path)
            tmp_path.mkdir()

        for t in tqdm(range(self.T)):
            switch_time: np.ndarray = (self.rng.random(self.N) *
                                       self.period).astype(int)
            active_unit_count: np.ndarray = np.zeros(self.N, dtype=int)
            load_count: np.ndarray = np.zeros(self.N, dtype=int)

            for p in range(self.period):
                # Avoid complete network paralysis
                tmp_status: np.ndarray = self.instant_status.copy()
                tmp_status[switch_time == p] = ~tmp_status[switch_time == p]
                if tmp_status.sum() != 0:
                    self.instant_status = tmp_status

                self._evolve()
                active_unit_count += self.instant_status
                load_count += self.instant_load

                if visualize:
                    ax.clear()
                    ax.scatter(
                        *self.units.T,
                        color=[
                            f"C{i}" if self.instant_status[i] else "gray"
                            for i in range(self.N)
                        ],
                        marker="s")
                    ax.scatter(*self.user_loc.T,
                               color=[
                                   f"C{self.user_alloc[str(i)]}"
                                   for i in range(self.M)
                               ],
                               alpha=3 / np.sqrt(self.M))

                    ax.set_xlim(-0.5, 1.5)
                    ax.set_ylim(-0.5, 1.5)
                    ax.set_axis_off()

                    ax.scatter(x=10,
                               y=10,
                               edgecolor="gray",
                               facecolor="white",
                               marker="s",
                               label="Resource Unit")
                    ax.scatter(x=10,
                               y=10,
                               edgecolor="gray",
                               facecolor="white",
                               label="User")
                    ax.legend(loc="upper right",
                              prop=FontProperties(family="serif",
                                                  size="xx-large"))

                    fig.savefig(tmp_path / f"{t * self.period + p}.png")

            self.status_data[t] = active_unit_count / self.period
            self.load_data[t] = load_count / self.period

        save_path: Path = Path("Simulation_Result")
        save_path.mkdir(exist_ok=True)

        np.savez(save_path / f"sim_data_N{self.N}_T{self.T}_M{self.M}_S{self.S}.npz",
                 status=self.status_data,
                 load=self.load_data)

        if visualize:
            self.generate_video(
                save_path / f"sim_data_N{self.N}_T{self.T}_M{self.M}_S{self.S}.mp4",
                tmp_path, fps)
            rmtree(tmp_path)

    @staticmethod
    def generate_video(save_path: Path, tmp_path: Path, fps: int = 60):
        tmp: np.ndarray = cv2.imread(str(tmp_path / "0.png"))
        h, w = tmp.shape[:2]
        video: cv2.VideoWriter = cv2.VideoWriter(
            str(save_path), cv2.VideoWriter_fourcc("m", "p", "4", "v"), fps,
            (w, h))

        n_frame: int = len(list(tmp_path.iterdir()))
        for i in tqdm(range(0, n_frame)):
            image: np.ndarray = cv2.imread(str(tmp_path / f"{i}.png"))
            video.write(image)

        video.release()


def parse_args() -> dict[str, Any]:
    args_dict: dict[str, Any] = {}

    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("N", help="number of resource units", type=int)
    parser.add_argument("T", help="size of historical data", type=int)
    parser.add_argument("M", help="number of total users", type=int)
    parser.add_argument("S",
                        help="scale parameter of user mobility",
                        type=float)

    parser.add_argument("-s",
                        "--seed",
                        help="seed of random number",
                        type=int,
                        default=0)
    parser.add_argument("-v",
                        "--visualize",
                        help="generate demo video",
                        action="store_true")
    parser.add_argument("-f",
                        "--fps",
                        help="fps of demo video",
                        type=int,
                        default=60)

    args: Namespace = parser.parse_args()

    assert args.N > 0, "N must be a postive integer"
    args_dict["N"] = args.N
    assert args.T > 0, "T must be a postive integer"
    args_dict["T"] = args.T
    assert args.M > 0, "M must be a postive integer"
    args_dict["M"] = args.M
    assert args.S > 0, "S must be a postive integer"
    args_dict["S"] = args.S

    args_dict["seed"] = args.seed
    args_dict["visualize"] = args.visualize

    assert args.fps > 0, "fps must be a postive integer"
    args_dict["fps"] = args.fps

    return args_dict


def main() -> None:
    args = parse_args()

    network = Network(N=args["N"],
                      T=args["T"],
                      M=args["M"],
                      S=args["S"],
                      seed=args["seed"])
    network.simulate(visualize=args["visualize"], fps=args["fps"])


if __name__ == "__main__":
    main()
