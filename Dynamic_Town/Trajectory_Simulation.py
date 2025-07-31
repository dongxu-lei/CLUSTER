"""
Author: Dongxu Lei
Date: 11/05/2024
Description:
This script simulates the movement of mobile users (commuters and wanderers) in a city environment. 
It saves the trajectories of movement of each user into a "xy_data.npy" file and and generate a video of the simulation.
"""

from __future__ import annotations
from typing import NewType, Optional, Literal, Self
from itertools import product
from functools import cached_property
from tqdm import tqdm
from pathlib import Path
import numpy as np
from numpy.random import Generator
import cv2
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import imread
from scipy.spatial import Delaunay
import networkx as nx
from shapely.geometry import Point, Polygon
from abc import ABC, abstractmethod
from datetime import time, datetime, timedelta
import os
import sys

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

# Change the working directory to the script's directory
os.chdir(script_dir)

# # Snowmake Test
# if __name__ == "__main__":
#     test_data = np.random.randn(10)
#     np.save("xy_data.npy", test_data)
#     with open("commuter800_wanderer200_D1.mp4", "wb") as file:
#         pass

# exit(0)

def city_params() -> tuple[np.ndarray, dict, dict]:
    # Define road intersection locations in the virtual city
    inters = np.array([[442, 407], [565, 407], [655, 407], [774, 407],
                       [845, 407], [927, 407], [1025, 407], [1151, 407],
                       [1220, 407], [1445, 407], [442, 496], [565, 496],
                       [655, 496], [774, 496], [845, 496], [927, 496],
                       [1025, 496], [1151, 496], [1220, 496], [1445, 496],
                       [442, 574], [565, 574], [655, 574], [774, 574],
                       [847, 574], [1025, 574], [1151, 574], [1220, 574],
                       [1445, 574], [442, 656], [565, 656], [655, 656],
                       [774, 656], [847, 656], [1025, 656], [1151, 656],
                       [1220, 656], [1445, 656], [5, 574], [655, 730],
                       [774, 730], [847, 730], [1025, 733], [1151, 733],
                       [1220, 733], [1445, 733], [442, 818], [565, 818],
                       [655, 818], [774, 818], [885, 818], [931, 818],
                       [1025, 818], [1151, 818], [1222, 818], [1445, 818],
                       [442, 939], [565, 939], [655, 939], [774, 939],
                       [885, 939], [931, 939], [1025, 939], [442, 1097],
                       [565, 1097], [565, 1017], [655, 1017], [655, 1097],
                       [774, 1017], [774, 1097], [1025, 1097], [1151, 1097],
                       [1217, 1097], [1217, 1017], [1850, 1508], [1445, 1097],
                       [1445, 983], [442, 1191], [565, 1191], [655, 1191],
                       [774, 1191], [909, 1191], [1025, 1191], [1151, 1191],
                       [1217, 1191], [1445, 1191], [5, 1097], [5, 1191],
                       [442, 1508], [774, 1508], [909, 1508], [1025, 1508],
                       [1151, 1508], [5, 656], [1217, 1508], [1445, 1508],
                       [1151, 1017], [1850, 1097], [1850, 984], [1850, 818],
                       [1850, 574], [1850, 496], [1850, 407], [565, 1508],
                       [655, 1508], [5, 818], [1850, 9], [5, 9], [5, 1508],
                       [442, 9], [565, 9], [655, 9], [774, 9], [1025, 9],
                       [1151, 9], [1445, 9], [5, 407], [5, 496], [887, 733]])

    # Define road links in the virtual city
    links = {
        1: (110, 2, 11, 117),
        2: (1, 111, 3, 12),
        3: (2, 112, 4, 13),
        4: (3, 113, 5, 14),
        5: (4, 6, 15),
        6: (5, 16, 7),
        7: (6, 114, 8, 17),
        8: (7, 115, 9, 18),
        9: (8, 19, 10),
        10: (9, 116, 103, 20),
        11: (118, 1, 12, 21),
        12: (11, 2, 13, 22),
        13: (12, 3, 14, 23),
        14: (13, 4, 15, 24),
        15: (14, 5, 16),
        16: (15, 6, 17),
        17: (16, 7, 18, 26),
        18: (17, 8, 19, 27),
        19: (18, 9, 20, 28),
        20: (19, 10, 102, 29),
        21: (39, 11, 22, 30),
        22: (21, 12, 23, 31),
        23: (22, 13, 24, 32),
        24: (23, 14, 25, 33),
        25: (24, 26, 34),
        26: (25, 17, 27, 35),
        27: (26, 18, 28, 36),
        28: (27, 19, 29, 37),
        29: (28, 20, 101, 38),
        30: (94, 21, 31, 47),
        31: (30, 22, 32, 48),
        32: (31, 23, 33, 40),
        33: (32, 24, 34, 41),
        34: (33, 25, 35, 42),
        35: (34, 26, 36, 43),
        36: (35, 27, 37, 44),
        37: (36, 28, 38, 45),
        38: (37, 29, 46),
        39: (118, 21, 94),
        40: (32, 41, 49),
        41: (40, 33, 42, 50),
        42: (41, 34, 119),
        43: (119, 35, 44, 53),
        44: (43, 36, 45, 54),
        45: (44, 37, 46, 55),
        46: (45, 38, 56),
        47: (106, 30, 48, 57),
        48: (47, 31, 49, 58),
        49: (48, 40, 50, 59),
        50: (49, 41, 51, 60),
        51: (50, 119, 52, 61),
        52: (51, 53, 62),
        53: (52, 43, 54, 63),
        54: (53, 44, 55, 97),
        55: (54, 45, 56),
        56: (55, 46, 100, 77),
        57: (47, 58, 64),
        58: (57, 48, 59, 66),
        59: (58, 49, 67),
        60: (50, 61, 69),
        61: (60, 51, 62),
        62: (61, 52, 63),
        63: (62, 53, 71),
        64: (87, 57, 65, 78),
        65: (64, 66, 68, 79),
        66: (58, 67, 65),
        67: (66, 59, 69, 68),
        68: (65, 67, 70, 80),
        69: (67, 60, 70),
        70: (68, 69, 71, 81),
        71: (70, 63, 72, 83),
        72: (71, 97, 73, 84),
        73: (72, 74, 76, 85),
        74: (97, 73),
        75: (96, 98),
        76: (73, 77, 98, 86),
        77: (56, 99, 76),
        78: (88, 64, 79, 89),
        79: (78, 65, 80, 104),
        80: (79, 68, 81, 105),
        81: (80, 70, 82, 90),
        82: (81, 83, 91),
        83: (82, 71, 84, 92),
        84: (83, 72, 85, 93),
        85: (84, 73, 86, 95),
        86: (85, 76, 96),
        87: (106, 64, 88),
        88: (87, 78, 109),
        89: (109, 78, 104),
        90: (105, 81, 91),
        91: (90, 82, 92),
        92: (91, 83, 93),
        93: (92, 84, 95),
        94: (39, 30, 106),
        95: (93, 85, 96),
        96: (95, 86, 75),
        97: (54, 74, 72),
        98: (76, 99, 75),
        99: (77, 100, 98),
        100: (56, 101, 99),
        101: (29, 102, 100),
        102: (20, 103, 101),
        103: (10, 107, 102),
        104: (89, 79, 105),
        105: (104, 80, 90),
        106: (94, 47, 87),
        107: (116, 103),
        108: (110, 117),
        109: (88, 89),
        110: (108, 111, 1),
        111: (110, 112, 2),
        112: (111, 113, 3),
        113: (112, 114, 4),
        114: (113, 115, 7),
        115: (114, 116, 8),
        116: (115, 107, 10),
        117: (108, 1, 118),
        118: (117, 11, 39),
        119: (42, 43, 51),
    }

    # Define blocks in the virtual city
    regions = {
        0: [84, 85, 95, 94],
        1: [107, 109, 0, 116],
        2: [109, 110, 1, 0],
        3: [110, 111, 2, 1],
        4: [111, 112, 3, 2],
        5: [112, 113, 6, 5, 4, 3],
        6: [113, 114, 7, 6],
        7: [114, 115, 9, 8, 7],
        8: [115, 106, 102, 9],
        9: [116, 0, 10, 117],
        10: [0, 1, 11, 10],
        11: [1, 2, 12, 11],
        12: [2, 3, 13, 12],
        13: [3, 4, 14, 13],
        14: [4, 5, 15, 14],
        15: [5, 6, 16, 15],
        16: [6, 7, 17, 16],
        17: [7, 8, 18, 17],
        18: [8, 9, 19, 18],
        19: [9, 102, 101, 19],
        20: [117, 10, 20, 38],
        21: [10, 11, 21, 20],
        22: [11, 12, 22, 21],
        23: [12, 13, 23, 22],
        24: [13, 14, 15, 16, 25, 24, 23],
        25: [16, 17, 26, 25],
        26: [17, 18, 27, 26],
        27: [18, 19, 28, 27],
        28: [19, 101, 100, 28],
        29: [38, 20, 29, 93],
        30: [20, 21, 30, 29],
        31: [21, 22, 31, 30],
        32: [22, 23, 32, 31],
        33: [23, 24, 33, 32],
        34: [24, 25, 34, 33],
        35: [25, 26, 35, 34],
        36: [26, 27, 36, 35],
        37: [27, 28, 37, 36],
        38: [28, 100, 99, 55, 45, 37],
        39: [93, 29, 46, 105],
        40: [29, 30, 47, 46],
        41: [30, 31, 39, 48, 47],
        42: [31, 32, 40, 39],
        43: [32, 33, 41, 40],
        44: [33, 34, 42, 118, 41],
        45: [34, 35, 43, 42],
        46: [35, 36, 44, 43],
        47: [36, 37, 45, 44],
        48: [39, 40, 49, 48],
        49: [40, 41, 118, 50, 49],
        50: [118, 42, 52, 51, 50],
        51: [42, 43, 53, 52],
        52: [43, 44, 54, 53],
        53: [44, 45, 55, 54],
        54: [105, 46, 56, 63, 86],
        55: [46, 47, 57, 56],
        56: [47, 48, 58, 57],
        57: [48, 49, 59, 68, 66, 58],
        58: [49, 50, 60, 59],
        59: [50, 51, 61, 60],
        60: [51, 52, 62, 61],
        61: [52, 53, 96, 71, 70, 62],
        62: [53, 54, 55, 76, 75, 72, 73, 96],
        63: [55, 99, 98, 76],
        64: [56, 57, 65, 64, 63],
        65: [57, 58, 66, 65],
        66: [59, 60, 61, 62, 70, 69, 68],
        67: [65, 66, 67, 64],
        68: [66, 68, 69, 67],
        69: [96, 73, 72, 71],
        70: [76, 98, 97, 75],
        71: [86, 63, 77, 87],
        72: [63, 64, 78, 77],
        73: [64, 67, 79, 78],
        74: [67, 69, 80, 79],
        75: [69, 70, 82, 81, 80],
        76: [70, 71, 83, 82],
        77: [71, 72, 84, 83],
        78: [72, 75, 85, 84],
        79: [75, 97, 74, 95, 85],
        80: [87, 77, 88, 108],
        81: [77, 78, 103, 88],
        82: [78, 79, 104, 103],
        83: [79, 80, 89, 104],
        84: [80, 81, 90, 89],
        85: [81, 82, 91, 90],
        86: [82, 83, 92, 91],
        87: [83, 84, 94, 92]
    }

    return inters, links, regions


class Coordinate(ABC):

    def __init__(self, x: float, y: float) -> None:
        self.x: float = x
        self.y: float = y

    def __eq__(self, other: Coordinate) -> bool:
        if not isinstance(other, Coordinate):
            raise TypeError(
                "Unsupported operand type: Must be a Point instance.")
        return (self.x == other.x) and (self.y == other.y)

    def __sub__(self, other: Coordinate) -> float:
        if not isinstance(other, Coordinate):
            raise TypeError(
                "Unsupported operand type: Must be a Point instance.")
        return np.hypot(self.x - other.x, self.y - other.y)


InterID = NewType("InterID", int)


class Intersection(Coordinate):

    def __init__(self, inter_id: InterID, x: float, y: float) -> None:
        super().__init__(x, y)
        self.inter_id: IntersectionID = inter_id


class Network:

    def __init__(self, inters: np.ndarray, links: dict, regions: dict) -> None:
        """
        Initialize the Network class.

        Args:
            inters (np.ndarray): Array of intersections with their coordinates.
            links (dict): Dictionary defining connections between intersections.
            regions (dict): Dictionary defining regions in the network.
        """
        self._init_intersections(inters)
        self._link_intersections(links)
        self.links = links
        self.regions = regions
        self._store_shortest_path()

    def _init_intersections(self, inters: np.ndarray) -> None:
        """
        Initialize the intersections of the network.

        Args:
            inters (np.ndarray): Array of coordinates for each intersection.
        """
        self.intersections: dict[InterID, Intersection] = {}

        for i, (x, y) in enumerate(inters):
            self.intersections[InterID(i)] = Intersection(inter_id=InterID(i),
                                                          x=x,
                                                          y=y)

    def _link_intersections(self, links: dict) -> None:
        """
        Establish links between intersections based on given adjacency information.

        Args:
            links (dict): A dictionary defining connections between intersections.
        """
        self.inter_xy = np.array([[
            self.intersections[InterID(i)].x, self.intersections[InterID(i)].y
        ] for i in range(len(links))])

        self.adj_mat: np.ndarray = np.zeros((len(links), len(links)),
                                            dtype=bool)
        self.adj_dist: np.ndarray = np.zeros((len(links), len(links)))

        for i in range(len(links)):
            for j in links[i + 1]:
                self.adj_mat[i, j - 1] = 1
                self.adj_dist[i, j - 1] = self.intersections[InterID(
                    i)] - self.intersections[InterID(j - 1)]

    def visualise(self,
                  figsize: tuple[float, float] = None,
                  show_inter: bool = False,
                  ax: Optional[Axes] = None) -> None:
        """
        Visualise the network graph.

        Args:
            figsize (tuple, optional): Figure size for the plot.
            show_inter (bool, optional): If True, displays intersections as scatter points.
            ax (Optional[Axes], optional): Matplotlib Axes instance to plot on.
        """

        if not ax:
            ax: Axes = plt.figure(figsize=figsize,
                                  constrained_layout=True).add_subplot()

        if show_inter:
            ax.scatter(self.inter_xy[:, 0], self.inter_xy[:, 1])

        for i in range(len(self.intersections)):
            for j in range(i, len(self.intersections)):
                if self.adj_mat[i, j]:
                    ax.plot((self.intersections[InterID(i)].x,
                             self.intersections[InterID(j)].x),
                            (self.intersections[InterID(i)].y,
                             self.intersections[InterID(j)].y),
                            "r-",
                            linewidth=0.4)

            ax.text(self.intersections[InterID(i)].x,
                    self.intersections[InterID(i)].y,
                    s=str(i))

        ax.spines[["right", "top", "left", "bottom"]].set_visible(False)
        ax.set_xticks(())
        ax.set_yticks(())

    def _store_shortest_path(self) -> None:
        """
        Precompute shortest paths between all pairs of intersections.
        """
        self.G: nx.Graph = nx.Graph()

        edge_list: list[tuple[str, str, float]] = []
        for i in self.intersections:
            for j in self.links[i + 1]:
                edge_list.append((f"{i}", f"{j - 1}", self.adj_dist[i, j - 1]))
        self.G.add_weighted_edges_from(edge_list)

        dist = nx.floyd_warshall(self.G)
        self.pred, self.dist = nx.floyd_warshall_predecessor_and_distance(
            self.G)

    def get_shortest_inter_path(
            self,
            src: InterID,
            dst: InterID,
            visualise: bool = False) -> tuple[list[str], float]:
        """
        Get the shortest path between two intersections.

        Args:
            src (InterID): Source intersection.
            dst (InterID): Destination intersection.
            visualise (bool, optional): If True, visualizes the path.

        Returns:
            tuple: A list representing the shortest path and the distance.
        """

        for arg in (src, dst):
            if not isinstance(arg, int):
                raise TypeError(
                    "Unsupported operand type: Must be a InterID instance.")

            if arg < 0 or arg >= len(self.intersections):
                raise ValueError(
                    f"Operand must be within the range of [0, {len(self.intersections)}]"
                )

        path: list[str] = nx.reconstruct_path(f"{src}", f"{dst}", self.pred)
        dist: float = self.dist[f"{src}"][f"{dst}"]

        if visualise:
            ax: Axes = plt.figure(figsize=(10, 10),
                                  constrained_layout=True).add_subplot()
            self.visualise(figsize=(10, 10), show_inter=True, ax=ax)
            for i in range(len(path) - 1):
                ax.plot((self.intersections[InterID(eval(path[i]))].x,
                         self.intersections[InterID(eval(path[i + 1]))].x),
                        (self.intersections[InterID(eval(path[i]))].y,
                         self.intersections[InterID(eval(path[i + 1]))].y),
                        "r-",
                        linewidth=1)

        return path, dist

    @staticmethod
    def find_perpendicular_points(x1, y1, x2, y2, d):
        # Convert points to numpy arrays for vector operations
        M = np.array([x1, y1])
        N = np.array([x2, y2])

        # Vector from M to N
        MN = N - M

        # Normalized perpendicular vector (to MN)
        # If MN = (a, b), a perpendicular vector is (-b, a) or (b, -a).
        # We choose (-b, a) to ensure that P1 is on the left side of the vector MN
        perp_vector = np.array([-MN[1], MN[0]])
        perp_vector_norm = np.linalg.norm(perp_vector)
        perp_unit_vector = perp_vector / perp_vector_norm

        # The midpoint of P1 and P2 is the same as the midpoint of M and N
        midpoint = (M + N) / 2

        # Find P1 and P2 by moving d/2 from the midpoint in the direction of the perpendicular
        # P1 = midpoint + perp_unit_vector * (d / 2)
        P2 = midpoint - perp_unit_vector * (d / 2)

        return P2[0], P2[1]


class Location(Coordinate):

    def __init__(self,
                 x: float,
                 y: float,
                 loc_type: Literal["path", "site"],
                 net: Optional[Network] = None) -> None:
        """
        Initialize a Location instance.

        Args:
            x (float): x-coordinate of the location.
            y (float): y-coordinate of the location.
            loc_type (Literal): Type of location, either "path" or "site".
            net (Optional[Network]): Network object for "site" type locations.
        """
        super().__init__(x, y)

        assert loc_type in ("path",
                            "site"), "'loc_type' must be 'path' or 'site'."
        if loc_type == "site":
            assert isinstance(net,
                              Network), "A Network object must be provided."
            self._find_nearest_loc(net)

    def _find_nearest_loc(self, net: Network) -> None:
        """
        Find the nearest road or edge to the current location in the given network.

        Args:
            net (Network): Network object containing intersections and regions.
        """
        self.region_id: Optional[int] = None
        for region_id, region_vertices in net.regions.items():
            if Polygon([
                (net.intersections[int(v)].x, net.intersections[int(v)].y)
                    for v in region_vertices
            ]).contains(Point((self.x, self.y))):
                self.region_id = region_id
                break

        self.min_dist: float = np.inf
        self.nearest_x: Optional[float] = None
        self.nearest_y: Optional[float] = None
        self.nearest_edge_vertex_A: Optional[InterID] = None
        self.nearest_edge_vertex_B: Optional[InterID] = None

        for i in range(len(net.regions[self.region_id])):

            distance, foot_x, foot_y = self.distance_from_point_to_line(
                self.x, self.y,
                net.intersections[net.regions[self.region_id][i]].x,
                net.intersections[net.regions[self.region_id][i]].y,
                net.intersections[net.regions[self.region_id][(i + 1) % len(
                    net.regions[self.region_id])]].x,
                net.intersections[net.regions[self.region_id][(i + 1) % len(
                    net.regions[self.region_id])]].y)

            if distance < self.min_dist:
                self.nearest_x = foot_x
                self.nearest_y = foot_y
                self.min_dist = distance

                self.nearest_edge_vertex_A = InterID(
                    net.regions[self.region_id][i])
                self.nearest_edge_vertex_B = InterID(
                    net.regions[self.region_id][(i + 1) % len(
                        net.regions[self.region_id])])

    @staticmethod
    def distance_from_point_to_line(px, py, ax, ay, bx, by):
        # Line coefficients A, B, C for the equation Ax + By + C = 0
        A = by - ay
        B = ax - bx
        C = (bx * ay) - (ax * by)

        # Perpendicular distance from point to line
        distance = abs(A * px + B * py + C) / np.sqrt(A**2 + B**2)

        # Check if the foot of the perpendicular is within the segment
        dot_product = (px - ax) * (bx - ax) + (py - ay) * (by - ay)
        length_sq = (bx - ax)**2 + (by - ay)**2
        projection = dot_product / length_sq

        # Foot of the perpendicular
        foot_x = ax + projection * (bx - ax)
        foot_y = ay + projection * (by - ay)

        # If the dot_product is negative, it means the perpendicular
        # intersection is beyond point A, so the distance is from the point to A.
        # If projection is greater than 1, it means it's beyond point B, so the distance is from the point to B.
        # Otherwise, it's the calculated perpendicular distance.
        if dot_product < 0:
            distance = np.sqrt((px - ax)**2 + (py - ay)**2)
            foot_x, foot_y = ax, ay
        elif projection > 1:
            distance = np.sqrt((px - bx)**2 + (py - by)**2)
            foot_x, foot_y = bx, by
        else:
            # It's within the line segment.
            pass

        return distance, foot_x, foot_y


class ShortestPath:

    def __init__(self, src: Location, dst: Location, net: Network) -> None:
        """
        Initialize a ShortestPath instance.

        Args:
            src (Location): The starting location.
            dst (Location): The destination location.
            net (Network): The network graph containing intersections and paths.
        """
        self.src: Location = src
        self.dst: Location = dst

        self.shortest_path, self.shortest_dist = self.get_shortest_path(net)

    def get_shortest_path(self, net: Network) -> tuple[list[Location], float]:
        """
        Calculate the shortest path from source to destination.

        Args:
            net (Network): The network graph containing intersections and paths.

        Returns:
            tuple: A tuple containing a list of locations representing the shortest path,
                   and the total distance of that path.
        """
        shortest_path: list[Location] = [
            Location(self.src.x, self.src.y, "path"),
            Location(self.src.nearest_x, self.src.nearest_y, "path")
        ]
        A: InterId = self.src.nearest_edge_vertex_A
        B: InterId = self.src.nearest_edge_vertex_B
        C: InterId = self.dst.nearest_edge_vertex_A
        D: InterId = self.dst.nearest_edge_vertex_B

        inter_path: list[Location] = None
        min_dist: float = np.inf

        if {A, B} == {C, D}:
            inter_path = []
            min_dist = self.src.min_dist + (
                Location(self.dst.nearest_x, self.dst.nearest_y, "path") -
                Location(self.src.nearest_x, self.src.nearest_y,
                         "path")) + self.dst.min_dist
        else:
            for M, N in product((A, B), (C, D)):
                if M == N:
                    path = [M]
                    d: float = 0
                else:
                    path, d = net.get_shortest_inter_path(M, N)

                dist = self.src.min_dist + (net.intersections[M] - Location(
                    self.src.nearest_x, self.src.nearest_y, "path")) + d + (
                        Location(self.dst.nearest_x, self.dst.nearest_y,
                                 "path") -
                        net.intersections[N]) + self.dst.min_dist

                if dist < min_dist:
                    min_dist = dist
                    inter_path = path

        for p in inter_path:
            shortest_path.append(
                Location(net.intersections[int(p)].x,
                         net.intersections[int(p)].y, "path"))
        shortest_path.extend([
            Location(self.dst.nearest_x, self.dst.nearest_y, "path"),
            Location(self.dst.x, self.dst.y, "path")
        ])

        return shortest_path, min_dist

    def __iter__(self) -> Self:
        """
        Initialize the iterator for the path.
        """
        self.current_pos: Location = Location(self.src.x, self.src.y, "path")
        self.current_idx: int = 0
        return self

    def __next__(self) -> Location:
        """
        Return the next location in the shortest path.

        Raises:
            StopIteration: When the destination is reached.
        """
        if self.current_pos == Location(self.dst.x, self.dst.y, "path"):
            raise StopIteration

        self.current_pos = self.shortest_path[self.current_idx]
        self.current_idx += 1
        return self.current_pos


class Person(ABC):

    # Class-level attributes shared by all instances of Person
    current_time: datetime = datetime(1, 1, 1, 0, 0)

    # Speeds for moving along different types of roads
    block_speed: float = 2
    avenue_speed: float = 10

    @classmethod
    def increment_time(cls):
        """
        Increment the simulation time by one minute.
        """
        cls.current_time += timedelta(minutes=1)

    def __init__(self, x: float, y: float, rng: Generator = None) -> None:
        """
        Initialize a Person with given coordinates.

        Args:
            x (float): Initial x-coordinate of the person.
            y (float): Initial y-coordinate of the person.
            rng (Generator, optional): Random number generator instance. Defaults to None.
        """
        self.x: float = x
        self.y: float = y

        # Set random number generator, default to NumPy's generator if not provided
        self.rng: Generator = rng if rng != None else np.random.default_rng()

        # Initialize path-related attributes
        self.current_path: Optional[ShortestPath] = None
        self.current_edge: Optional[tuple[Location, Location]] = None
        self.current_speed: float = 0

        self.arrival_time: datetime = datetime(1, 1, 1, 0, 0)

    def move(self) -> None:
        """
        Update the person's position based on the current path and speed.
        """
        if self.current_edge != None:
            t_remain: float = 1
            while True:
                if self.current_edge[0] == self.current_path.shortest_path[
                        0] or self.current_edge[
                            -1] == self.current_path.shortest_path[-1]:
                    self.current_speed = Person.block_speed
                else:
                    self.current_speed = Person.avenue_speed

                dist_vec: np.ndarray = np.array(
                    (self.current_edge[1].x,
                     self.current_edge[1].y)) - np.array((self.x, self.y))
                dir_vec = dist_vec / np.linalg.norm(dist_vec, 2)

                if np.linalg.norm(dist_vec) >= self.current_speed * t_remain:
                    self.x += dir_vec[0] * self.current_speed * t_remain
                    self.y += dir_vec[1] * self.current_speed * t_remain
                    break

                else:
                    self.x = self.current_edge[-1].x
                    self.y = self.current_edge[-1].y

                    if self.current_edge[
                            -1] == self.current_path.shortest_path[-1]:
                        self.current_path = None
                        self.current_edge = None
                        self.arrival_time = Person.current_time
                        break

                    else:
                        t_remain -= np.linalg.norm(
                            dist_vec) / self.current_speed
                        self.current_edge = (self.current_edge[-1],
                                             next(self.path_iter))

    @abstractmethod
    def evolve(self) -> None:
        """
        Abstract method to be implemented by subclasses to update the person's state.
        """
        pass


class Commuter(Person):

    def __init__(self,
                 home: Location,
                 work: Location,
                 net: Network,
                 rng: Generator = None) -> None:
        """
        Initialize a Commuter instance.

        Args:
            home (Location): The home location of the commuter.
            work (Location): The work location of the commuter.
            net (Network): The city network.
            rng (Generator, optional): Random number generator. Defaults to None.
        """
        super().__init__(home.x, home.y, rng)

        # Set home and work locations for the commuter
        self.home: Location = home
        self.work: Location = work

        # Pre-calculate paths between home and work
        self.work_path: ShortestPath = ShortestPath(self.home, self.work, net)
        self.home_path: ShortestPath = ShortestPath(self.work, self.home, net)

    @cached_property
    def commuting_time(self) -> timedelta:
        """
        Calculate the total commuting time based on distance from home to work.

        Returns:
            timedelta: Estimated commuting time.
        """
        d1: float = self.home.min_dist
        d3: float = self.work.min_dist
        d2: float = self.work_path.shortest_dist - d1 - d3

        # Calculate commuting time based on speed limits for different segments
        return timedelta(minutes=int((d1 + d3) / self.block_speed +
                                     d2 / self.avenue_speed))

    def _determine_moving_time(self) -> None:
        """
        Determine the times at which the commuter leaves home and returns from work.
        """
        Δt0: int = int(self.rng.integers(low=-20, high=20))

        while True:
            Δt1: int = self.rng.geometric(p=0.15)
            if Δt1 <= 30:
                break

        while True:
            Δt2: int = self.rng.geometric(p=0.003)
            if Δt2 <= 180:
                break

        self.leave_time: time = (
            datetime(year=1, month=1, day=1, hour=9, minute=0) +
            timedelta(minutes=Δt0) - self.commuting_time -
            timedelta(minutes=Δt1)).time()
        self.return_time: time = (
            datetime(year=1, month=1, day=1, hour=16, minute=0) +
            timedelta(minutes=Δt2)).time()

    def evolve(self) -> None:
        """
        Update the commuter's state throughout the day, determining if they need to move.
        """
        if Person.current_time.time() == time(0, 0):
            self._determine_moving_time()

        if Person.current_time.time() == self.leave_time:
            self.current_path = self.work_path
            self.path_iter = iter(self.current_path)
            self.current_edge = (next(self.path_iter), next(self.path_iter))

        if Person.current_time.time() == self.return_time:
            self.current_path = self.home_path
            self.path_iter = iter(self.current_path)
            self.current_edge = (next(self.path_iter), next(self.path_iter))

        self.move()


class Wanderer(Person):

    # Class-level constants defining wandering behaviour
    low_wander_prob: float = 0.033
    high_wander_prob: float = 0.066
    back_prob: float = 0.1
    stay_time: timedelta = timedelta(minutes=15)

    # Time intervals governing wanderer's activity
    start_time: time = time(hour=5, minute=30)
    end_time: time = time(hour=19, minute=30)

    # Morning and afternoon peak wandering times
    morning_wander_start: time = time(hour=9, minute=30)
    morning_wander_end: time = time(hour=11, minute=30)
    afternoon_wander_start: time = time(hour=13, minute=30)
    afternoon_wander_end: time = time(hour=15, minute=30)

    def __init__(self,
                 home: Location,
                 net: Network,
                 rng: Generator = None) -> None:
        """
        Initialize a Wanderer instance.

        Args:
            home (Location): The home location of the wanderer.
            net (Network): The city network in which the wanderer navigates.
            rng (Generator, optional): Random number generator instance. Defaults to None.
        """
        super().__init__(home.x, home.y, rng)

        self.home: Location = home

    def evolve(self, net: Network) -> None:
        """
        Advance the state of the wanderer based on the time and wandering probabilities.
        
        Args:
            net (Network): The city network in which the wanderer navigates.
        """
        if (self.current_path
                == None) and (Person.current_time - self.arrival_time
                              >= self.stay_time):
            if Person.current_time.time(
            ) >= self.start_time and Person.current_time.time(
            ) < self.end_time:

                if (self.morning_wander_start <= Person.current_time.time() <
                        self.morning_wander_end) or (
                            self.afternoon_wander_start <=
                            Person.current_time.time() <
                            self.afternoon_wander_end):
                    wander_prob: float = self.high_wander_prob
                else:
                    wander_prob = self.low_wander_prob

                if self.rng.random() < wander_prob:
                    while True:
                        dst_loc: Location = Location(self.rng.uniform(5, 1850),
                                                     self.rng.uniform(9, 1508),
                                                     "site", net)
                        if dst_loc.min_dist > 15:
                            break

                    self.current_path = ShortestPath(
                        Location(self.x, self.y, loc_type="site", net=net),
                        dst_loc, net)
                    self.path_iter = iter(self.current_path)
                    self.current_edge = (next(self.path_iter),
                                         next(self.path_iter))

            if Person.current_time.time(
            ) < self.start_time or Person.current_time.time() >= self.end_time:
                if self.x != self.home.x or self.y != self.home.y:
                    if self.rng.random() < self.back_prob:
                        self.current_path = ShortestPath(
                            Location(self.x, self.y, loc_type="site", net=net),
                            self.home, net)
                        self.path_iter = iter(self.current_path)
                        self.current_edge = (next(self.path_iter),
                                             next(self.path_iter))

        self.move()


class Simulation:

    def __init__(self,
                 day: int,
                 n_commuter: int,
                 n_wanderer: int,
                 inters: np.ndarray,
                 links: dict,
                 regions: dict,
                 rng: Generator = None) -> None:
        """
        Initialize the Simulation class.

        Args:
            day (int): Number of days to run the simulation.
            n_commuter (int): Number of commuters.
            n_wanderer (int): Number of wanderers.
            inters (np.ndarray): Road intersections.
            links (dict): Road links between intersections.
            regions (dict): Regions in the city.
            rng (Generator): Random number generator instance.
        """
        # Set simulation parameters: number of days, commuters, and wanderers
        self.day: int = day
        self.n_commuter: int = n_commuter
        self.n_wanderer: int = n_wanderer

        # Define the number of neighborhoods in the city
        self.n_neighbourhood: int = 15

        # If no random generator is provided, create a default one
        if rng == None:
            rng = np.random.default_rng()
        self.rng: Generator = rng

        # Initialize the network of intersections, links, and regions
        self.net: Network = Network(inters, links, regions)

        # Initialize the commuter and wanderer users
        self._initialise_commuters()
        self._initialise_wanderers()

    def _initialise_commuters(self) -> None:
        """
        Initialize commuter users in the simulation.
        """
        self.commuters: list[Commuter] = []

        # Create random neighborhood centres for the city
        self.neighbourhood_centres: dict[int, dict[Literal["x", "y"],
                                                   float]] = {}
        for i in range(self.n_neighbourhood):
            self.neighbourhood_centres[i] = {
                "x": self.rng.uniform(305, 1585),
                "y": self.rng.uniform(244, 1268)
            }

        # Create home and work locations for each commuter
        for _ in range(self.n_commuter):
            # Randomly assign a neighbourhood for each commuter
            cluster_id: int = self.rng.integers(0, self.n_neighbourhood)

            # Assign a home location within the neighborhood bounds
            while True:
                x = self.rng.normal(
                    self.neighbourhood_centres[cluster_id]["x"], 80)
                y = self.rng.normal(
                    self.neighbourhood_centres[cluster_id]["y"], 65)

                if not ((5 < x < 1850) and (9 < y < 1508)):
                    continue

                home_loc: Location = Location(x, y, "site", self.net)

                if home_loc.min_dist > 15:
                    break

            # Assign a work location for the commuter
            while True:
                x = self.rng.standard_normal() * 300 + 945
                y = self.rng.standard_normal() * 250 + 750

                if not ((5 < x < 1850) and (9 < y < 1508)):
                    continue

                work_loc: Location = Location(x, y, "site", self.net)
                if work_loc.min_dist > 15:
                    break

            # Create a commuter with assigned home and work locations
            self.commuters.append(
                Commuter(home=home_loc,
                         work=work_loc,
                         net=self.net,
                         rng=self.rng))

    def _initialise_wanderers(self) -> None:
        """
        Initialize wanderer agents in the simulation.
        """
        self.wanderers: list[Wanderer] = []

        # Assign home locations to wanderers
        for _ in range(self.n_wanderer):
            cluster_id: int = self.rng.integers(0, self.n_neighbourhood)
            while True:
                home_loc: Location = Location(
                    self.rng.normal(
                        self.neighbourhood_centres[cluster_id]["x"], 80),
                    self.rng.normal(
                        self.neighbourhood_centres[cluster_id]["y"], 65),
                    "site", self.net)
                if (home_loc.min_dist > 15) and (5 < home_loc.x < 1850) and (
                        9 < home_loc.y < 1508):
                    break

            self.wanderers.append(
                Wanderer(home=home_loc, net=self.net, rng=self.rng))

    def run(self) -> np.ndarray:
        """
        Run the simulation of commuters and wanderers over a specified period.
        """
        # Create data structures to store the trace (x, y positions) of commuters and wanderers
        self.commuter_trace = {}
        self.wanderer_trace = {}
        self.all_trace = {}

        # Initialize the data structures for recording positions
        for i in range(self.n_commuter):
            self.commuter_trace[i] = {"x": [], "y": []}
        for i in range(self.n_wanderer):
            self.wanderer_trace[i] = {"x": [], "y": []}

        self.n_moving = np.zeros(24 * 60 * self.day, dtype=int)
        self.commuter_leave_time = []
        self.commuter_return_time = []

        # Loop through each minute of the simulation
        for t in tqdm(range(24 * 60 * self.day)):
            if Person.current_time.time() == time(hour=0, minute=0):
                for commuter in self.commuters:
                    commuter._determine_moving_time()

            self.all_trace[t] = {"x": [], "y": []}

            # Update the state of each commuter and record positions
            for i, commuter in enumerate(self.commuters):
                self.commuter_trace[i]["x"].append(commuter.x)
                self.commuter_trace[i]["y"].append(commuter.y)

                self.all_trace[t]["x"].append(commuter.x)
                self.all_trace[t]["y"].append(commuter.y)

                if commuter.current_edge:
                    self.n_moving[t] += 1

                # Record leaving and returning times for commuters
                if t % (24 * 60) == 0:
                    self.commuter_leave_time.append(commuter.leave_time)
                    self.commuter_return_time.append(
                        [commuter.return_time, commuter.commuting_time])

                commuter.evolve()  # Update commuter position

            # Update the state of each wanderer and record positions
            for i, wanderer in enumerate(self.wanderers):
                self.wanderer_trace[i]["x"].append(wanderer.x)
                self.wanderer_trace[i]["y"].append(wanderer.y)

                self.all_trace[t]["x"].append(wanderer.x)
                self.all_trace[t]["y"].append(wanderer.y)

                if wanderer.current_edge:
                    self.n_moving[t] += 1

                wanderer.evolve(self.net)  # Update wanderer position

            Person.increment_time(
            )  # Increment the simulation time for all users

        # Save the trace data for all agents to an output file
        dataset: np.ndarray = np.empty(
            (24 * 60 * self.day, self.n_commuter + self.n_wanderer, 2))
        for t in tqdm(range(24 * 60 * self.day)):
            dataset[t, :, 0] = np.array(self.all_trace[t]["x"])
            dataset[t, :, 1] = np.array(self.all_trace[t]["y"])

        # The shape of the dataset is (n_time_instant, n_user, 2)
        # The last dimension corresponds to the x-y coordinates of each user
        np.save("xy_data.npy", dataset)

        return dataset
        
        # np.save("n_moving_data.npy", self.n_moving)
        # np.save("commuter_leave_time.npy", np.array(commuter_leave_time))
        # np.save("commuter_return_time.npy", np.array(commuter_return_time))

    def visualise(self,
                  show_roads: bool = True,
                  background: np.ndarray = None) -> None:
        """
        Visualise the movement of agents in the city.

        Args:
            show_roads (bool): Whether to show road network in the visualisation.
            background (np.ndarray): Optional background image for the visualisation.
        """
        # Set up parameters for video output
        video_name: str = f"commuter{self.n_commuter}_wanderer{self.n_wanderer}_D{self.day}.mp4"
        save_path: Path = Path.cwd() / video_name
        fps: int = 24

        # Set up the figure for plotting
        fig, ax = plt.subplots(figsize=(12.8, 10.4), constrained_layout=True)
        fig.canvas.draw()

        # Set up the video writer
        w, h = fig.canvas.get_width_height()
        video = cv2.VideoWriter(str(save_path),
                                cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        colour_palette = [
            f"C{i}" for i in range(self.n_commuter + self.n_wanderer)
        ]

        # Iterate through each time step to generate video frames
        for t in tqdm(range(24 * 60 * self.day)):
            ax.clear()

            # Show the roads if required
            if show_roads:
                self.net.visualise(ax=ax)

            # Add the background image if provided
            if isinstance(background, np.ndarray):
                ax.imshow(background, origin="lower")
                ax.set_xticks([])
                ax.set_yticks([])

            # Plot the users' current positions
            ax.scatter(self.all_trace[t]["x"],
                       self.all_trace[t]["y"],
                       color=colour_palette,
                       edgecolor="white",
                       alpha=0.75)

            # Set the title to the current simulation time and axis limits for the plot
            ax.set_title(f"{datetime(1, 1, 1, 0, 0) + timedelta(minutes=t)}")
            ax.set_xlim((305, 1585))
            ax.set_ylim((244, 1268))

            # Draw the canvas and capture the current frame
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.buffer_rgba(),
                                  dtype=np.uint8).reshape(h, w, 4)[..., :-1]
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Write the frame to the video
            video.write(image)

        # Release the video writer after writing all frames
        video.release()


# Main function to run the simulation
def main() -> None:
    # Initialize random number generator
    rng = np.random.default_rng(42)

    # Get city parameters (intersections, links, regions)
    inters, links, regions = city_params()

    # Create a Simulation object with initial parameters
    sim = Simulation(day=360,
                     n_commuter=700,
                     n_wanderer=300,
                     inters=inters,
                     links=links,
                     regions=regions,
                     rng=rng)

    # Run the simulation
    dataset = sim.run()

    # Generate a demo video
    # Only run this for a demo video, as it requires considerable amount of time
    # sim.visualise(show_roads=False, background=imread("./Street_Map.png"))


# Entry point of the script
if __name__ == "__main__":
    main()
