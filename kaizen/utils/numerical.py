import math
from typing import Tuple, Union, Any

import numpy as np
from numpy import dot


def unit_vector(vec: tuple):
    """

    :param vec:
    :return: unit vector
    """
    return vec / np.linalg.norm(vec)


def angle_between_vector(v1: tuple, v2: tuple) -> float:
    """
    two vectors have either the same direction -  https://stackoverflow.com/a/13849249/71522
    :param v1:
    :param v2:
    :return:
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def dot_between_vector(v1: tuple, v2: tuple) -> Any:
    """
    two vectors have either the same direction -  https://stackoverflow.com/a/13849249/71522
    :param v1:
    :param v2:
    :return:
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)


def get_point_after_certain_distance(
    start: tuple, end: tuple, d: float, dt: float
) -> Tuple[float, float]:
    """
    https://math.stackexchange.com/questions/175896/finding-a-point-along-a-line-a-certain-distance-away-from-another-point

    :param start:
    :param end:
    :param d:
    :param dt:
    :return:
    """
    if d == 0:
        return start
    t = dt / d
    new_x = ((1 - t) * start[0]) + (t * end[0])
    new_y = ((1 - t) * start[1]) + (t * end[1])
    return new_x, new_y


def get_midpoint(start: tuple, end: tuple) -> Tuple[float, float]:
    """
    https://www.mathsisfun.com/algebra/line-midpoint.html

    :param start:
    :param end:
    :return:
    """
    return ((start[0] + end[0]) / 2), ((start[1] + end[1]) / 2)


def get_end_coordinate(
    start: tuple, angle_in_degree: float, distance: float
) -> Tuple[float, float]:
    """
    # https://math.stackexchange.com/questions/39390/determining-end-coordinates-of-line-with-the-specified-length-and-angle

    :param start:
    :param angle_in_degree:
    :param distance:
    :return:
    """
    x2 = start[0] + (distance * math.cos(angle_in_degree))
    y2 = start[1] + (distance * math.sin(angle_in_degree))
    return x2, y2


def get_center_of_mass(x, y) -> Tuple[Union[Any, float], Union[Any, float]]:
    """
    https://math.stackexchange.com/questions/24485/find-the-average-of-a-collection-of-points-in-2d-space

    :param x:
    :param y:
    :return:
    """
    return np.mean(x), np.mean(y)


def get_perpendicular_point(
    start: tuple, end: tuple, offset=10
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    https://stackoverflow.com/questions/133897/how-do-you-find-a-point-at-a-given-perpendicular-distance-from-a-line?rq=1

    :param start:
    :param end:
    :param offset:
    :return:
    """
    x1, y1 = start[0], start[1]
    x2, y2 = end[0], end[1]

    dx = x1 - x2
    dy = y1 - y2

    dist = math.sqrt((dx * dx) + (dy * dy))

    dx /= dist
    dy /= dist

    x3 = x1 + (offset * dy)
    y3 = y1 - (offset * dx)

    x4 = x1 - (offset * dy)
    y4 = y1 + (offset * dx)

    return (x3, y3), (x4, y4)


def vector(p1, p2) -> Tuple[float, float]:
    """

    :param p1:
    :param p2:
    :return:
    """
    return p1[0] - p2[0], p1[1] - p2[1]


def cosine_similarity(p1, p2, p3):
    """

    :param p1:
    :param p2:
    :param p3:
    :return:
    """
    a = vector((p1[0], p1[1]), (p2[0], p2[1]))
    b = vector((p1[0], p1[1]), (p3[0], p3[1]))
    return dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def diagonal_distance(p1: tuple, p2: tuple, d1: float, d2: float) -> float:
    """

    :param p1:
    :param p2:
    :param d1:
    :param d2:
    :return:
    """
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])

    return d1 * (dx + dy) + (d2 - 2 * d1) * min(dx, dy)


def euclidean_distance(p1: tuple, p2: tuple) -> float:
    """

    :param p1:
    :param p2:
    :return:
    """
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def manhattan_distance(p1: tuple, p2: tuple) -> float:
    """

    :param p1:
    :param p2:
    :return:
    """
    return abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])


def compute_center_of_mass(input_array, axis: int = 1):
    """
    Compute center of mass

    :param axis:
    :param input_array: (dim x N)
    :return:
    """
    return np.mean(input_array, axis=axis)


def new_mass(mean, input_array):
    """
    Subtract the corresponding center of mass from every point

    new_mass = (dim x N) - (dim x 1)

    :param mean:
    :param input_array:
    :return:
    """

    return input_array - mean[:, np.newaxis]


def decompose_matrix(matrix):
    """

    :param matrix:
    :return:
    """

    return np.linalg.svd(matrix)