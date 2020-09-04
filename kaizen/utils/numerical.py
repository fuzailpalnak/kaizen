import math

import numpy as np
from numpy import dot


def unit_vector(vec: tuple):
    """

    :param vec:
    :return: unit vector
    """
    return vec / np.linalg.norm(vec)


def angle_between_vector(v1: tuple, v2: tuple):
    """
    two vectors have either the same direction -  https://stackoverflow.com/a/13849249/71522
    :param v1:
    :param v2:
    :return:
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_point_after_certain_distance(
    start: tuple, end: tuple, d: float, dt: float
) -> tuple:
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


def get_midpoint(start: tuple, end: tuple) -> tuple:
    """
    https://www.mathsisfun.com/algebra/line-midpoint.html

    :param start:
    :param end:
    :return:
    """
    return ((start[0] + end[0]) / 2), ((start[1] + end[1]) / 2)


def get_end_coordinate(start: tuple, angle_in_degree: float, distance: float) -> tuple:
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


def get_center_of_mass(x, y):
    """
    https://math.stackexchange.com/questions/24485/find-the-average-of-a-collection-of-points-in-2d-space

    :param x:
    :param y:
    :return:
    """
    return np.mean(x), np.mean(y)


def get_perpendicular_point(start: tuple, end: tuple, offset=10):
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


def vector(p1, p2):
    return p1[0] - p2[0], p1[1] - p2[1]


def cosine_similarity(p1, p2, p3):
    a = vector((p1[0], p1[1]), (p2[0], p2[1]))
    b = vector((p1[0], p1[1]), (p3[0], p3[1]))
    return dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
