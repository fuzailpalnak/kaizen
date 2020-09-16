import cv2
import geopandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

from shapely.geometry import Polygon, mapping

from kaizen.map.trace import TracePoint
from kaizen.utils.gis import geom_check, decompose_data_frame_row


class Obstacle:
    def __init__(self, obs_map):
        self.obs_map = obs_map

    def animate_obstacle(self):
        x, y = np.where(self.obs_map == True)
        plt.plot(x, y, ".k")

    @staticmethod
    def inner_obstacle_pixel(mx, my, boundary_obstacle):
        """
        MARK ALL INNER PIXEL AS OBSTACLE THIS WILL HELP IN IDENTIFYING IF ANY OF THE INPUT VERTICES FOR PATH FINDING
        LIES INSIDE THE POLYGON

        # https://stackoverflow.com/questions/21339448/how-to-get-list-of-points-inside-a-polygon-in-python

        :param mx:
        :param my:
        :param boundary_obstacle:
        :return:
        """
        x, y = mx.flatten(), my.flatten()
        points = np.vstack((x, y)).T

        polygon = Path(boundary_obstacle)
        inner_pixel = polygon.contains_points(points)
        return inner_pixel

    @staticmethod
    def obstacle_mesh(grid):
        mx, my = np.mgrid[0 : grid.x_width, 0 : grid.y_width]
        mx = (mx * grid.resolution) + grid.minx
        my = (my * grid.resolution) + grid.miny
        return mx, my

    def collision_check(self, trace_point: TracePoint):
        return self.obs_map[trace_point.x][trace_point.y]


def generate_obstacle(obstacle_file: str, grid, extend_boundary_pixel: int) -> Obstacle:
    """
    Generate Obstacle for GeoSpatial Polygon

    :param obstacle_file:
    :param grid:
    :param extend_boundary_pixel:
    :return:
    """
    obstacle_data = geopandas.read_file(obstacle_file)
    assert geom_check(
        obstacle_data, "Polygon"
    ), "Expected all geometries in to be Polygon"

    extent = np.zeros([grid.y_width, grid.x_width], np.uint8)

    for idx, feature in obstacle_data.iterrows():
        feature_geometry, feature_property = decompose_data_frame_row(feature)

        coordinates = feature_geometry["coordinates"]
        for sub_coordinate in coordinates:
            boundary_coordinates = mapping(Polygon(sub_coordinate).boundary)[
                "coordinates"
            ]
            corner_points = [
                grid.to_pixel_position(bo[0], bo[1]) for bo in boundary_coordinates
            ]
            corner_points = np.array(corner_points[:-1], np.int32)

            # https://stackoverflow.com/questions/14161331/creating-your-own-contour-in-opencv-using-python
            cv2.drawContours(
                extent,
                [corner_points],
                0,
                color=(255, 255, 255),
                thickness=extend_boundary_pixel,
            )
            cv2.fillPoly(extent, pts=[corner_points], color=(255, 255, 255))

    obs_map = extent >= 255
    return Obstacle(obs_map=obs_map.T)
