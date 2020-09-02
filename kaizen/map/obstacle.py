import fiona
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

from shapely.geometry import Polygon, mapping


class Obstacle:
    def __init__(self, obstacle_position):
        self.obstacle_position = obstacle_position

    @classmethod
    def from_polygon(cls, grid, polygon: fiona.collection, extend_boundary_pixel: 2):
        """
        GENERATE OBSTACLE MAP FROM SPATIAL GEOMETRY [POLYGON]

        :param grid:
        :param polygon:
        :param extend_boundary_pixel:
        :return:
        """
        extent = np.zeros([grid.y_width, grid.x_width], np.uint8)  # create a single channel 200x200 pixel black image

        for i, feature in enumerate(polygon):
            if feature["type"] == "MultiPolygon":
                raise NotImplementedError("MultiPolygon Not Implemented")
            coordinates = feature["geometry"]["coordinates"]
            for sub_coordinate in coordinates:
                boundary_coordinates = mapping(Polygon(sub_coordinate).boundary)[
                    "coordinates"
                ]
                corner_points = [grid.to_pixel_position(bo[0], bo[1]) for bo in boundary_coordinates]
                corner_points = np.array(corner_points[:-1], np.int32)

                # https://stackoverflow.com/questions/14161331/creating-your-own-contour-in-opencv-using-python
                cv2.drawContours(extent, [corner_points], 0, color=(255, 255, 255), thickness=extend_boundary_pixel)
                cv2.fillPoly(extent, pts=[corner_points], color=(255, 255, 255))

        obstacle_position = (extent >= 255)
        return cls(obstacle_position=obstacle_position.T)

    def animate_obstacle(self):
        x, y = np.where(self.obstacle_position == True)
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
        mx, my = np.mgrid[0: grid.x_width, 0: grid.y_width]
        mx = (mx * grid.resolution) + grid.minx
        my = (my * grid.resolution) + grid.miny
        return mx, my
