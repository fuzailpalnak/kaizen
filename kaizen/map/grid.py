from typing import Union

import cv2
import numpy as np

from affine import Affine
from geopandas import GeoDataFrame
from shapely.geometry import Polygon, mapping

from kaizen.utils.gis import (
    geom_check,
    decompose_data_frame_row,
    generate_affine,
    pixel_position, supported_crs,
)


class Grid:
    def __init__(
        self, bounds: tuple, resolution: int, dimension: tuple, transform: Affine
    ):
        """

        :param resolution:
        :param bounds: bounds of the grid
        :param dimension: height and width
        :param transform: Affine transformation of the grid
        """
        self.min_x, self.min_y, self.max_x, self.max_y = bounds
        self.resolution = resolution
        self.width, self.height = dimension

        self._obstacle = np.zeros([self.width, self.height], np.uint8)
        self._transform = transform

    @property
    def transform(self):
        return self._transform

    @property
    def extent(self):
        return self.min_x, self.min_y, self.max_x, self.max_y

    @property
    def obstacle(self):
        return self._obstacle

    @obstacle.setter
    def obstacle(self, value):
        self._obstacle = value

    @property
    def dimension(self):
        return self.width, self.height

    def grid_index(self, x, y):
        """
        GET THE POSITION OF X, Y ON THE GRID
        IF THE GRID IS 10X10 THEN THERE ARE 100 SUB GRID OF PIXEL 1, THIS FUNCTION TELLS WHICH SUB GRID THE X,Y BELONG

        :param x:
        :param y:
        :return:
        """
        return (y - self.min_y) * self.width + (x - self.min_x)

    def collision_check(self, x, y):
        """
        check if the given x, y lie in the obstacle zone
        :param y:
        :param x:
        :return:
        """
        return self.obstacle[x][y]

    def add_obstacle(
        self, obstacle_data_frame: GeoDataFrame, extend_boundary_pixel: int
    ):
        """
        Add obstacle to the grid

        :param obstacle_data_frame: obstacle data read from GeoDataFrame
        :param extend_boundary_pixel: by how many pixel to extend the obstacle boundary
        :return:
        """
        pass

    def extent_check(self, bounds: Union[np.ndarray, list, tuple]) -> bool:
        min_x, min_y, max_x, max_y = bounds
        row_start, col_start = pixel_position(min_x, max_y, self.transform)
        row_stop, col_stop = pixel_position(max_x, min_y, self.transform)
        return (self.width > abs(round((row_stop - row_start) / self.resolution))
                and self.height > abs(round((col_stop - col_start) / self.resolution)))


class PixelGrid(Grid):
    """
    Pixel Grid Over the Spatial Coordinates
    """

    def __init__(
        self,
        bounds: tuple,
        resolution: int,
        dimension: tuple,
        transform: Affine,
    ):
        super().__init__(bounds, resolution, dimension, transform)

    @classmethod
    def pixel_grid(cls, resolution: int, grid_bounds: Union[np.ndarray, list, tuple]):
        """
        Generate a Pixel grid

        :param grid_bounds: the region in which the navigation has to be done
        should be large enough to cover all possible path,
        How to choose bounds - if running on obstacle and trace file, choose the extent such that they
        both fit in the extent

        If running only on the trace file, then  choose the extent such that all possible path to explore can fit

        NOTE - DON'T USE EXTENT OF INDIVIDUAL POLYGON, LINESTRING
        NOTE - DON'T USE EXTENT OF THE FILE IF THE FILE JUST CONTAINS FEW GEOMETRIES CLOSE TO EACH OTHER

        :param resolution:
        :return:
        """

        assert type(grid_bounds) in [np.ndarray, tuple, list], (
            "Expected grid_bounds to be of type Tuple"
            "got %s", (type(grid_bounds), )
        )

        assert type(resolution) is int, (
            "Expected resolution to be of type int"
            "got %s", (type(resolution), )
        )
        min_x, min_y, max_x, max_y = grid_bounds

        grid_transform = generate_affine(min_x, max_y, resolution)

        row_start, col_start = pixel_position(min_x, max_y, grid_transform)
        row_stop, col_stop = pixel_position(max_x, min_y, grid_transform)

        return cls(
            bounds=(row_start, col_start, row_stop, col_stop),
            resolution=resolution,
            dimension=(
                abs(round((row_stop - row_start) / resolution)),
                abs(round((col_stop - col_start) / resolution)),
            ),
            transform=grid_transform,
        )

    def add_obstacle(self, obstacle_data_frame: GeoDataFrame, extend_boundary_pixel=2):
        """
        Add obstacle to the grid

        :param obstacle_data_frame: obstacle data read from GeoDataFrame
        :param extend_boundary_pixel: by how many pixel to extend the obstacle boundary
        :return:
        """

        assert supported_crs(obstacle_data_frame), (
            "Supported CRS ['epsg:26910', 'epsg:32649']"
            "got %s", (obstacle_data_frame.crs,)
        )

        assert geom_check(
            obstacle_data_frame, "Polygon"
        ), "Expected all geometries in to be Polygon"

        assert self.extent_check(obstacle_data_frame.total_bounds), (
            "Expected the bounds of obstacle to fit in the Grid but the obstacle Bounds are Much Bigger"
            " than the grid bounds"
        )
        width, height = self.dimension
        extent = np.zeros([height, width], np.uint8)

        for idx, feature in obstacle_data_frame.iterrows():
            feature_geometry, feature_property = decompose_data_frame_row(feature)

            coordinates = feature_geometry["coordinates"]
            for sub_coordinate in coordinates:
                boundary_coordinates = mapping(Polygon(sub_coordinate).boundary)[
                    "coordinates"
                ]
                corner_points = [
                    pixel_position(bo[0], bo[1], self.transform)
                    for bo in boundary_coordinates
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

        self.obstacle = extent.T >= 255
