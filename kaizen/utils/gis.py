from typing import Union, Tuple
import numpy as np
import ogr
import osr

import geopandas
import visvalingamwyatt as vw

from affine import Affine
from geopandas import GeoDataFrame
from pandas import Series
from rasterio.transform import rowcol, xy
from shapely.geometry import mapping, box, Point, Polygon, LineString


def decompose_data_frame_row(row: Series):
    if "geometry" not in list(row.keys()):
        raise KeyError("Missing Keys, Must have keys ['geometry']")

    feature_geometry = mapping(row["geometry"])

    feature_property = dict()
    for geometry_property in list(row.keys()):
        if geometry_property not in ["geometry"]:
            feature_property[geometry_property] = row[geometry_property]

    return feature_geometry, feature_property


def geom_check(data_frame: GeoDataFrame, geom_type: str) -> bool:
    assert geom_type in [
        "LineString",
        "Point",
        "MultiLineString",
        "MultiPolygon",
        "Polygon",
    ], (
        "Expected geomtype to be in ['LineString', 'Point', 'MultiLineString', 'MultiPolygon', 'Polygon'] to check"
        "got %s",
        (geom_type,),
    )
    return all(data_frame.geom_type.array == geom_type)


def pixel_position(x: float, y: float, transform: Affine) -> list:
    """
    CONVERT SPATIAL COORDINATES TO PIXEL X AND Y

    :param transform:
    :param x:
    :param y:
    :return:
    """
    return rowcol(transform, x, y)


def spatial_position(x: int, y: int, transform: Affine) -> list:
    """
    CONVERT PIXEL COORDINATE TO SPATIAL COORDINATES

    :param transform:
    :param x:
    :param y:
    :return:
    """
    return xy(transform, x, y)


def generate_affine(minx: float, maxy: float, resolution: float) -> Affine:
    """
    Generate affine transform over the spatial coordinates
    :param minx: min x of the extent
    :param maxy: max y of the extent
    :param resolution: what resolution to use to convert the spatial coordinates in pixel
    :return:
    """
    return Affine.translation(minx, maxy) * Affine.scale(resolution, -resolution)


def total_bounds(data_frame: GeoDataFrame):
    return data_frame.total_bounds


def get_maximum_bound(data_frame_1: GeoDataFrame, data_frame_2: GeoDataFrame):
    return (
        data_frame_1.total_bounds
        if box(*data_frame_1.total_bounds).area > box(*data_frame_2.total_bounds).area
        else data_frame_2.total_bounds
    )


def compute_diagonal_distance_of_extent(data_frame: GeoDataFrame) -> float:
    min_x, min_y, max_x, max_y = total_bounds(data_frame)
    return Point((min_x, min_y)).distance(Point((max_x, max_y)))


def my_crs(crs: str):
    return crs in ["epsg:26910", "epsg:32649"]


def supported_crs(data_frame: GeoDataFrame):
    return data_frame.crs in ["epsg:26910", "epsg:32649"]


def read_data_frame(path: str):
    return geopandas.read_file(path)


def crs_conversion(
    crs_from: str, crs_to: str, coordinate: tuple
) -> Tuple[float, float]:
    # https://gis.stackexchange.com/questions/78838/converting-projected-coordinates-to-lat-lon-using-python

    assert len(coordinate) == 2, (
        "Expected 'point' in format '(X, Y)'" "got %s",
        (coordinate,),
    )

    crs_from = int(crs_from.split(":")[-1])
    crs_to = int(crs_to.split(":")[-1])

    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(coordinate[0], coordinate[1])

    in_spatial_ref = osr.SpatialReference()
    in_spatial_ref.ImportFromEPSG(crs_from)

    out_spatial_ref = osr.SpatialReference()
    out_spatial_ref.ImportFromEPSG(crs_to)

    coordinate_transform = osr.CoordinateTransformation(in_spatial_ref, out_spatial_ref)

    point.Transform(coordinate_transform)
    return point.GetX(), point.GetY()


def bounding_box_crs_conversion(
    bounds: Union[np.ndarray, list, tuple], crs_to: str, crs_from="epsg:4326"
) -> list:

    assert len(bounds) == 1, ("Expected a single bounding box" "got %s", (len(bounds)))
    assert my_crs(crs_to), (
        "CRS Provided not in supported list" "Expected %s got %s",
        (
            ["epsg:26910", "epsg:32649"],
            crs_to,
        ),
    )
    converted_boundary = list()
    for point in bounds[0]:
        converted_boundary.append(
            crs_conversion(crs_from, crs_to, (point[0], point[1]))
        )

    return converted_boundary


def convert_and_get_extent(
    bounds: Union[np.ndarray, list, tuple], crs_to: str, crs_from="epsg:4326"
) -> tuple:

    assert len(bounds) == 1, ("Expected a single bounding box" "got %s", (len(bounds)))
    assert my_crs(crs_to), (
        "CRS Provided not in supported list" "Expected %s got %s",
        (
            ["epsg:26910", "epsg:32649"],
            crs_to,
        ),
    )

    return Polygon(bounding_box_crs_conversion(bounds, crs_to, crs_from)).bounds


def line_simplify(coordinates: list, area_threshold_im_meters: float):
    # https://github.com/Permafacture/Py-Visvalingam-Whyatt/blob/master/polysimplify.py
    # https://pypi.org/project/visvalingamwyatt/
    # https://hull-repository.worktribe.com/preview/376364/000870493786962263.pdf
    return vw.simplify(coordinates, threshold=area_threshold_im_meters)
