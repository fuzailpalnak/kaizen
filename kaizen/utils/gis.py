from affine import Affine
from geopandas import GeoDataFrame
from pandas import Series
from rasterio.transform import rowcol, xy
from shapely.geometry import mapping, box


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
