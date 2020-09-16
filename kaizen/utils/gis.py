from geopandas import GeoDataFrame
from pandas import Series
from shapely.geometry import mapping


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
    assert geom_type in ["LineString", "Point", "MultiLineString", "MultiPolygon"], (
        "Expected geomtype to be in ['LineString', 'Point', 'MultiLineString', 'MultiPolygon'] to check"
        "got %s",
        (geom_type,),
    )
    return all(data_frame.geom_type.array == geom_type)
