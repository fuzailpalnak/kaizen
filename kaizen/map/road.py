from collections import OrderedDict, namedtuple
from types import SimpleNamespace

import geopandas
import networkx as nx
from shapely.geometry import shape, Polygon
from shapely.strtree import STRtree

from kaizen.utils.gis import (
    geom_check,
    decompose_data_frame_row,
    compute_diagonal_distance_of_extent,
    supported_crs,
    read_data_frame,
)


class RoadTable(OrderedDict):
    """
    STORE ROAD DATA
    """

    def __init__(self):
        super().__init__()
        self._entry = namedtuple("Road", ["fid", "property", "geometry"])

    def add(self, feature_id: int, feature_property: dict, feature_geometry: dict):
        assert (
            feature_id is not None
            and feature_property is not None
            and feature_geometry is not None
        ), "Expected ['feature_id', 'feature_property', 'feature_geometry'] to be not None"

        assert type(feature_id) is int, (
            "Expected type to be 'int'," "got %s",
            (type(feature_id),),
        )

        assert type(feature_property) is dict, (
            "Expected type to be 'dict'," "got %s",
            (type(feature_property),),
        )

        assert type(feature_geometry) is dict, (
            "Expected type to be 'dict'," "got %s",
            (type(feature_geometry),),
        )
        self[feature_id] = self._entry(
            fid=feature_id,
            property=SimpleNamespace(**feature_property),
            geometry=feature_geometry,
        )


class RoadNetwork:
    """
    INFORMATION ASSOCIATED TO THE ROAD GEOMETRY WHICH IS TO BE USED AS BASE FOR MATCHING
    """

    def __init__(self, road_table, maximum_distance):
        self._road_table = road_table
        self.maximum_distance = maximum_distance

        self.tree = self.generate_tree()
        self.graph = self.generate_graph()

    def intersection(self, geometry: Polygon):
        """
        GET ALL THE INTERSECTING POLYGON IN THE EXTENT OF THE GEOMETRY
        :param geometry: a shapely object
        :return:
        """
        assert type(geometry) == Polygon, (
            "Expected type to be 'Polygon'" "got %s",
            (type(geometry),),
        )
        return [road for road in self.tree.query(geometry) if road.intersects(geometry)]

    def geometry(self, fid):
        """
        GET ROAD GEOMETRY FROM THE ROAD TABLE
        :param fid: unique id to identify road element by
        :return:
        """
        return self._road_table[fid].geometry

    def entry(self, fid):
        """
        GET THE ENTIRE INFORMATION ABOUT THE ROAD
        :param fid:
        :return:
        """
        return self._road_table[fid]

    def generate_graph(self):
        """
        Generate DiGraph of the connected road network
        :return:
        """
        # https://stackoverflow.com/questions/30770776/networkx-how-to-create-multidigraph-from-shapefile
        road_graph = nx.DiGraph()
        for _, entry in self._road_table.items():
            intermediate_nodes = list()
            geometry = entry.geometry
            line_string_coordinate = geometry["coordinates"]
            for coordinates in line_string_coordinate[1:-1]:
                intermediate_nodes.append((coordinates[0], coordinates[1]))

            # TODO instead of making assumption that the road are consecutive let teh data of connectivity come from
            # the user
            road_graph.add_edges_from(
                [
                    (line_string_coordinate[0], line_string_coordinate[-1]),
                ],
                intermediate_nodes=intermediate_nodes,
                fid=entry.fid,
                weight=shape(geometry).length,
            )
        return road_graph

    def get_intermediate_nodes(self, start_node, end_node):
        return self.graph[start_node][end_node]["intermediate_nodes"]

    def get_fid(self, start_node, end_node):
        return self.graph[start_node][end_node]["fid"]

    def get_geometry(self, start_node, end_node):
        return self.geometry(self.get_fid(start_node, end_node))

    def generate_tree(self):
        """
        Generate RTree for the road network
        :return:
        """
        road_geometry = list()
        for _, entry in self._road_table.items():
            road_geometry.append(shape(entry.geometry))
        return STRtree(road_geometry)


def road_network(path: str) -> RoadNetwork:
    """
    Generate the connected road network from GeoSpatial LineString
    :param path:
    :return:
    """

    road_data = read_data_frame(path)
    assert supported_crs(road_data), (
        "Supported CRS ['epsg:26910', 'epsg:32649']" "got %s",
        (road_data.crs,),
    )

    road_table = RoadTable()

    assert geom_check(
        road_data, "LineString"
    ), "Expected all geometries in to be LineString"

    for idx, feature in road_data.iterrows():
        feature_geometry, feature_property = decompose_data_frame_row(feature)
        if "fid" in feature_property:
            idx = feature_property["fid"]
        road_table.add(idx, feature_property, feature_geometry)

    return RoadNetwork(
        road_table=road_table,
        maximum_distance=compute_diagonal_distance_of_extent(road_data),
    )
