from collections import OrderedDict, namedtuple
from types import SimpleNamespace

import networkx as nx
import rtree
from geopandas import GeoDataFrame
from shapely.geometry import shape, Polygon

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
        self._entry = namedtuple("Road", ["fid", "property", "geometry", "weight"])

    def add(
        self,
        feature_id: int,
        feature_property: dict,
        feature_geometry: dict,
        weight: float,
    ):
        assert None not in {
            feature_id,
            feature_property,
            feature_geometry,
            weight,
        }, "Expected ['feature_id', 'feature_property', 'feature_geometry', 'weight'] to be not None"

        assert type(feature_id) is int, (
            "Expected 'feature_id' type to be 'int'," "got %s",
            (type(feature_id),),
        )

        assert type(feature_property) is dict, (
            "Expected 'feature_property' type to be 'dict'," "got %s",
            (type(feature_property),),
        )

        assert type(feature_geometry) is dict, (
            "Expected 'feature_geometry' type to be 'dict'," "got %s",
            (type(feature_geometry),),
        )

        assert type(weight) is float, (
            "Expected 'weight' type to be 'float'," "got %s",
            (type(weight),),
        )

        self[feature_id] = self._entry(
            fid=feature_id,
            property=SimpleNamespace(**feature_property),
            geometry=feature_geometry,
            weight=weight,
        )


class RoadNetwork:
    """
    INFORMATION ASSOCIATED TO THE ROAD GEOMETRY WHICH IS TO BE USED AS BASE FOR MATCHING
    """

    def __init__(self, road_table, maximum_distance):
        self._road_table = road_table
        self.maximum_distance = maximum_distance

        self.tree = self._generate_tree()
        self.graph = self._generate_graph()

    def intersection(self, geometry: Polygon):
        """
        GET ALL THE INTERSECTING POLYGON IN THE EXTENT OF THE GEOMETRY
        :param geometry: a shapely object
        :return:
        """
        assert type(geometry) == Polygon, (
            "Expected 'geometry' type to be 'Polygon'" "got %s",
            (type(geometry),),
        )
        intersection_info = dict()
        intersecting_fid = list(self.tree.intersection(geometry.bounds))
        for fid in intersecting_fid:
            entry = self._road_table[fid]
            road = shape(entry.geometry)
            if road.intersects(geometry):
                intersection_info[fid] = road
        return intersection_info

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

    def _generate_graph(self):
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

            road_graph.add_edges_from(
                [
                    (entry.property.u, entry.property.v),
                ],
                intermediate_nodes=intermediate_nodes,
                fid=entry.fid,
                weight=entry.weight,
            )
        return road_graph

    def get_intermediate_nodes(self, start_node, end_node):
        return self.graph[start_node][end_node]["intermediate_nodes"]

    def get_fid(self, start_node, end_node):
        return self.graph[start_node][end_node]["fid"]

    def get_geometry(self, start_node, end_node):
        return self.geometry(self.get_fid(start_node, end_node))

    def get_graph_data(self, start_node, end_node):
        return self.graph[start_node][end_node]

    def _generate_tree(self):
        """
        Generate RTree for the road network
        :return:
        """
        index = rtree.index.Index()
        for _, entry in self._road_table.items():
            index.insert(entry.fid, shape(entry.geometry).bounds)
        return index


def road_network_from_data_frame(road_data: GeoDataFrame) -> RoadNetwork:
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

        assert "u" in feature_property and "v" in feature_property, (
            "Expected 'u' and 'v' to be present in the property"
            "indicating the start node and the end node of the provided"
            "geometry"
        )
        if "fid" in feature_property:
            idx = feature_property["fid"]
        road_table.add(
            idx, feature_property, feature_geometry, shape(feature_geometry).length
        )

    return RoadNetwork(
        road_table=road_table,
        maximum_distance=compute_diagonal_distance_of_extent(road_data),
    )


def road_network_from_path(path: str) -> RoadNetwork:
    """
    Generate the connected road network from GeoSpatial LineString
    :param path:
    :return:
    """

    return road_network_from_data_frame(read_data_frame(path))
