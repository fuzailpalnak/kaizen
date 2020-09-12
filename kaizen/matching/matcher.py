import math
import operator
from collections import defaultdict, namedtuple, OrderedDict
from types import SimpleNamespace

import fiona
import networkx as nx
from shapely.geometry import shape, Point, Polygon, mapping
from shapely.strtree import STRtree

from scipy import spatial
from scipy import stats

from kaizen.matching.data_format import (
    CandidateCollection,
    TraceCollection,
    CandidateData,
    RoadData,
    Transition,
)


class Road:
    def __init__(self, road_table):
        self._road_table = road_table

        self.tree = self.generate_tree()
        self.graph = self.generate_graph()

    def intersection(self, geometry: Polygon):
        """

        :param geometry:
        :return:
        """
        return [road for road in self.tree.query(geometry) if road.intersects(geometry)]

    def geometry(self, fid):
        """

        :param fid:
        :return:
        """
        return self._road_table[fid].geometry

    def entry(self, fid):
        """

        :param fid:
        :return:
        """
        return self._road_table[fid]

    def generate_graph(self):
        """

        :return:
        """
        road_graph = nx.DiGraph()
        for _, entry in self._road_table.items():
            intermediate_nodes = list()
            geometry = entry.geometry
            line_string_coordinate = geometry["coordinates"]
            for coordinates in line_string_coordinate[1:-1]:
                intermediate_nodes.append((coordinates[0], coordinates[1]))

            road_graph.add_edge(
                u_of_edge=line_string_coordinate[0],
                v_of_edge=line_string_coordinate[-1],
                intermediate_nodes=intermediate_nodes,
                fid=entry.fid,
                weight=shape(geometry).length,
            )
        return road_graph

    def generate_tree(self):
        """

        :return:
        """
        road_geometry = list()
        for _, entry in self._road_table.items():
            road_geometry.append(shape(entry.geometry))
        return STRtree(road_geometry)


class TransitionTable:
    def __init__(self):
        self._transition_table = Transition()

    def add_entry(
        self,
        previous_candidate: namedtuple,
        current_candidate: namedtuple,
        shortest_path,
        shortest_distance,
        shortest_road_id,
        probability,
    ):
        """

        :param previous_candidate:
        :param current_candidate:
        :param shortest_path:
        :param shortest_distance:
        :param shortest_road_id:
        :param probability:
        :return:
        """
        assert (
            previous_candidate is not None and current_candidate is not None
        ), "Expected 'previous_candidate' and 'current_candidate' to be not None"

        self._transition_table.add(
            previous_candidate=previous_candidate,
            current_candidate=current_candidate,
            shortest_path=shortest_path,
            shortest_distance=shortest_distance,
            shortest_road_id=shortest_road_id,
            probability=probability,
        )

    def table(self):
        return self._transition_table

    def entry(self, previous_candidate: namedtuple, current_candidate: namedtuple):
        return self._transition_table[
            self._transition_table.generate_unique_key(
                previous_candidate, current_candidate
            )
        ]


class Match:
    def __init__(self, road: Road, trace_store: TraceCollection):
        self._road = road
        self._trace_store = trace_store

        self._transition_table = TransitionTable()

        # TODO effienct way to decide self._max_distance
        self._max_distance = 5000

    def _get_candidates(self, trace_point: namedtuple):
        """

        :param trace_point:
        :return:
        """
        tr_point = Point(trace_point.x, trace_point.y)
        candidate_roads = self._road.intersection(tr_point.buffer(30))

        candidate_points = CandidateData()
        for idx, candidate in enumerate(candidate_roads):
            # [REFERENCE IN PAPER]
            # Definition 6 (Line Segment Projection): The line segment projection of a point 𝑝 to a road segment
            # 𝑒 is the point 𝑐 on 𝑒 such that 𝑐 = arg 𝑚𝑖𝑛∀ 𝑐𝑖∈𝑒 𝑑𝑖𝑠𝑡(𝑐𝑖, 𝑝) , where 𝑑𝑖𝑠𝑡(𝑐𝑖, 𝑝) returns the distance
            # between p and any point ci on 𝑒.

            # PROJECT THE POINT ON THE ALL THE ROAD SEGMENT THAT LIE IN THE BUFFER ZONE OF - 30 AND GET THE
            # POINT ON THE LINE WITH SHORTEST DISTANCE TO THE TRACE_REC

            # https://stackoverflow.com/questions/24415806/coordinates-of-the-closest-points-of-two-geometries-in-shapely
            fraction = candidate.project(tr_point, normalized=True)
            project_point = candidate.interpolate(fraction, normalized=True)

            attr = self._road.graph[mapping(candidate)["coordinates"][0]][
                mapping(candidate)["coordinates"][-1]
            ]

            # https://gist.github.com/href/1319371
            # https://stackoverflow.com/questions/35282222/in-python-how-do-i-cast-a-class-object-to-a-dict/35282286
            candidate_points.add(
                candidate_id=str(trace_point.trace_point_id) + "_" + str(idx),
                x=project_point.x,
                y=project_point.y,
                distance=fraction,
                road_information={
                    **attr,
                    **{
                        "start": mapping(candidate)["coordinates"][0],
                        "end": mapping(candidate)["coordinates"][-1],
                    },
                },
                trace_information=trace_point,
            )

        return candidate_points

    def _add_intermediate_edge(self, previous_layer_candidate, current_layer_candidate):
        """

        :param previous_layer_candidate:
        :param current_layer_candidate:
        :return:
        """

        previous_candidate_road_projected_point = (
            previous_layer_candidate.x,
            previous_layer_candidate.y,
        )
        current_candidate_road_projected_point = (
            current_layer_candidate.x,
            current_layer_candidate.y,
        )

        if (
            previous_layer_candidate.distance != 0
            and previous_layer_candidate.distance != 1
        ):
            self._road.graph.add_edge(
                previous_layer_candidate.road.start,
                previous_candidate_road_projected_point,
                weight=Point(previous_layer_candidate.road.start).distance(
                    Point(previous_candidate_road_projected_point)
                ),
                fid=previous_layer_candidate.road.fid,
            )
            self._road.graph.add_edge(
                previous_candidate_road_projected_point,
                previous_layer_candidate.road.end,
                weight=Point(previous_layer_candidate.road.end).distance(
                    Point(previous_candidate_road_projected_point)
                ),
                fid=previous_layer_candidate.road.fid,
            )

        if (
            current_layer_candidate.distance != 0
            and current_layer_candidate.distance != 1
        ):
            self._road.graph.add_edge(
                current_layer_candidate.road.start,
                current_candidate_road_projected_point,
                weight=Point(current_layer_candidate.road.start).distance(
                    Point(current_candidate_road_projected_point)
                ),
                fid=current_layer_candidate.road.fid,
            )
            self._road.graph.add_edge(
                current_candidate_road_projected_point,
                current_layer_candidate.road.end,
                weight=Point(current_layer_candidate.road.end).distance(
                    Point(current_candidate_road_projected_point)
                ),
                fid=current_layer_candidate.road.fid,
            )

    def _remove_intermediate_edge(
        self, previous_layer_candidate, current_layer_candidate
    ):
        """

        :param previous_layer_candidate:
        :param current_layer_candidate:
        :return:
        """

        previous_candidate_road_projected_point = (
            previous_layer_candidate.x,
            previous_layer_candidate.y,
        )
        current_candidate_road_projected_point = (
            current_layer_candidate.x,
            current_layer_candidate.y,
        )
        if (
            previous_layer_candidate.distance != 0
            and previous_layer_candidate.distance != 1
        ):
            self._road.graph.remove_edge(
                previous_layer_candidate.road.start,
                previous_candidate_road_projected_point,
            )
            self._road.graph.remove_edge(
                previous_candidate_road_projected_point,
                previous_layer_candidate.road.end,
            )

        if (
            current_layer_candidate.distance != 0
            and current_layer_candidate.distance != 1
        ):
            self._road.graph.remove_edge(
                current_layer_candidate.road.start,
                current_candidate_road_projected_point,
            )
            self._road.graph.remove_edge(
                current_candidate_road_projected_point, current_layer_candidate.road.end
            )

    def _road_ids_along_shortest_path(self, shortest_path):
        """

        :param shortest_path:
        :return:
        """
        road_ids = list()
        for previous, current in zip(shortest_path, shortest_path[1:]):
            fid = self._road.graph[previous][current]["fid"]
            if fid not in road_ids:
                road_ids.append(fid)
        return road_ids

    def _path_information(self, previous_layer_candidate, current_layer_candidate):
        """

        :param previous_layer_candidate:
        :param current_layer_candidate:
        :return:
        """

        previous_candidate_road_projected_point = (
            previous_layer_candidate.x,
            previous_layer_candidate.y,
        )
        current_candidate_road_projected_point = (
            current_layer_candidate.x,
            current_layer_candidate.y,
        )

        if previous_layer_candidate.road.fid == current_layer_candidate.road.fid:
            if current_layer_candidate.distance <= previous_layer_candidate.distance:
                # TODO CHANGE self._max_distance
                shortest_distance = self._max_distance
                shortest_path = None
                road_ids_along_shortest_path = None
            else:
                shortest_distance = Point(
                    previous_candidate_road_projected_point
                ).distance(Point(current_candidate_road_projected_point))
                shortest_path = [
                    previous_layer_candidate.road.start,
                    previous_layer_candidate.road.end,
                ]
                road_ids_along_shortest_path = [previous_layer_candidate.road.fid]

        else:
            self._add_intermediate_edge(
                previous_layer_candidate, current_layer_candidate
            )

            shortest_distance = nx.astar_path_length(
                self._road.graph,
                previous_candidate_road_projected_point,
                current_candidate_road_projected_point,
            )

            shortest_path = nx.astar_path(
                self._road.graph,
                previous_candidate_road_projected_point,
                current_candidate_road_projected_point,
            )

            road_ids_along_shortest_path = self._road_ids_along_shortest_path(
                shortest_path
            )

            self._remove_intermediate_edge(
                previous_layer_candidate, current_layer_candidate
            )

        return shortest_distance, shortest_path, road_ids_along_shortest_path

    @staticmethod
    def _observation_probability(x, y, trace_point):
        """

        :param x:
        :param y:
        :param trace_point:
        :return:
        """

        # [REFERENCE IN PAPER]
        # Definition 7 (Observation Probability): The observation probability is defined as the likelihood
        # that a GPS sampling point 𝑝𝑖 matches a candidate point computed based on the distance between the two
        # points 𝑑𝑖𝑠𝑡(𝑐𝑖, 𝑝𝑖)
        # we use a zero-mean normal distribution with a standard deviation of 20 meters based on empirical evaluation.

        # COMPUTE THE EUCLIDEAN DISTANCE BETWEEN THE CANDIDATE AND TRACE_REC
        return stats.norm.pdf(
            spatial.distance.euclidean([trace_point.x, trace_point.y], [x, y]),
            loc=0,
            scale=30,
        )

    def _transmission_probability(
        self, previous_layer_candidate, current_layer_candidate
    ):
        """
        [REFERENCE IN PAPER]
        SECTION 5.2 - Spatial Analysis
         𝑑𝑖−1→𝑖 = 𝑑𝑖𝑠𝑡(𝑝𝑖, 𝑝𝑖−1) is the Euclidean distance between 𝑝𝑖 and 𝑝𝑖−1 , and 𝑤 𝑖−1,𝑡 →(𝑖,𝑠)
         is the length of shortest path from 𝑐𝑖−1  to 𝑐𝑖

        :param previous_layer_candidate:
        :param current_layer_candidate:
        :return:
        """

        entry = self._transition_table.entry(
            previous_layer_candidate, current_layer_candidate
        )

        return entry.probability

    def _transition_data(self, previous_layer_candidate, current_layer_candidate):

        (
            shortest_distance,
            shortest_path,
            road_ids_along_shortest_path,
        ) = self._path_information(previous_layer_candidate, current_layer_candidate)

        euclidean_distance = spatial.distance.euclidean(
            [
                previous_layer_candidate.trace.x,
                previous_layer_candidate.trace.y,
            ],
            [
                current_layer_candidate.trace.x,
                current_layer_candidate.trace.y,
            ],
        )

        self._transition_table.add_entry(
            previous_layer_candidate,
            current_layer_candidate,
            shortest_path=shortest_path,
            shortest_distance=shortest_distance,
            shortest_road_id=road_ids_along_shortest_path,
            probability=0.99
            if shortest_distance == 0
            else euclidean_distance / shortest_distance,
        )

    def _construct_graph(self, candidates_per_trace: CandidateCollection):
        """
        CANDIDATE POINTS FOR EVERY TRACE_REC FORM A CONNECTION WITH ITS SUBSEQUENT CANDIDATE POINTS
        CONSIDER A TRACE HAS TWO TRACE_REC [1, 2]
        TRACE_REC 1 HAS - 2  AND CANDIDATE POINTS TRACE_REC 2 HAS - 3 CANDIDATE POINTS
        [GENERATED FROM get_candidates FUNCTION CALL]
        GRAPH CONSTRUCTED -

        [TRACE RECORD 1]
        TRACE_REC_1_CANDIDATE_POINT_1 ---|--t1--|---> going to [t2_c1] --|
                                         |--t2--|---> going to [t2_c2] --|
                                         |--t3--|---> going to [t2_c3] --|            [TRACE RECORD 2]
                                                                         |      _________________________________
        t{} = transition_probability                                     |-->   |  TRACE_REC_2_CANDIDATE_POINT_1 |
                                                                                |  TRACE_REC_2_CANDIDATE_POINT_2 |
                                                                         |-->   |  TRACE_REC_2_CANDIDATE_POINT_3 |
                                                                         |      ----------------------------------
        TRACE_REC_1_CANDIDATE_POINT_2 ---|--t4--|---> going to [t2_c1] --|
                                         |--t5--|---> going to [t2_c2] --|
                                         |--t6--|---> going to [t2_c3] --|

        :param candidates_per_trace: candidates belonging to each trace_rec_uuid
        :return:
        """

        graph = nx.Graph()

        previous_layer_collection = dict()

        for uuid, candidates in candidates_per_trace.items():
            # GET CLOSET CANDIDATE POINTS FOR EVERY TRACE_POINT IN A SINGLE TRACE
            assert len(candidates) > 0

            current_layer_collection = dict()

            for idx, current_layer_candidate in enumerate(candidates):
                current_node_id = current_layer_candidate.candidate_id
                current_layer_collection[current_node_id] = current_layer_candidate

                graph.add_node(
                    current_node_id,
                    observation_probability=self._observation_probability(
                        current_layer_candidate.x,
                        current_layer_candidate.y,
                        current_layer_candidate.trace,
                    ),
                )
                if len(previous_layer_collection) == 0:
                    continue
                else:

                    for (
                        previous_node_id,
                        previous_layer_candidate,
                    ) in previous_layer_collection.items():
                        self._transition_data(
                            previous_layer_candidate, current_layer_candidate
                        )

                        graph.add_edge(
                            previous_node_id,
                            current_node_id,
                            transmission_probability=self._transmission_probability(
                                previous_layer_candidate, current_layer_candidate
                            ),
                        )

            previous_layer_collection = current_layer_collection
        return graph

    @staticmethod
    def _find_matched_sequence(graph, candidates_per_trace):
        highest_score_computed = dict()
        parent_of_the_current_candidate = dict()
        to_explore_uuid = list(candidates_per_trace.keys())
        previous_uuid = None

        # STORE THE VALUES OF ALL THE CANDIDATES OF THE FIRST TRACE POINT
        for current_uuid in to_explore_uuid[0:1]:
            for idx, candidate in enumerate(candidates_per_trace[current_uuid]):
                max_node_id = str(current_uuid) + "_" + str(idx)
                highest_score_computed[max_node_id] = graph.nodes[max_node_id][
                    "observation_probability"
                ]
            previous_uuid = current_uuid

        # LOOP OVER THE REMAINING TRACE POINTS [2, N]
        for current_uuid in to_explore_uuid[1:]:
            my_candidates = candidates_per_trace[current_uuid]

            # LOOP OVER EACH CANDIDATE OF THE TRACE POINT
            for candidate in my_candidates:
                maximum = -math.inf
                current_node_id = candidate.candidate_id

                # LOOP OVER THE CANDIDATES OF THE PREDECESSOR TRACE POINT
                for previous_candidates in candidates_per_trace[previous_uuid]:
                    # alt = highest_score_computed[previous_candidate] +
                    # transmission[previous_candidate, current_candidate] * observation[current_candidate]

                    previous_node_id = previous_candidates.candidate_id
                    alt = (
                        graph[previous_node_id][current_node_id][
                            "transmission_probability"
                        ]
                        * graph.nodes[current_node_id]["observation_probability"]
                        + highest_score_computed[previous_node_id]
                    )
                    if alt > maximum:
                        maximum = alt
                        parent_of_the_current_candidate[
                            current_node_id
                        ] = previous_node_id
                    highest_score_computed[current_node_id] = maximum
            previous_uuid = current_uuid

        # https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
        max_node_id = max(highest_score_computed.items(), key=operator.itemgetter(1))[0]

        r_list = list()
        for _ in range(len(to_explore_uuid[1:])):
            r_list.append(max_node_id)
            max_node_id = parent_of_the_current_candidate[max_node_id]

        r_list.append(max_node_id)
        r_list.reverse()

        matched_sequence = [
            candidates_per_trace[int(node_id.split("_")[0])][int(node_id.split("_")[1])]
            for node_id in r_list
        ]

        return matched_sequence

    def _until_match(self, candidates_per_trace: CandidateCollection):
        """
        candidates_per_trace contains collection of candidates for each trace rec present in the trace
        candidates_per_trace = {trace_rec_uuid_1: [candidates for trace_rec_uuid_1],
        trace_rec_uuid_2: [candidates for trace_rec_uuid], ...., trace_rec_uuid_n: [candidates for trace_rec_uuid]} ,
        one trace_rec_uuid can have more than one candidates, every trace_rec_uuid_{} belong to same 'trace_id'

        :param candidates_per_trace: candidates belonging to each trace_rec_uuid
        :return:
        """
        graph = self._construct_graph(
            candidates_per_trace=candidates_per_trace,
        )
        matched_sequence = self._find_matched_sequence(graph, candidates_per_trace)
        return matched_sequence

    def _get_connected_road_geometry(self, matched_sequence):
        connected_road_path = list()
        for previous, current in zip(matched_sequence, matched_sequence[1:]):
            entry = self._transition_table.entry(previous, current)
            for road in entry.shortest_road_id:
                if int(road) not in connected_road_path:
                    connected_road_path.append(int(road))

        return connected_road_path

    def match(self):

        for trace_id, trace in self._trace_store.items():
            candidates_per_trace = CandidateCollection()

            for trace_point in trace:
                # [REFERENCE IN PAPER]
                # SECTION 5.1 Candidate Preparation
                # FOR EVERY TRACE POINT, PROJECT THE POINT ON TO ROAD SEGMENTS WITHIN CERTAIN BUFFER AND NOTE THE
                # CANDIDATES
                # EACH TRACE_REC CAN HAVE MULTIPLE CANDIDATES

                # TODO Handle when no candidate found
                candidates = self._get_candidates(trace_point)
                candidates_per_trace[trace_point.trace_point_id] = candidates

            # FIND A MATCH FOR A SINGLE TRACE
            matched_sequence = self._until_match(candidates_per_trace)
            connected_path = self._get_connected_road_geometry(matched_sequence)
            yield trace_id, connected_path

    @classmethod
    def match_trace_to_road_shape_file(cls, road_shp_pth: str, trace_shp_pth: str):
        """

        :param road_shp_pth:
        :param trace_shp_pth:
        :return:
        """
        road = road_from_shape_file(road_shp_pth)
        trace = trace_from_point_shape_file(trace_shp_pth)
        return cls(road=road, trace_store=trace)

    def match_road_to_road_shape_file(self):
        pass

    def match_trace_to_road_geojson(self):
        pass

    def match_road_to_road_geojson(self):
        pass


def line_geometry(feature: dict) -> (int, dict, dict):
    """
    :param feature:
    :return:
    """
    if (
        "id" not in feature.keys()
        and "geometry" not in feature.keys()
        and "properties" not in feature.keys()
    ):
        raise KeyError("Missing Keys, Must have keys ['id', 'geometry', 'properties']")
    if "type" not in feature["geometry"] and "coordinates" not in feature["geometry"]:
        raise KeyError(
            "Missing Keys, Must have keys ['type'] and ['coordinates'] in feature['geometry']"
        )

    assert feature["geometry"]["type"] in ["LineString"], (
        "Expected geometry to be in ['LineString']",
        "got %s",
        (feature["geometry"]["type"],),
    )

    return (
        int(feature["id"]),
        feature["properties"],
        feature["geometry"],
    )


def road_from_shape_file(shape_file_path: str) -> Road:
    """

    :param shape_file_path:
    :return:
    """
    obj = fiona.open(shape_file_path)

    road_table = RoadData()

    assert obj.schema["geometry"] in ["LineString"], (
        "Expected geometry to be in ['LineString']",
        "got %s",
        (obj.schema["geometry"],),
    )

    for feature in obj:
        assert feature["geometry"]["type"] in ["LineString"], (
            "Expected geometry to be in ['LineString']",
            "got %s",
            (feature["geometry"]["type"],),
        )
        feature_id, feature_property, feature_geometry = line_geometry(feature)

        road_table.add(feature_id, feature_property, feature_geometry)

    return Road(road_table=road_table)


def trace_from_point_shape_file(shape_file_path: str) -> TraceCollection:
    """

    :param shape_file_path:
    :return:
    """
    obj = fiona.open(shape_file_path)

    assert all(
        element in list(obj.schema["properties"].keys())
        for element in [
            "uuid",
            "track_id",
        ]
    ), (
        "Expected keys to be in ShapeFile ['uuid', 'track_id',]",
        "got %s",
        list(obj.schema["properties"].keys()),
    )

    assert obj.schema["geometry"] in ["Point"], (
        "Expected geometry to be in ['Point']",
        "got %s",
        (obj.schema["geometry"],),
    )

    trace_collection = TraceCollection()

    for feature in obj:
        geometry = feature["geometry"]
        x = geometry["coordinates"][0]
        y = geometry["coordinates"][1]
        properties = feature["properties"]
        track_id = properties["track_id"]

        # TODO Assumes every trace record has a unique uuid and a common trace_id
        # TODO RENAME UUID to UNIQUE_ID or ID
        trace_collection.add(
            x=x,
            y=y,
            trace_point_id=properties["uuid"],
            trace_id=track_id,
            **properties,
        )

    return trace_collection
