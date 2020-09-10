import math
from collections import defaultdict, OrderedDict, namedtuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

import fiona
import networkx as nx
from shapely.geometry import shape, Point, Polygon, mapping
from shapely.strtree import STRtree

from scipy import spatial
from scipy import stats


# TODO remove forced parameters and add kwargs
@dataclass
class TracePoint:
    x: Any
    y: Any
    uuid: Any
    track_id: Any
    log_time: Any
    car_id: Any
    v: Any


@dataclass
class Candidate:
    x: Any
    y: Any
    nearest_distance: Any
    road: field(default_factory=namedtuple)
    trace_point: field(default_factory=namedtuple)


class Road:
    def __init__(self, road_tree, road_graph, road_geometry):
        self._tree = road_tree
        self._geometry = road_geometry
        self.graph = road_graph
        self.shortest_path_traversed_history = defaultdict(list)

    def intersection(self, geometry: Polygon):
        return [
            road for road in self._tree.query(geometry) if road.intersects(geometry)
        ]

    def geometry(self, fid):
        return self._geometry[fid]


class TraceStore:
    def __init__(self, store: defaultdict):
        self._store = store

    def __iter__(self):
        for trace_id, trace in self._store.items():
            yield trace_id, trace


class Store:
    pass


class Match:
    def __init__(self, road: Road, trace_store: TraceStore):
        self._road = road
        self._trace_store = trace_store

    def _get_candidates(self, trace_point: TracePoint):
        tr_point = Point(trace_point.x, trace_point.y)
        candidate_roads = self._road.intersection(tr_point.buffer(30))

        candidate_points = list()
        for candidate in candidate_roads:
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
            candidate_points.append(
                Candidate(
                    x=project_point.x,
                    y=project_point.y,
                    nearest_distance=fraction,
                    road=namedtuple("road", attr)(**attr),
                    trace_point=namedtuple("trace", asdict(trace_point))(
                        **asdict(trace_point)
                    ),
                )
            )
        return candidate_points

    def _shortest_path_length(
        self, previous_layer_candidate, current_layer_candidate, path=False
    ):
        # TODO OPTIMIZATION DUPLICATE COMPUTATION HAPPENNING HERE
        if previous_layer_candidate.road.fid == current_layer_candidate.road.fid:
            # TODO HANDLE WHEN CANDIDATE ARE ON THE SAME ROAD
            print("j")

        previous_candidate_road_projected_point = (
            previous_layer_candidate.x,
            previous_layer_candidate.y,
        )
        current_candidate_road_projected_point = (
            current_layer_candidate.x,
            current_layer_candidate.y,
        )

        if (
            previous_layer_candidate.nearest_distance != 0
            and previous_layer_candidate.nearest_distance != 1
        ):
            # TODO RENAME WEIGHT TO DISTANCE
            self._road.graph.add_edge(
                previous_layer_candidate.road.src,
                previous_candidate_road_projected_point,
                weight=Point(previous_layer_candidate.road.src).distance(
                    Point(previous_candidate_road_projected_point)
                ),
                fid=previous_layer_candidate.road.fid,
            )
            self._road.graph.add_edge(
                previous_candidate_road_projected_point,
                previous_layer_candidate.road.tg,
                weight=Point(previous_layer_candidate.road.tg).distance(
                    Point(previous_candidate_road_projected_point)
                ),
                fid=previous_layer_candidate.road.fid,
            )

        if (
            current_layer_candidate.nearest_distance != 0
            and current_layer_candidate.nearest_distance != 1
        ):
            self._road.graph.add_edge(
                current_layer_candidate.road.src,
                current_candidate_road_projected_point,
                weight=Point(current_layer_candidate.road.src).distance(
                    Point(current_candidate_road_projected_point)
                ),
                fid=current_layer_candidate.road.fid,
            )
            self._road.graph.add_edge(
                current_candidate_road_projected_point,
                current_layer_candidate.road.tg,
                weight=Point(current_layer_candidate.road.tg).distance(
                    Point(current_candidate_road_projected_point)
                ),
                fid=current_layer_candidate.road.fid,
            )

        shortest_distance = nx.astar_path_length(
            self._road.graph,
            previous_candidate_road_projected_point,
            current_candidate_road_projected_point,
        )
        if path:
            shortest_path = defaultdict(list)

            traversed_shortest_path = nx.astar_path(
                self._road.graph,
                previous_candidate_road_projected_point,
                current_candidate_road_projected_point,
            )
            for previous, current in zip(
                traversed_shortest_path, traversed_shortest_path[1:]
            ):
                fid = self._road.graph[previous][current]["fid"]
                if len(shortest_path) == 0:
                    shortest_path[tuple(traversed_shortest_path)].append(fid)
                elif fid != shortest_path[tuple(traversed_shortest_path)][-1]:
                    shortest_path[tuple(traversed_shortest_path)].append(fid)
                else:
                    continue
        else:
            shortest_path = None

        if (
            previous_layer_candidate.nearest_distance != 0
            and previous_layer_candidate.nearest_distance != 1
        ):
            self._road.graph.remove_edge(
                previous_layer_candidate.road.src,
                previous_candidate_road_projected_point,
            )
            self._road.graph.remove_edge(
                previous_candidate_road_projected_point,
                previous_layer_candidate.road.tg,
            )

        if (
            current_layer_candidate.nearest_distance != 0
            and current_layer_candidate.nearest_distance != 1
        ):
            self._road.graph.remove_edge(
                current_layer_candidate.road.src, current_candidate_road_projected_point
            )
            self._road.graph.remove_edge(
                current_candidate_road_projected_point, current_layer_candidate.road.tg
            )
        return shortest_distance, shortest_path

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
        shortest_distance, _ = self._shortest_path_length(
            previous_layer_candidate, current_layer_candidate
        )

        euclidean_distance = spatial.distance.euclidean(
            [
                previous_layer_candidate.trace_point.x,
                previous_layer_candidate.trace_point.y,
            ],
            [
                current_layer_candidate.trace_point.x,
                current_layer_candidate.trace_point.y,
            ],
        )

        if shortest_distance == 0:
            trans_prob = 0.99
        else:
            trans_prob = euclidean_distance / shortest_distance
        print(
            f"previous {previous_layer_candidate.road.fid} current {current_layer_candidate.road.fid} "
            f"prob {trans_prob} distance {shortest_distance}"
        )
        return trans_prob

    def _construct_graph(self, candidates_per_trace: defaultdict):
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
                current_node_id = str(uuid) + "_" + str(idx)
                current_layer_collection[current_node_id] = current_layer_candidate

                graph.add_node(
                    current_node_id,
                    observation_probability=self._observation_probability(
                        current_layer_candidate.x,
                        current_layer_candidate.y,
                        current_layer_candidate.trace_point,
                    ),
                )
                if len(previous_layer_collection) == 0:
                    continue
                else:

                    for (
                        previous_node_id,
                        previous_layer_candidate,
                    ) in previous_layer_collection.items():
                        transmission_probability = self._transmission_probability(
                            previous_layer_candidate, current_layer_candidate
                        )
                        graph.add_edge(
                            previous_node_id,
                            current_node_id,
                            transmission_probability=transmission_probability,
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
            for idx, candidate in enumerate(my_candidates):
                maximum = -math.inf
                current_node_id = str(current_uuid) + "_" + str(idx)

                # LOOP OVER THE CANDIDATES OF THE PREDECESSOR TRACE POINT
                for previous_idx, previous_candidates in enumerate(
                    candidates_per_trace[previous_uuid]
                ):
                    # alt = highest_score_computed[previous_candidate] +
                    # transmission[previous_candidate, current_candidate] * observation[current_candidate]

                    previous_node_id = str(previous_uuid) + "_" + str(previous_idx)
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
        print(highest_score_computed)
        r_list = list()

        max_probability = -math.inf
        max_node_id = None
        for node_id, probability in highest_score_computed.items():
            if probability > max_probability:
                max_node_id = node_id
                max_probability = probability

        for i in range(len(to_explore_uuid[1:])):
            r_list.append(max_node_id)
            max_node_id = parent_of_the_current_candidate[max_node_id]

        r_list.append(max_node_id)
        r_list.reverse()

        matched_sequence = []
        for node_id in r_list:
            uuid, idx = node_id.split("_")
            matched_sequence.append(candidates_per_trace[int(uuid)][int(idx)])

        return matched_sequence

    def _until_match(self, candidates_per_trace: defaultdict):
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
            _, path = self._shortest_path_length(previous, current, path=True)
            for _, fid in path.items():
                for road in fid:
                    if len(connected_road_path) == 0:
                        connected_road_path.append(int(road))
                    elif road != connected_road_path[-1]:
                        connected_road_path.append(int(road))
                    else:
                        continue
        return connected_road_path

    def match(self):

        for trace_id, trace in self._trace_store:
            candidates_per_trace = defaultdict(list)

            for trace_point in trace:
                # [REFERENCE IN PAPER]
                # SECTION 5.1 Candidate Preparation
                # FOR EVERY TRACE POINT, PROJECT THE POINT ON TO ROAD SEGMENTS WITHIN CERTAIN BUFFER AND NOTE THE
                # CANDIDATES
                # EACH TRACE_REC CAN HAVE MULTIPLE CANDIDATES
                candidates = self._get_candidates(trace_point)
                candidates_per_trace[trace_point.uuid] = candidates

            # FIND A MATCH FOR A SINGLE TRACE
            matched_sequence = self._until_match(candidates_per_trace)
            print("length of matched sequence {}".format(len(matched_sequence)))
            connected_path = self._get_connected_road_geometry(matched_sequence)
            yield trace_id, connected_path


def line_geometry(feature: dict):
    """
    :param feature:
    :return:
    """
    if "id" not in feature.keys() and "geometry" not in feature.keys():
        raise KeyError("Missing Keys, Must have keys ['id'] and ['geometry']")
    if "type" not in feature["geometry"] and "coordinates" not in feature["geometry"]:
        raise KeyError(
            "Missing Keys, Must have keys ['type'] and ['coordinates'] in feature['geometry']"
        )

    assert feature["geometry"]["type"] in ["LineString"], (
        "Expected geometry to be in ['LineString']",
        "got %s",
        (feature["geometry"]["type"],),
    )

    feature_id = int(feature["id"])
    geometry = shape(feature["geometry"])

    return feature_id, geometry


def road_from_shape_file(shape_file_path: str):
    """

    :param shape_file_path:
    :return:
    """
    obj = fiona.open(shape_file_path)

    road_graph = nx.DiGraph()
    road_geometry = dict()

    assert obj.schema["geometry"] in ["LineString"], (
        "Expected geometry to be in ['LineString']",
        "got %s",
        (obj.schema["geometry"],),
    )

    road_geom = list()
    for feature in obj:
        assert feature["geometry"]["type"] in ["LineString"], (
            "Expected geometry to be in ['LineString']",
            "got %s",
            (feature["geometry"]["type"],),
        )
        intermediate_nodes = list()
        feature_id, feature_geometry = line_geometry(feature)

        road_geom.append(feature_geometry)
        road_geometry[feature_id] = feature["geometry"]

        line_string_coordinate = feature["geometry"]["coordinates"]
        for coordinates in line_string_coordinate[1:-1]:
            intermediate_nodes.append((coordinates[0], coordinates[1]))

        # TODO weight=feature_geometry.length,
        # TODO fishy
        # TODO add hash
        road_graph.add_edge(
            u_of_edge=line_string_coordinate[0],
            v_of_edge=line_string_coordinate[-1],
            intermediate_nodes=intermediate_nodes,
            fid=feature_id,
            src=line_string_coordinate[0],
            tg=line_string_coordinate[-1],
            **feature["properties"],
        )

    return Road(
        road_tree=STRtree(road_geom), road_graph=road_graph, road_geometry=road_geometry
    )


def trace_from_point_shape_file(shape_file_path: str):
    """

    :param shape_file_path:
    :return:
    """
    obj = fiona.open(shape_file_path)

    assert all(
        element in list(obj.schema["properties"].keys())
        for element in ["uuid", "track_id", "log_time", "car_id", "v"]
    ), (
        "Expected keys to be in ShapeFile ['uuid', 'track_id', 'log_time', 'car_id', 'v']",
        "got %s",
        list(obj.schema["properties"].keys()),
    )

    assert obj.schema["geometry"] in ["Point"], (
        "Expected geometry to be in ['Point']",
        "got %s",
        (obj.schema["geometry"],),
    )

    trace_collection = defaultdict(list)

    for feature in obj:
        geometry = feature["geometry"]
        x = geometry["coordinates"][0]
        y = geometry["coordinates"][1]
        properties = feature["properties"]
        track_id = properties["track_id"]

        # TODO Assumes every trace record has a unique uuid and a common trace_id
        # TODO RENAME UUID to UNIQUE_ID or ID
        trace_collection[track_id].append(
            TracePoint(
                x,
                y,
                properties["uuid"],
                track_id,
                datetime.strptime(properties["log_time"], "%Y-%m-%d %H:%M:%S"),
                properties["car_id"],
                properties["v"],
            )
        )
    return TraceStore(trace_collection)


d = fiona.open(r"D:\GIS\map-match\shp\input\connected_road.shp")
ma = Match(
    road_from_shape_file(r"D:\GIS\map-match\shp\input\connected_road.shp"),
    trace_from_point_shape_file(r"D:\GIS\map-match\shp\input\track.shp"),
)

from collections import OrderedDict


crs = {"init": "epsg:32649"}
driver = "ESRI Shapefile"

schema = {"properties": OrderedDict([("idx", "int")]), "geometry": "LineString"}
for trace_id1, connected_path1 in ma.match():

    import fiona

    out_c = fiona.open(
        r"D:\Cypherics\Library\kaizen\shp\new_path{}.shp".format(trace_id1),
        "w",
        driver=driver,
        crs=crs,
        schema=schema,
    )
    for i, road_id in enumerate(connected_path1):
        rec = {
            "type": "Feature",
            "id": "-1",
            "geometry": ma._road.geometry(int(road_id)),
            "properties": OrderedDict([("idx", i)]),
        }
        out_c.write(rec)
    out_c.close()
