import math
import operator
from collections import defaultdict, OrderedDict, namedtuple
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Tuple, Union, List

import networkx as nx
from shapely.geometry import Point, mapping, shape, LineString, MultiLineString

from scipy import spatial
from scipy import stats

from kaizen_mapping.map import refresh_print
from kaizen_mapping.map.road import RoadNetwork, road_network_from_path
from kaizen_mapping.map.trace import Traces, TracePoint
from kaizen_mapping.utils.gis import (
    line_referencing,
    line_referencing_series_of_coordinates,
)


@dataclass
class Candidate:
    candidate_id: Any
    x: float
    y: float
    distance: float
    road: namedtuple
    trace: Any


class CandidatesPerTracePoint(list):
    """
    COLLECT THE POTENTIAL CANDIDATE FOUND FOR EVERY TRACE POINT
    """

    def __init__(self):
        super().__init__()

    def add(
        self,
        x,
        y,
        candidate_id: str,
        distance: float,
        road_information: namedtuple,
        trace_information: Any,
    ):
        """
        EVERY FOUND CANDIDATE SHOULD CONTAIN  ITS POSITION, UNIQUE ID, CANDIDATES PERPENDICULAR DISTANCE FROM THE
        NEAREST ROAD, INFORMATION ABOUT THE ROAD ELEMENT TO WHICH IT IS ASSOCIATED AND INFORMATION OF THE
        TRACE POINT IT BELONGS TO

        :param x:
        :param y:
        :param candidate_id: unique id
        :param distance:  line referenced distance to the road
        :param road_information: information of the road element on which the the candidate is projected
        :param trace_information:  information of the trace point for which the candidate was obtained
        :return:
        """
        assert all(
            v is not None for v in [x, y, distance, road_information, trace_information]
        ), "Expected ['x', 'y', 'candidate_id', 'distance', 'road_information', 'trace_information'] to be not None"

        assert type(distance) is float, (
            "Expected type to be 'float'," "got %s",
            (type(distance),),
        )

        assert hasattr(
            road_information, "weight"
        ), "Expected road information to have ['weight']"

        assert hasattr(road_information.property, "u") and hasattr(
            road_information.property, "v"
        ), ("Expected road to have start node 'u' and end node 'v'" "for every edge")

        if isinstance(trace_information, dict):
            trace_information = SimpleNamespace(**trace_information)

        self.append(
            Candidate(
                candidate_id=candidate_id,
                x=x,
                y=y,
                distance=distance,
                road=road_information,
                trace=trace_information,
            )
        )

    def coordinates(self):
        return [(candidate.x, candidate.y) for candidate in self]


class Candidates(defaultdict):
    def __init__(self):
        super().__init__(list)

    def add(self, idx, candidate_per_trace_point: CandidatesPerTracePoint):
        self[idx] = candidate_per_trace_point

    def coordinates(self):
        return [candidate.coordinates() for _, candidate in self.items()]


class Match:
    def __init__(self, road: RoadNetwork, observation_error=30):
        self.road_network = road
        self.observation_error = observation_error

    def _get_candidates(self, trace_point: TracePoint) -> CandidatesPerTracePoint:
        """

        :param trace_point: Observed data point
        :return:
        """
        tr_point = Point(trace_point.x, trace_point.y)

        candidate_roads = self.road_network.intersection(
            tr_point.buffer(self.observation_error)
        )
        candidates_per_trace_points = CandidatesPerTracePoint()
        for idx, (fid, candidate) in enumerate(candidate_roads.items()):

            # [REFERENCE IN PAPER - Map-Matching for Low-Sampling-Rate GPS Trajectories]
            # DEFINITION 6 (LINE SEGMENT PROJECTION): THE LINE SEGMENT PROJECTION OF A POINT 𝑝 TO A ROAD SEGMENT
            # 𝑒 IS THE POINT 𝑐 ON 𝑒 SUCH THAT 𝑐 = ARG 𝑚𝑖𝑛∀ 𝑐𝑖∈𝑒 𝑑𝑖𝑠𝑡(𝑐𝑖, 𝑝) , WHERE 𝑑𝑖𝑠𝑡(𝑐𝑖, 𝑝) RETURNS THE DISTANCE
            # BETWEEN P AND ANY POINT CI ON 𝑒.

            # PROJECT THE POINT ON THE ALL THE ROAD SEGMENT THAT LIE IN THE BUFFER ZONE OF - 30 AND GET THE
            # POINT ON THE LINE WITH SHORTEST DISTANCE TO THE TRACE_REC

            # https://stackoverflow.com/questions/24415806/coordinates-of-the-closest-points-of-two-geometries-in-shapely
            fraction, project_point = line_referencing(candidate, tr_point)

            # https://gist.github.com/href/1319371
            # https://stackoverflow.com/questions/35282222/in-python-how-do-i-cast-a-class-object-to-a-dict/35282286
            candidates_per_trace_points.add(
                candidate_id=str(trace_point.trace_point_id) + "_" + str(idx),
                x=project_point.x,
                y=project_point.y,
                distance=fraction,
                road_information=self.road_network.entry(fid),
                trace_information=trace_point,
            )

        return candidates_per_trace_points

    def _road_ids_along_shortest_path(self, shortest_path: list) -> list:
        """
        GET THE ROAD IDS OF THE SHORTEST TRAVERSED PATH

        :param shortest_path:
        :return:
        """
        road_ids = list()
        for previous, current in zip(shortest_path, shortest_path[1:]):
            fid = self.road_network.get_fid(previous, current)
            if fid not in road_ids:
                road_ids.append(fid)
        return road_ids

    def _path_information(
        self, previous_layer_candidate: Candidate, current_layer_candidate: Candidate
    ) -> float:
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
            if previous_layer_candidate.distance >= current_layer_candidate.distance:
                # IT INDICATES THAT THE VEHICLE LEAVES EDGE E THEN RE-ENTERING THE SAME EDGE E
                shortest_distance = self.road_network.maximum_distance

            elif previous_layer_candidate.distance < current_layer_candidate.distance:
                # IT REPRESENTS THAT THE VEHICLE STAYS ON EDGE E WHEN MOVING FROM TRACE_POINT_1 TO  TRACE_POINT_2

                shortest_distance = Point(
                    previous_candidate_road_projected_point
                ).distance(Point(current_candidate_road_projected_point))
            else:
                raise Exception("Something went horribly Wrong")

        else:
            # CANDIDATES ARE ON DIFFERENT EDGES

            graph_distance = nx.astar_path_length(
                self.road_network.graph,
                previous_layer_candidate.road.property.v,
                current_layer_candidate.road.property.u,
            )

            # https://people.kth.se/~cyang/bib/fmm.pdf [Computation]
            shortest_distance = (
                (
                    previous_layer_candidate.road.weight
                    - previous_layer_candidate.distance
                )
                + graph_distance
                + current_layer_candidate.distance
            )

        return shortest_distance

    def _observation_probability(
        self, x: float, y: float, trace_point: TracePoint
    ) -> float:
        """

        :param x:
        :param y:
        :param trace_point:
        :return:
        """

        # [REFERENCE IN PAPER - Map-Matching for Low-Sampling-Rate GPS Trajectories]
        # DEFINITION 7 (OBSERVATION PROBABILITY): THE OBSERVATION PROBABILITY IS DEFINED AS THE LIKELIHOOD
        # THAT A GPS SAMPLING POINT 𝑝𝑖 MATCHES A CANDIDATE POINT COMPUTED BASED ON THE DISTANCE BETWEEN THE TWO
        # POINTS 𝑑𝑖𝑠𝑡(𝑐𝑖, 𝑝𝑖)
        # WE USE A ZERO-MEAN NORMAL DISTRIBUTION WITH A STANDARD DEVIATION OF 20 METERS BASED ON EMPIRICAL EVALUATION.

        # COMPUTE THE EUCLIDEAN DISTANCE BETWEEN THE CANDIDATE AND TRACE_REC
        return stats.norm.pdf(
            spatial.distance.euclidean([trace_point.x, trace_point.y], [x, y]),
            loc=0,
            scale=self.observation_error,
        )

    def _transmission_probability(
        self, previous_layer_candidate: Candidate, current_layer_candidate: Candidate
    ) -> float:
        """
        [REFERENCE IN PAPER - Map-Matching for Low-Sampling-Rate GPS Trajectories]
        SECTION 5.2 - SPATIAL ANALYSIS
         𝑑𝑖−1→𝑖 = 𝑑𝑖𝑠𝑡(𝑝𝑖, 𝑝𝑖−1) IS THE EUCLIDEAN DISTANCE BETWEEN 𝑝𝑖 AND 𝑝𝑖−1 , AND 𝑤 𝑖−1,𝑡 →(𝑖,𝑠)
         IS THE LENGTH OF SHORTEST PATH FROM 𝑐𝑖−1  TO 𝑐𝑖

        :param previous_layer_candidate:
        :param current_layer_candidate:
        :return:
        """
        shortest_distance = self._path_information(
            previous_layer_candidate, current_layer_candidate
        )

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

        return euclidean_distance / shortest_distance

    def _construct_graph(self, candidates: Candidates) -> nx.DiGraph:
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

        :param candidates: candidates belonging to each trace_rec_uuid
        :return:
        """

        graph = nx.DiGraph()

        previous_layer_collection = dict()

        for _, candidates_per_trace_point in candidates.items():
            # GET CLOSET CANDIDATE POINTS FOR EVERY TRACE_POINT IN A SINGLE TRACE

            current_layer_collection = dict()

            for current_layer_candidate in candidates_per_trace_point:
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
    def _find_matched_sequence(
        graph: nx.DiGraph, candidates: Candidates
    ) -> List[Candidate]:
        """
        FIND THE MATCHED SEQUENCE FROM GIVEN THE TRANSMISSION AND OBSERVATION PROBABILITIES

        :param graph:
        :param candidates:
        :return:
        """
        # TODO LOTS OF FOR LOOP HERE, SEE IF THAT CAN BE OPTIMIZED

        highest_score_computed = dict()
        parent_of_the_current_candidate = dict()
        to_explore_uuid = list(candidates.keys())
        previous_uuid = None

        # STORE THE VALUES OF ALL THE CANDIDATES OF THE FIRST TRACE POINT
        for current_uuid in to_explore_uuid[0:1]:
            for idx, candidate in enumerate(candidates[current_uuid]):
                max_node_id = candidate.candidate_id
                highest_score_computed[max_node_id] = graph.nodes[max_node_id][
                    "observation_probability"
                ]
            previous_uuid = current_uuid

        # LOOP OVER THE REMAINING TRACE POINTS [2, N]
        for current_uuid in to_explore_uuid[1:]:
            my_candidates = candidates[current_uuid]

            # LOOP OVER EACH CANDIDATE OF THE TRACE POINT
            for candidate in my_candidates:
                maximum = -math.inf
                current_node_id = candidate.candidate_id

                # LOOP OVER THE CANDIDATES OF THE PREDECESSOR TRACE POINT
                for previous_candidates in candidates[previous_uuid]:
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
            candidates[int(candidate_id.split("_")[0])][
                int(candidate_id.split("_")[-1])
            ]
            for candidate_id in r_list
        ]

        return matched_sequence

    def _until_match(self, candidates: Candidates) -> List[Candidate]:
        """
        CANDIDATES CONTAINS COLLECTION OF CANDIDATES FOR EACH TRACE REC PRESENT IN THE TRACE
        CANDIDATES = {TRACE_REC_UUID_1: [CANDIDATES FOR TRACE_REC_UUID_1],
        TRACE_REC_UUID_2: [CANDIDATES FOR TRACE_REC_UUID], ...., TRACE_REC_UUID_N: [CANDIDATES FOR TRACE_REC_UUID]} ,
        ONE TRACE_REC_UUID CAN HAVE MORE THAN ONE CANDIDATES, EVERY TRACE_REC_UUID_{} BELONG TO SAME 'TRACE_ID'

        :param candidates: candidates belonging to each trace_rec_uuid
        :return:
        """
        graph = self._construct_graph(
            candidates=candidates,
        )
        matched_sequence = self._find_matched_sequence(graph, candidates)
        return matched_sequence

    def _get_connected_road_geometry(
        self, matched_sequence: List[Candidate]
    ) -> Tuple[List[LineString], Union[defaultdict, OrderedDict]]:

        # TODO IMPROVE FInd CONNECTED GEOMETRY ALGORITHM
        connected_shape = list()
        connected_info = OrderedDict()
        visited = list()

        # TODO ALTERNATIVES TO FOR LOOP
        for previous, current in zip(matched_sequence, matched_sequence[1:]):
            road_ids = self._road_ids_along_shortest_path(
                nx.astar_path(
                    self.road_network.graph,
                    previous.road.property.u,
                    current.road.property.v,
                )
            )
            for road in road_ids:
                if int(road) not in visited:
                    connected_shape.append(shape(self.road_network.geometry(int(road))))
                    connected_info[int(road)] = self.road_network.entry(int(road))
                    visited.append(int(road))

        return connected_shape, connected_info

    def _match(
        self, trace_id, trace: List[TracePoint]
    ) -> Tuple[List[LineString], Union[defaultdict, OrderedDict], List[Point]]:
        candidates = Candidates()

        for iterator, trace_point in enumerate(trace):
            refresh_print(
                f"Map Matcher Progress: TraceID {trace_id}, Traces - {iterator+1}/{len(trace)}"
            )
            # [REFERENCE IN PAPER]
            # SECTION 5.1 Candidate Preparation
            # FOR EVERY TRACE POINT, PROJECT THE POINT ON TO ROAD SEGMENTS WITHIN CERTAIN BUFFER AND NOTE THE
            # CANDIDATES
            # EACH TRACE_REC CAN HAVE MULTIPLE CANDIDATES

            candidates_per_trace_point = self._get_candidates(trace_point)
            if len(candidates_per_trace_point) != 0:
                candidates.add(trace_point.trace_point_id, candidates_per_trace_point)

        # FIND A MATCH FOR A SINGLE TRACE
        matched_sequence = self._until_match(candidates)
        connected_shape, connected_info = self._get_connected_road_geometry(
            matched_sequence
        )
        if len(connected_shape) > 0:
            referenced_poi = line_referencing_series_of_coordinates(
                MultiLineString(connected_shape),
                Traces.trace_point_to_coordinates(trace),
            )
        else:
            referenced_poi = list()

        return (
            connected_shape,
            connected_info,
            referenced_poi,
        )

    def match_trace(
        self, trace_id, trace: List[TracePoint]
    ) -> Tuple[List[LineString], Union[defaultdict, OrderedDict], List[Point]]:
        assert all(
            [isinstance(trace_point, TracePoint) for trace_point in trace]
        ), "Expected all points to be TracePoint, got types %s." % (
            ", ".join([str(type(v)) for v in trace])
        )
        return self._match(trace_id, trace)

    def match_traces(
        self, traces: Traces
    ) -> Tuple[List[LineString], Union[defaultdict, OrderedDict], List[Point]]:
        assert isinstance(traces, Traces), (
            "Expected 'traces' to be instance of 'Traces'" "got %s",
            (type(traces),),
        )
        for trace_id, trace in traces.items():
            yield self._match(trace_id, trace)

    @classmethod
    def init(cls, road_network_file: str):
        """

        :param road_network_file:
        :return:
        """
        road = road_network_from_path(road_network_file)
        return cls(road=road)
