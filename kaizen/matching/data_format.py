from collections import OrderedDict, namedtuple, defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


class RoadData(OrderedDict):
    def __init__(self):
        super().__init__()
        self._entry = namedtuple("Road", ["fid", "property", "geometry"])

    def add(
        self, feature_id: int, feature_property: OrderedDict, feature_geometry: dict
    ):
        assert (
            feature_id is not None
            and feature_property is not None
            and feature_geometry is not None
        ), "Expected ['feature_id', 'feature_property', 'feature_geometry'] to be not None"

        assert type(feature_id) is int, (
            "Expected type to be 'int'," "got %s",
            (type(feature_id),),
        )

        assert type(feature_property) is OrderedDict, (
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


@dataclass
class Candidate:
    candidate_id: str
    x: float
    y: float
    road: Any
    trace: Any


class CandidateData(list):
    def __init__(self):
        super().__init__()
        self._entry = namedtuple(
            "Candidate", ["candidate_id", "x", "y", "distance", "road", "trace"]
        )

    def add(
        self,
        x,
        y,
        candidate_id: str,
        distance: float,
        road_information: Any,
        trace_information: Any,
    ):
        assert (
            x is not None
            and y is not None
            and distance is not None
            and road_information is not None
            and trace_information is not None
        ), "Expected ['x', 'y', 'candidate_id', 'distance', 'road_information', 'trace_information'] to be not None"

        assert type(distance) is float, (
            "Expected type to be 'float'," "got %s",
            (type(distance),),
        )

        assert all(
            element in road_information for element in ["weight", "start", "end"]
        ), (
            "Expected keys to be in road_information ['weight', 'start', 'end',]",
            "got %s",
            road_information,
        )

        if isinstance(road_information, dict):
            road_information = SimpleNamespace(**road_information)

        if isinstance(trace_information, dict):
            trace_information = SimpleNamespace(**trace_information)

        self.append(
            self._entry(
                candidate_id=candidate_id,
                x=x,
                y=y,
                distance=distance,
                road=road_information,
                trace=trace_information,
            )
        )


class TraceCollection(defaultdict):
    def __init__(self):
        super().__init__(list)
        self._entry = namedtuple(
            "TracePoint", ["x", "y", "trace_point_id", "trace_id", "additional"]
        )

    def add(self, x, y, trace_point_id, trace_id, **kwargs):
        assert (
            x is not None
            and y is not None
            and trace_point_id is not None
            and trace_id is not None
        ), "Expected ['x', 'y', 'trace_point_id', 'trace_id'] to be not None"
        self[trace_id].append(
            self._entry(
                x=x,
                y=y,
                trace_point_id=trace_point_id,
                trace_id=trace_id,
                additional=SimpleNamespace(**kwargs),
            )
        )


class CandidateCollection(defaultdict):
    def __init__(self):
        super().__init__(list)

    def add(self, idx, candidate: CandidateData):
        self[idx] = candidate


class Transition(dict):
    def __init__(self):
        super().__init__()
        self._entry = namedtuple(
            "TransitionEntry",
            [
                "previous",
                "current",
                "shortest_path",
                "shortest_distance",
                "shortest_road_id",
                "probability",
            ],
        )

    @staticmethod
    def generate_unique_key(previous_candidate, current_candidate):
        return hash(
            str(previous_candidate.road.fid)
            + str(previous_candidate.candidate_id)
            + str(current_candidate.road.fid)
            + str(current_candidate.candidate_id)
        )

    def add(
        self,
        previous_candidate: namedtuple,
        current_candidate: namedtuple,
        shortest_path,
        shortest_distance,
        shortest_road_id,
        probability,
    ):
        self[
            self.generate_unique_key(previous_candidate, current_candidate)
        ] = self._entry(
            previous=previous_candidate,
            current=current_candidate,
            shortest_path=shortest_path,
            shortest_distance=shortest_distance,
            shortest_road_id=shortest_road_id,
            probability=probability,
        )
