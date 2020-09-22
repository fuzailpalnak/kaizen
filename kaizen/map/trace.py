import uuid
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, List

from geopandas import GeoDataFrame

from kaizen.utils.gis import (
    geom_check,
    decompose_data_frame_row,
    supported_crs,
    read_data_frame,
)


@dataclass
class TracePoint:
    x: Any
    y: Any
    trace_point_id: Any
    trace_id: Any
    additional: field(default_factory=namedtuple)


class Traces(defaultdict):
    """
    COLLECTION OF THE POINTS
    """

    def __init__(self):
        super().__init__(list)

    @staticmethod
    def trace_point_to_coordinates(trace: List[TracePoint]) -> List[tuple]:
        return [(trace_point.x, trace_point.y) for trace_point in trace]

    def coordinates(self) -> List[List]:
        trace_coordinates = list()
        for idx, trace in self.items():
            trace_coordinates.append(self.trace_point_to_coordinates(trace))
        return trace_coordinates

    def add(self, x, y, trace_point_id, trace_id, **kwargs):
        """
        EVERY TRACE MUST HAVE A UNIQUE TRACE_ID AND THE POINTS IN THE TRACE [TRACE POINTS] MUST HAVE A UNIQUE ID OF
        THEIR OWN

        {trace_1_id: [trace_point_1_id, trace_point_2_id]}

        :param x:
        :param y:
        :param trace_point_id: unique id
        :param trace_id: which trace does the trace point belong to
        :param kwargs:
        :return:
        """
        assert all(
            v is not None for v in [x, y, trace_point_id, trace_id]
        ), "Expected ['x', 'y', 'trace_point_id', 'trace_id'] to be not None"

        self[trace_id].append(
            TracePoint(x, y, trace_point_id, trace_id, SimpleNamespace(**kwargs))
        )

    def from_data_frame(self, trace_data: GeoDataFrame):
        """

        :param trace_data:
        :return:
        """

        if geom_check(trace_data, "Point"):

            assert all(
                element in list(trace_data.columns.values)
                for element in [
                    "trace_id",
                ]
            ), (
                "Expected ['trace_id',] key to be in Data",
                "got %s",
                list(trace_data.columns.values),
            )

            for idx, feature in trace_data.iterrows():
                feature_geometry, feature_property = decompose_data_frame_row(feature)
                if "trace_point_id" in feature_property:
                    trace_point_id = feature_property["trace_point_id"]
                else:
                    trace_point_id = uuid.uuid1().int
                self.add(
                    x=feature_geometry["coordinates"][0],
                    y=feature_geometry["coordinates"][1],
                    trace_point_id=trace_point_id,
                    **feature_property,
                )

        elif geom_check(trace_data, "LineString"):
            for idx, feature in trace_data.iterrows():
                trace_id = uuid.uuid1().int

                feature_geometry, feature_property = decompose_data_frame_row(feature)
                line_string_coordinate = feature_geometry["coordinates"]

                for nodes in line_string_coordinate:
                    trace_point_id = uuid.uuid1().int
                    self.add(
                        x=nodes[0],
                        y=nodes[-1],
                        trace_point_id=trace_point_id,
                        trace_id=trace_id,
                        **feature_property,
                    )
        else:
            raise ValueError("Expected Geometry Type to be in ['Point', 'LineString']")

        return self

    def single_trace(self, trace: list):
        """

        :param trace:
        :return:
        """
        assert trace is not None, (
            "Expected trace_point_list to be of type 'list'" "got None"
        )
        assert type(trace) is list, (
            "Expected trace_point_list to be of type 'list'" "got %s",
            (type(trace)),
        )
        assert len(trace) != 0, (
            "Expected trace_point_list to be greater than zero" "got %s",
            (len(trace),),
        )
        trace_id = 0
        for trace in trace:
            assert len(trace) == 2, (
                "Expected trace to have 2 values 'X' and 'Y'" "got %s",
                (trace,),
            )
            trace_point_id = uuid.uuid1()
            self.add(
                x=trace[0],
                y=trace[-1],
                trace_point_id=trace_point_id.int,
                trace_id=trace_id,
            )
        return self

    def multiple_trace(self, traces: list):
        """

        :param traces:
        :return:
        """
        assert traces is not None, (
            "Expected multiple_trace_point_list to be of type 'list'" "got None"
        )
        assert type(traces) is list, (
            "Expected multiple_trace_point_list to be of type 'list'" "got %s",
            (type(traces)),
        )
        assert len(traces) != 0, (
            "Expected multiple_trace_point_list to be greater than zero" "got %s",
            (len(traces),),
        )
        for trace_point_list in traces:
            assert type(trace_point_list) is list, (
                "Expected trace_point_list to be of type 'list'" "got %s",
                (type(trace_point_list)),
            )

            trace_id = uuid.uuid1().int
            for trace in trace_point_list:
                assert len(trace) == 2, (
                    "Expected trace to have 2 values 'X' and 'Y'" "got %s",
                    (trace,),
                )
                trace_point_id = uuid.uuid1().int
                self.add(
                    x=trace[0],
                    y=trace[-1],
                    trace_point_id=trace_point_id,
                    trace_id=trace_id,
                )
        return self


def traces_from_data_frame(trace_data: GeoDataFrame) -> Traces:
    """
    Generate traces from GeoSpatial LineString or Point
    :param trace_data:
    :return:
    """
    assert supported_crs(trace_data), (
        "Supported CRS ['epsg:26910', 'epsg:32649']" "got %s",
        (trace_data.crs,),
    )
    return Traces().from_data_frame(trace_data)


def single_trace(trace: list) -> Traces:
    """
    Generate single trace from List of Coordinates
    [(trace_1_coord_1_x, trace_1_coord_1_y), (trace_1_coord_2_x, trace_1_coord_2_y),
    (trace_1_coord_3_x, trace_1_coord_3_y), (trace_1_coord_4_x, trace_1_coord_4_y)]

    :param trace:
    :return:
    """

    return Traces().single_trace(trace)


def multiple_traces(traces: list) -> Traces:
    """
    Generate collection of trace from List of Coordinates
    [
    [(trace_1_coord_1_x, trace_1_coord_1_y), (trace_1_coord_2_x, trace_1_coord_2_y),
    (trace_1_coord_3_x, trace_1_coord_3_y), (trace_1_coord_4_x, trace_1_coord_4_y)],

    [(trace_2_coord_1_x, trace_2_coord_1_y), (trace_2_coord_2_x, trace_2_coord_2_y),
    (trace_2_coord_3_x, trace_2_coord_3_y), (trace_2_coord_4_x, trace_2_coord_4_y)]
    ]

    :param traces:
    :return:
    """

    return Traces().multiple_trace(traces)


def trace_from_file(path: str) -> Traces:
    """

    :param path:
    :return:
    """
    return traces_from_data_frame(read_data_frame(path))
