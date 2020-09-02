import fiona
from collections import defaultdict


class Navigate:
    # TODO create a connectivity graph
    # TODO Adjacency matrix for start and end vertex of trace
    def __init__(self):
        self.connection = list()

    def generate_path(self, line_string: fiona.collection, obstacle_map, grid):
        for iterator, sub_feature in enumerate(line_string):
            trace = Trace(iterator)
            line_string_coordinate = sub_feature["geometry"]["coordinates"]
            for coordinates in line_string_coordinate:
                pixel_pos = grid.to_pixel_position(coordinates[0], coordinates[1])
                sx = pixel_pos[0]
                sy = pixel_pos[1]
                if obstacle_map[sx][sy]:
                    trace.add_vertex(Vertex(sx, sy, is_conflict=True))
                else:
                    trace.add_vertex(Vertex(sx, sy))

            self.connection.append(trace)
        return self.connection


class Vertex:
    def __init__(self, x, y, is_conflict=False):
        self.x = x
        self.y = y

        self._is_conflict = is_conflict

    def __str__(self):
        return (
            str(self.x)
            + ","
            + str(self.y)
            + ","
            + str(self._is_conflict)
        )

    @property
    def is_conflict(self):
        return self._is_conflict

    @is_conflict.setter
    def is_conflict(self, value: bool):
        self._is_conflict = value


class Trace:
    def __init__(self, trace_id: int):
        self.trace = defaultdict(list)
        self._trace_id = trace_id

    def add_vertex(self, vertex: Vertex):
        self.trace[self._trace_id].append(vertex)
