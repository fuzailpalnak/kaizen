import fiona
import networkx as nx

from kaizen.path.node import Vertex


class Graph:
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    @classmethod
    def generate(cls, line_string: fiona.collection, obstacle_map, grid):
        graph = nx.Graph()
        for iterator, sub_feature in enumerate(line_string):
            trace = list()

            line_string_coordinate = sub_feature["geometry"]["coordinates"]

            for coordinates in line_string_coordinate[1:-1]:
                pixel_pos = grid.to_pixel_position(coordinates[0], coordinates[1])
                sx = pixel_pos[0]
                sy = pixel_pos[1]
                trace.append(Vertex.obstacle_vertex(sx, sy, obstacle_map))

            start_vertex = Vertex.obstacle_vertex(
                *(
                    grid.to_pixel_position(
                        line_string_coordinate[0][0], line_string_coordinate[0][1]
                    )
                ),
                obstacle_map
            )
            end_vertex = Vertex.obstacle_vertex(
                *(
                    grid.to_pixel_position(
                        line_string_coordinate[-1][0], line_string_coordinate[-1][1]
                    )
                ),
                obstacle_map
            )

            graph.add_edge(start_vertex, end_vertex, trace=trace)
        return cls(graph=graph)

    @classmethod
    def generate_with_reference(
        cls,
        line_string: fiona.collection,
        reference_line_string: fiona.collection,
        obstacle_map,
        grid,
    ):
        # TODO del
        # THIS METHOD IS IMPLEMENTED FOR SHOWCASE
        # TO IDENTIFY REFERENCE MAP MATCHING WILL BE REQUIRED
        graph = nx.Graph()
        start, end = line_string[0], line_string[-1]

        for iterator, sub_feature in enumerate(reference_line_string):
            trace = list()
            reference_line_string_coordinate = sub_feature["geometry"]["coordinates"]

            for coordinates in reference_line_string_coordinate[1:-1]:
                pixel_pos = grid.to_pixel_position(coordinates[0], coordinates[1])
                sx = pixel_pos[0]
                sy = pixel_pos[1]
                trace.append(Vertex.obstacle_vertex(sx, sy, obstacle_map))

            start_vertex = Vertex.obstacle_vertex(
                *(grid.to_pixel_position(start[0], start[1])), obstacle_map
            )
            end_vertex = Vertex.obstacle_vertex(
                *(grid.to_pixel_position(end[0], end[1])), obstacle_map
            )

            graph.add_edge(start_vertex, end_vertex, trace=trace)
        return cls(graph=graph)
