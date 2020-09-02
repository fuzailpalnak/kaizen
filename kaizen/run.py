import fiona
import networkx as nx

from kaizen.navigate import Navigate
from kaizen.path.finder import  AStar
from kaizen.map.grid import PixelGrid
from kaizen.map.obstacle import Obstacle

ox, oy = [], []
sx, sy = 0, 0
gx, gy = 0, 0

# f = fiona.open(r"D:\Cypherics\Library\kaizen\data\demo1_bfp.shp")
f = fiona.open(r"D:\Cypherics\Library\kaizen\data\demo2_bfp.shp")
# f = fiona.open(r"D:\Cypherics\Library\kaizen\data\BFP_obstacle.shp")


# f = fiona.open(r"D:\Cypherics\Library\aligner\data\BFPTOTAL.shp")

minx, miny, maxx, maxy = f.bounds


grid = PixelGrid.pixel_computation(minx, miny, maxx, maxy, 1)
import time

# starting time
start = time.time()
obs_map = Obstacle.from_polygon(grid, f, 2)
end = time.time()

# total time taken
print(f"Runtime of the program is {end - start}")
# exit()
# k = fiona.open(r"D:\Cypherics\Library\kaizen\data\demo1.shp")

k = fiona.open(r"D:\Cypherics\Library\kaizen\data\demo2.shp")
# k = fiona.open(r"D:\Cypherics\Library\aligner\data\goal_with_mutiple_conflicts_26910.shp")

navigate = Navigate()
connection = navigate.generate_path(k, obs_map.obstacle_position, grid)
# G=nx.read_shp(r"D:\Cypherics\Library\kaizen\data\demo1.shp")
import matplotlib.pyplot as plt

obs_map.animate_obstacle()
plt.plot(sx, sy, "og")
plt.plot(gx, gy, "xb")
plt.grid(True)
plt.axis("equal")
# plt.show()

for connect in connection:
    for trace_id, trace in connect.trace.items():
        start = trace[0]
        goal = trace[-1]
        support_vertices = trace[1:-1]
        f1 = [ver for ver in trace[1:-1] if not ver.is_conflict]
        astar = AStar.navigate_from_start_and_goal_with_intermediate(start, goal, f1)

        rx, ry = astar.find_path(grid, obs_map.obstacle_position, space_threshold=10)

        print(rx, ry)
        lk = list()
        for i in range(len(rx)):
            lk.append(grid.to_spatial_position(rx[i], ry[i]))

        from shapely.geometry import mapping, LineString

        with fiona.open(r"D:\Cypherics\Library\aligner\data\my_shp123.shp.shp", 'w', 'ESRI Shapefile', k.schema, crs=f.crs) as c:
            ## If there are multiple geometries, put the "for" loop here
            c.write({
                'geometry': mapping(LineString(lk)),
                'properties': {'id': 123},
            })
            c.flush()
# f = fiona.open(r"D:\Cypherics\Library\kaizen\data\road_goal.shp")
# for i, feature in enumerate(k):
#     coords = feature["geometry"]["coordinates"]
#     print(feature)
#     for co in coords:
#         d = grid.to_pixel_position(co[0], co[1])
#         sx = d[0]
#         sy = d[1]
#         if obs_map[sx][sy]:
#             print(True)

#         d = grid.to_pixel_position(co[1][0], co[1][1])
#
#         gx = d[0]
#         gy = d[1]
# #
# import matplotlib.pyplot as plt
#
# obstacle_obj.animate_obstacle()
# plt.plot(sx, sy, "og")
# plt.plot(gx, gy, "xb")
# plt.grid(True)
# plt.axis("equal")
#
#
#
# rx, ry = astar.path(sx, sy, gx, gy)
# print(rx, ry)
# lk = list()
# for i in range(len(rx)):
#     lk.append(grid.to_spatial_position(rx[i], ry[i]))
#
# from shapely.geometry import mapping, LineString
#
# with fiona.open(r"D:\Cypherics\Library\kaizen\data\my_shp123.shp", 'w', 'ESRI Shapefile', f.schema, crs=f.crs) as c:
#     ## If there are multiple geometries, put the "for" loop here
#     c.write({
#         'geometry': mapping(LineString(lk)),
#         'properties': {'id': 123},
#     })
#     c.flush()
#

# G=nx.Graph()
# G.add_node("a")
# G.add_nodes_from(["b","c"])
#
# G.add_edge(1,2)
# G.add_edge(1, 3)
# edge = ("d", "e")
# G.add_edge(*edge)
# G.add_edge(*edge)
#
# print("Nodes of graph: ")
# print(G.nodes())
# print("Edges of graph: ")
# print(G.edges(1))
# import fiona
# f = fiona.open(r"D:\Cypherics\Library\kaizen\data\road_goal.shp")
# d = nx.read_shp(r"D:\Cypherics\Library\kaizen\data\test_road_grid_26910.shp")
# print(d.nodes())
# print(d.edges(list(d.nodes())[0]))
#
# for f1 in f:
#     print(f1)


# cv2.imwrite("j{}.png".format(i), img1)

# for boundary_coordinate_iterator in range(
#         len(boundary_coordinates) - 1
# ):
#     interpolate_line = mapping(
#         interpolate_linestring(
#             LineString(
#                 (
#                     boundary_coordinates[boundary_coordinate_iterator],
#                     boundary_coordinates[
#                         boundary_coordinate_iterator + 1
#                         ],
#                 )
#             ),
#             interpolate_len,
#         )
#     )["coordinates"]
#     for co in interpolate_line:
#         # extend the obstacle boundary by Motion.radius() from the polygon boundary
#         pixel_pos = grid.to_pixel_position(co[0], co[1])
#         df = np.array(
#             [list(mx - pixel_pos[0]), list(my - pixel_pos[1])]
#         )
#
#         dfs = np.hypot(*df)
#         obstacle_map[dfs <= 1] = True
#
#         boundary_obstacle.append((pixel_pos[0], pixel_pos[1]))
#
# inner_obstacle_pixel = cls.inner_obstacle_pixel(mx, my, boundary_obstacle)
# inner_obstacle_pixel = inner_obstacle_pixel.reshape(grid.x_width, grid.y_width)
#
# obstacle_map[inner_obstacle_pixel == True] = True
