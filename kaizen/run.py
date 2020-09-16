import fiona
from shapely.geometry import mapping

from kaizen import resolve_conflict
import geopandas

ox, oy = [], []
sx, sy = 0, 0
gx, gy = 0, 0

# f = fiona.open(r"D:\Cypherics\Library\kaizen\data\demo1_bfp.shp")
# f = fiona.open(r"D:\Cypherics\Library\kaizen\data\demo2_bfp.shp")
# f = fiona.open(r"D:\Cypherics\Library\kaizen\data\BFP_obstacle.shp")


# f = fiona.open(r"D:\Cypherics\Library\aligner\data\BFPTOTAL.shp")

# minx, miny, maxx, maxy = f.bounds


# exit()
# k = fiona.open(r"D:\Cypherics\Library\kaizen\data\demo1.shp")

# k = fiona.open(r"D:\Cypherics\Library\kaizen\data\demo2.shp")
# k = fiona.open(
#     r"D:\Cypherics\Library\kaizen\data\goal_with_mutiple_conflicts_26910.shp"
# )
fff = r"D:\GIS\map-match\shp\input\connected_road.shp"
f = geopandas.read_file(r"D:\Cypherics\Library\kaizen\data\demo1_bfp.shp")
f1 = geopandas.read_file(fff)
print(list(f1.columns.values))
# print()
for row in f1.iterrows():
    print(mapping(row))
print("j")
# resolve_conflict.resolution(obstacle="D:\Cypherics\Library\kaizen\data\demo1_bfp.shp",
#                             resolve_path="D:\Cypherics\Library\kaizen\data\demo1.shp",
#                             obstacle_thickness=2, grid_resolution=1,
#                             search_space_threshold=20)
