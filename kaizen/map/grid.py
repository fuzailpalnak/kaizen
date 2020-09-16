from affine import Affine
from rasterio.transform import rowcol, xy


class Grid:
    def __init__(self, minx, miny, maxx, maxy, resolution):
        """

        :param minx:
        :param miny:
        :param maxx:
        :param maxy:
        :param resolution:
        """
        self.minx = minx
        self.miny = miny

        self.maxx = maxx
        self.maxy = maxy

        self.resolution = resolution

        # Compute Pixel Dimension
        self.x_width = abs(round((self.maxx - self.minx) / self.resolution))
        self.y_width = abs(round((self.maxy - self.miny) / self.resolution))

    def x_pos(self, x):
        """

        :param x:
        :return:
        """
        return x * self.resolution + self.minx

    def y_pos(self, y):
        """

        :param y:
        :return:
        """
        return y * self.resolution + self.miny

    def grid_index(self, x, y):
        """
        GET THE POSITION OF X, Y ON THE GRID
        IF THE GRID IS 10X10 THEN THERE ARE 100 SUB GRID OF PIXEL 1, THIS FUNCTION TELLS WHICH SUB GRID THE X,Y BELONG

        :param x:
        :param y:
        :return:
        """
        return (y - self.miny) * self.x_width + (x - self.minx)


class PixelGrid(Grid):
    """
    Pixel Grid Over the Spatial Coordinates
    """
    def __init__(
        self,
        minx: float,
        miny: float,
        maxx: float,
        maxy: float,
        resolution: float,
        transform: Affine,
    ):
        super().__init__(minx, miny, maxx, maxy, resolution)

        self._transform = transform

    @property
    def transform(self):
        return self._transform

    def to_pixel_position(self, x: float, y: float) -> (float, float):
        """
        CONVERT SPATIAL COORDINATES TO PIXEL X AND Y

        :param x:
        :param y:
        :return:
        """
        return rowcol(self.transform, x, y)

    def to_spatial_position(self, x: int, y: int) -> (float, float):
        """
        CONVERT PIXEL COORDINATE TO SPATIAL COORDINATES

        :param x:
        :param y:
        :return:
        """
        return xy(self.transform, x, y)

    @staticmethod
    def generate_affine(minx: float, maxy: float, resolution: float) -> Affine:
        """
        Generate affine transform over the spatial coordinates
        :param minx: min x of the extent
        :param maxy: max y of the extent
        :param resolution: what resolution to use to convert the spatial coordinates in pixel
        :return:
        """
        return Affine.translation(minx, maxy) * Affine.scale(resolution, -resolution)

    @classmethod
    def pixel_computation(
        cls, minx: float, miny: float, maxx: float, maxy: float, resolution: float
    ):
        """
        convert spatial coordinates to image height and image width
        :param minx:
        :param miny:
        :param maxx:
        :param maxy:
        :param resolution:
        :return:
        """
        transform = cls.generate_affine(minx, maxy, resolution)
        row_start, col_start = rowcol(transform, minx, maxy)
        row_stop, col_stop = rowcol(transform, maxx, miny)
        return cls(row_start, col_start, row_stop, col_stop, resolution, transform)
