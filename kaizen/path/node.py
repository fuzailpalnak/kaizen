class Node:
    def __init__(self, x: int, y: int, cost: float, parent_index):
        self.x = x  # index of grid
        self.y = y  # index of grid
        self.cost = cost
        self.parent_index = parent_index
        self.parent_node = None

    def __str__(self):
        return (
            str(self.x)
            + ","
            + str(self.y)
            + ","
            + str(self.cost)
            + ","
            + str(self.parent_index)
        )


class StartNode(Node):
    def __init__(self, x: int, y: int, cost: float, parent_index):
        super().__init__(x, y, cost, parent_index)


class GoalNode(Node):
    def __init__(
        self, x: int, y: int, cost: float, parent_index, is_intermediate=False
    ):
        super().__init__(x, y, cost, parent_index)
        self._is_intermediate = is_intermediate

    @property
    def is_intermediate(self):
        return self._is_intermediate


class Vertex:
    def __init__(self, x, y, is_conflict=False):
        self.x = x
        self.y = y

        self._is_conflict = is_conflict

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(self._is_conflict)

    @property
    def is_conflict(self):
        return self._is_conflict

    @is_conflict.setter
    def is_conflict(self, value: bool):
        self._is_conflict = value

    @classmethod
    def obstacle_vertex(cls, x, y, obstacle_map):
        if obstacle_map[x][y]:
            return cls(x, y, is_conflict=True)
        else:
            return cls(x, y)

    @classmethod
    def vertex(cls, x, y):
        return cls(x, y)
