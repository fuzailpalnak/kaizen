
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
    def __init__(self, x: int, y: int, cost: float, parent_index, is_intermediate=False):
        super().__init__(x, y, cost, parent_index)
        self._is_intermediate = is_intermediate

    @property
    def is_intermediate(self):
        return self._is_intermediate
