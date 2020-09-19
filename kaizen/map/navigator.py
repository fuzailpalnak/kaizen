import math
from collections import OrderedDict
from typing import Tuple, List

import numpy as np

import matplotlib.pyplot as plt

from kaizen.map.robot import Robot
from kaizen.map.trace import TracePoint
from kaizen.utils.gis import pixel_position, spatial_position
from kaizen.utils.numerical import (
    angle_between_vector,
    vector,
    diagonal_distance,
    euclidean_distance,
    manhattan_distance,
)

SHOW_ANIMATION = False


class Node:
    def __init__(self, x: int, y: int, cost: float, parent_index):
        """

        :param x:
        :param y:
        :param cost:
        :param parent_index:
        """
        self.x = x
        self.y = y
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
        """

        :param x:
        :param y:
        :param cost:
        :param parent_index:
        """
        super().__init__(x, y, cost, parent_index)


class GoalNode(Node):
    def __init__(
        self, x: int, y: int, cost: float, parent_index, is_intermediate=False
    ):
        """

        :param x:
        :param y:
        :param cost:
        :param parent_index:
        :param is_intermediate:
        """
        super().__init__(x, y, cost, parent_index)
        self._is_intermediate = is_intermediate

    @property
    def is_intermediate(self):
        """
        Informs weather the goal is either intermediate of final
        :return:
        """
        return self._is_intermediate


class Navigator:
    # http://theory.stanford.edu/~amitp/GameProgramming/
    def __init__(self, grid):
        self._grid = grid

        #  INITIAL NAVIGATION SPACE IS SET TO 90 DEG, LATER ON THE PATH FINDER WILL
        #  ADJUST AS PER THE LOCATION OF GOAL NODE

        # [USE CASE SPECIFIC]
        # TO SIMPLIFY THE SEARCH SPACE AND AVOID SUPERFLUOUS -POTENTIAL NODES.
        # THE IDEA IS THAT, NODES USUALLY FOLLOW SUCCESSIVE PATH USE THIS TO OUR ADVANTAGE
        # AND TO MINIMIZE THE SEARCH SPACE.
        # ONLY THOSE NODES ARE EXPLORED WHICH LIE IN THE NAVIGATE_SPACE
        self._navigate_space = 90

        self._direction = Robot().direction()

    def _node_connectivity(
        self, start_node: StartNode, goal_collection: list
    ) -> OrderedDict:
        """
        Calculates the distance from the current node along with consecutive nodes to the final goal node
        :param start_node:
        :param goal_collection:
        :return:
        """

        connectivity_meta = OrderedDict()

        # ELEMENT FROM 0 th INDEX IS DELETED ON 'del self.goal_collection[0]', WHICH RESULTS IN A GLOBAL DELETE,
        # AND IF 'self.goal_collection'IS COPIED DIRECTLY THEN THE ELEMENTS FROM ITS REFERENCE IS ALSO DELETED, HENCE
        # 'FOR' LOOP IS USED TO CREATE A NEW COPY.
        distance = self.diagonal(goal_collection[0], start_node)
        for j in range(len(goal_collection) - 1):
            distance += self.diagonal(goal_collection[j], goal_collection[j + 1])

        my_intermediate_goal = goal_collection[0]
        connectivity_meta[start_node] = {
            "connectivity": [goal for goal in goal_collection],
            "total_to_goal": distance,
            "my_intermediate_goal": my_intermediate_goal,
        }

        for i, goal in enumerate(goal_collection):
            collection = goal_collection[i + 1 :]
            if len(collection) == 0:
                distance = 0
                my_intermediate_goal = None

            else:
                my_intermediate_goal = collection[0]
                distance = self.diagonal(goal, collection[0])
                for j in range(len(collection) - 1):
                    distance += self.diagonal(collection[j], collection[j + 1])

            connectivity_meta[goal] = {
                "connectivity": [goal for goal in collection],
                "total_to_goal": distance,
                "my_intermediate_goal": my_intermediate_goal,
            }
        return connectivity_meta

    @property
    def navigate_space(self):
        return self._navigate_space

    @navigate_space.setter
    def navigate_space(self, value):
        self._navigate_space = value

    def pre_compute_goal_heuristics(self):
        """

        :return:
        """
        raise NotImplementedError

    @staticmethod
    def _search_space(goal, start, potential):
        """

        :param goal:
        :param start:
        :param potential:
        :return:
        """
        angle = angle_between_vector(
            vector((start.x, start.y), (potential.x, potential.y)),
            vector((start.x, start.y), (goal.x, goal.y)),
        )

        return np.degrees(angle)

    @staticmethod
    def diagonal(goal: GoalNode, node: Node, d1=1, d2=math.sqrt(2)):
        """

        :param goal:
        :param node:
        :param d1:
        :param d2:
        :return:
        """
        # https://stackoverflow.com/questions/46974075/a-star-algorithm-distance-heuristics
        # http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html
        # https://www.growingwiththeweb.com/2012/06/a-pathfinding-algorithm.html

        return diagonal_distance((node.x, node.y), (goal.x, goal.y), d1, d2)

    @staticmethod
    def euclidean(goal: GoalNode, node: Node):
        """

        :param goal:
        :param node:
        :return:
        """
        return euclidean_distance((node.x, node.y), (goal.x, goal.y))

    @staticmethod
    def manhattan(goal: GoalNode, node: Node):
        """

        :param goal:
        :param node:
        :return:
        """
        return manhattan_distance((node.x, node.y), (goal.x, goal.y))

    def path(
        self,
        trace: list,
        space_threshold: float,
    ) -> list:
        """

        :param trace:
        :param space_threshold:
        :return:
        """
        raise NotImplementedError

    def calc_heuristic(self, goal: GoalNode, potential_node: Node, **kwargs) -> float:
        """

        :param goal:
        :param potential_node:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def obstacle_check(self, grid, node: Node) -> bool:
        raise NotImplementedError

    def final_path(self, grid, goal: GoalNode, closed_set: dict) -> Tuple[List, List]:
        """

        :param grid:
        :param goal:
        :param closed_set:
        :return:
        """
        raise NotImplementedError

    def _spatial_path(self, rx: list, ry: list):
        assert len(rx) == len(ry), (
            "Expected Equal number of points in rx and ry"
            "got %s and %s", (len(rx), len(ry), )
        )

        assert (len(rx) > 1 and len(ry) > 1), (
            "Expected to have more than one coordinates"
        )
        spatial_coordinate = list()
        for x, y in zip(rx, ry):
            spatial_coordinate.append(spatial_position(x, y, self._grid.transform))

        return spatial_coordinate

    def _generate_nodes(
        self, start: TracePoint, goal: TracePoint, intermediate_goals: list
    ) -> Tuple[StartNode, GoalNode, list]:
        """
        Generate nodes fro navigator to traverse

        :param start:
        :param goal:
        :param intermediate_goals:
        :return:
        """
        assert not self._grid.collision_check(
            *pixel_position(start.x, start.y, self._grid.transform)
        ) and not self._grid.collision_check(
            *pixel_position(start.x, start.y, self._grid.transform)
        ), (
            "Expected Start Node and Goal Node to located outside the Obstacle Zone"
            "Apply 'filter_trace' to eliminate trace point which lie on obstacle"
        )
        start_node = StartNode(
            *pixel_position(start.x, start.y, self._grid.transform),
            cost=0.0,
            parent_index=-1,
        )

        goal_node = GoalNode(
            *pixel_position(goal.x, goal.y, self._grid.transform),
            cost=0.0,
            parent_index=-1,
        )

        goals = list()
        for index, trace_point in enumerate(intermediate_goals):

            goal = GoalNode(
                *pixel_position(trace_point.x, trace_point.y, self._grid.transform),
                cost=0.0,
                parent_index=-1,
                is_intermediate=True,
            )

            if not self._grid.collision_check(goal.x, goal.y):
                goals.append(goal)

        goals.append(goal_node)
        return start_node, goal_node, goals

    def _make_data(self, trace: list) -> Tuple[StartNode, GoalNode, list]:
        assert len(trace) >= 2, (
            "Expected at least two trace points" "got %s",
            (len(trace),),
        )
        assert all(
            [isinstance(trace_point, TracePoint) for trace_point in trace]
        ), "Expected all points to be TracePoint, got types %s." % (
            ", ".join([str(type(v)) for v in trace])
        )
        return self._generate_nodes(trace[0], trace[-1], trace[1:-1])

    def _filter_trace(self, trace: list) -> List[TracePoint]:
        assert len(trace) >= 2, (
            "Expected at least two trace points" "got %s",
            (len(trace),),
        )
        assert all(
            [isinstance(trace_point, TracePoint) for trace_point in trace]
        ), "Expected all points to be TracePoint, got types %s." % (
            ", ".join([str(type(v)) for v in trace])
        )
        return [
            trace_point
            for trace_point in trace
            if not self._grid.collision_check(
                *pixel_position(trace_point.x, trace_point.y, self._grid.transform)
            )
        ]

    def animate_obstacle(self):
        x, y = np.where(self._grid.obstacle == True)
        plt.plot(x, y, ".k")


class AStar(Navigator):
    def __init__(self, grid):
        super().__init__(grid)

    def pre_compute_goal_heuristics(self):
        pass

    def path(self, trace: list, space_threshold=30, filter_trace=True) -> list:
        """

        The first point of the trace will be assigned as start and end point as the final goal
        and all the points in between will be assigned as intermediate goal

        :param filter_trace: bool, if the trace passed over the obstacle pixel, set this to true
        this will remove such points which pass over obstacle zone
        :param trace:
        :param space_threshold:
        :return:
        """

        if filter_trace:
            trace = self._filter_trace(trace)

        start, goal, goal_collection = self._make_data(trace)

        connectivity_meta = self._node_connectivity(start, goal_collection)

        open_set, closed_set = dict(), dict()
        open_set[self._grid.grid_index(start.x, start.y)] = start

        n_goal = goal_collection[0]
        start.parent_node = start
        parent_node = start

        previous_goal = n_goal

        while 1:
            if len(open_set) == 0:
                print("Couldn't Reach to Goal")
                break

            grid_id = min(
                open_set,
                key=lambda o: open_set[o].cost
                + self.calc_heuristic(
                    n_goal, open_set[o], connectivity_meta=connectivity_meta
                ),
            )

            current = open_set[grid_id]

            if current.parent_index != -1:
                # VEC(start -> potential).dot(VEC(start -> current_goal)) lies in between
                # VEC(start -> previous_goal).dot(VEC(start -> current_goal))
                # VEC(start -> previous_goal).dot(VEC(start -> current_goal)) is nothing but self.navigate_space

                if (
                    self._search_space(previous_goal, start, current)
                    > self.navigate_space
                    or self._search_space(n_goal, start, current) > self.navigate_space
                ):
                    del open_set[grid_id]
                    continue

            # show graph
            if SHOW_ANIMATION:  # pragma: no cover
                plt.plot(current.x, current.y, "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect(
                    "key_release_event",
                    lambda event: [exit(0) if event.key == "escape" else None],
                )
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == n_goal.x and current.y == n_goal.y:
                if not n_goal.is_intermediate:
                    # FINAL GOAL FOUND
                    n_goal.parent_index = current.parent_index
                    n_goal.cost = current.cost
                    break
                else:
                    # INTERMEDIATE GOAL FOUND
                    intermediate_goal = current

                    previous_goal = n_goal
                    parent_node = n_goal

                    # REMOVE THE GOAL FROM COLLECTION AS IT HAS BEEN FOUND
                    del goal_collection[0]

                    # GET THE NEW GOAL
                    n_goal = goal_collection[0]

                    # CALCULATE THE SEARCH SPACE FOR NEW GOAL
                    self.navigate_space = self._search_space(
                        n_goal, start, intermediate_goal
                    )

                    # [USE CASE SPECIFIC]
                    # IF NAVIGATE SPACE IS SMALL THAN THRESHOLD THEN INCREASE
                    # REASON FOR THIS CHECK IS, SOME NODES ARE SUCCESSIVE WITH A HUGE OBSTACLE IN BETWEEN
                    # AND TO AVOID THE OBSTACLE SEARCH SPACE HAS TO BE EXPANDED AS THE OBSTACLE IS LARGER THAN THE
                    # NAVIGATE SPACE.

                    if self.navigate_space < space_threshold:
                        self.navigate_space = self.navigate_space + space_threshold

            # REMOVE VISITED FROM OPEN SET
            del open_set[grid_id]

            # MARK DONE
            closed_set[grid_id] = current

            # EXPAND IN 8 DIRECTION, DIAGONAL WEIGHT IS math.sqrt(2) AND 1 OTHERWISE
            for i, _ in enumerate(self._direction):
                node = Node(
                    current.x + self._direction[i][0],
                    current.y + self._direction[i][1],
                    current.cost + self._direction[i][2],
                    grid_id,
                )
                new_node_grid_id = self._grid.grid_index(node.x, node.y)

                # SKIP NODE WHEN OBSTACLE ENCOUNTERED
                if not self.obstacle_check(self._grid, node):
                    continue

                # SKIP IF MARKED VISITED
                if new_node_grid_id in closed_set:
                    continue

                # NOT YET EXPLORED
                if new_node_grid_id not in open_set:
                    # PARENT ID IS USED TO TRACK, DURING WHICH GOAL SEARCH THE NODE WAS CONSIDERED
                    if node.parent_node is None:
                        node.parent_node = parent_node

                    # NEW NODE
                    open_set[new_node_grid_id] = node
                else:
                    if open_set[new_node_grid_id].cost > node.cost:
                        # PARENT ID IS USED TO TRACK, DURING WHICH GOAL SEARCH THE NODE WAS CONSIDERED
                        if node.parent_node is None:
                            node.parent_node = parent_node

                        # BEST PATH FOUND
                        open_set[new_node_grid_id] = node

        return self._spatial_path(*self.final_path(self._grid, n_goal, closed_set))

    def calc_heuristic(self, goal: GoalNode, potential_node: Node, **kwargs) -> float:
        """

        :param goal:
        :param potential_node:
        :param kwargs:
        :return:
        """
        connectivity_meta = kwargs["connectivity_meta"]
        weight = 1

        # LOOK FOR THE INTERMEDIATE GOAL OF THE CONSIDERED NODE AND COMPUTE COST ACCORDINGLY
        # [GET MY ACTUAL GOAL]
        #       THEN - COMPUTE COST(MY_ACTUAL_GOAL, POTENTIAL_NODE) + COST FROM (MY_ACTUAL_GOAL, TO FINAL GOAL)
        my_cost_from_intermediate_to_final_goal = 0
        my_actual_goal = connectivity_meta[potential_node.parent_node][
            "my_intermediate_goal"
        ]

        if my_actual_goal.is_intermediate:
            my_cost_from_intermediate_to_final_goal = connectivity_meta[my_actual_goal][
                "total_to_goal"
            ]

        cost_from_node_to_actual_goal = self.diagonal(my_actual_goal, potential_node)
        cost = cost_from_node_to_actual_goal + my_cost_from_intermediate_to_final_goal

        return weight * cost

    def obstacle_check(self, grid, node: Node) -> bool:
        """

        :param grid:
        :param node:
        :return:
        """
        px = node.x
        py = node.y

        if px < grid.min_x:
            return False
        elif py < grid.min_y:
            return False
        elif px >= grid.max_x:
            return False
        elif py >= grid.max_y:
            return False
        # collision check
        if grid.obstacle[node.x][node.y]:
            return False
        return True

    def final_path(self, grid, goal: GoalNode, closed_set: dict) -> Tuple[List, List]:
        """

        :param grid:
        :param goal:
        :param closed_set:
        :return:
        """
        rx, ry = [goal.x], [goal.y]
        parent_index = goal.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]

            rx.append(n.x)
            ry.append(n.y)

            parent_index = n.parent_index
        return rx, ry
