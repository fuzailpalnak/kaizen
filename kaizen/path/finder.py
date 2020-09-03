import math
from collections import OrderedDict

import numpy as np

import matplotlib.pyplot as plt

from kaizen.map.obstacle import Obstacle
from kaizen.path.node import StartNode, GoalNode, Node, Vertex
from kaizen.path.robot import Robot
from kaizen.utils.numerical import angle_between_vector, vector

show_animation = True


class PathFinder:
    # http://theory.stanford.edu/~amitp/GameProgramming/
    def __init__(self, grid, obstacle: Obstacle):
        self._grid = grid
        self._obstacle = obstacle

        #  INITIAL NAVIGATION SPACE IS SET TO 90 DEG, LATER ON THE PATH FINDER WILL
        #  ADJUST AS PER THE LOCATION OF GOAL NODE

        # [USE CASE SPECIFIC]
        # TO SIMPLIFY THE SEARCH SPACE AND AVOID SUPERFLUOUS -POTENTIAL NODES.
        # THE IDEA IS THAT, NODES USUALLY FOLLOW SUCCESSIVE PATH USE THIS TO OUR ADVANTAGE
        # AND TO MINIMIZE THE SEARCH SPACE.
        # ONLY THOSE NODES ARE EXPLORED WHICH LIE IN THE NAVIGATE_SPACE
        self._navigate_space = 90

        self._direction = Robot().direction()

    @staticmethod
    def node_connectivity(start_node: StartNode, goal_collection: list):
        connectivity_meta = OrderedDict()

        # ELEMENT FROM 0 th INDEX IS DELETED ON 'del self.goal_collection[0]', WHICH RESULTS IN A GLOBAL DELETE,
        # AND IF 'self.goal_collection'IS COPIED DIRECTLY THEN THE ELEMENTS FROM ITS REFERENCE IS ALSO DELETED, HENCE
        # 'FOR' LOOP IS USED TO CREATE A NEW COPY.
        connectivity_meta[start_node] = {
            "connectivity": [goal for goal in goal_collection],
        }

        for i, goal in enumerate(goal_collection):
            connectivity_meta[goal] = {
                "connectivity": [goal for goal in goal_collection[i + 1 :]],
            }
        return connectivity_meta

    @property
    def navigate_space(self):
        return self._navigate_space

    @navigate_space.setter
    def navigate_space(self, value):
        self._navigate_space = value

    def pre_compute_goal_heuristics(self):
        raise NotImplementedError

    @staticmethod
    def search_space(goal, start, potential):
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
        # https://stackoverflow.com/questions/46974075/a-star-algorithm-distance-heuristics
        # http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html
        # https://www.growingwiththeweb.com/2012/06/a-pathfinding-algorithm.html

        dx = abs(node.x - goal.x)
        dy = abs(node.y - goal.y)

        return d1 * (dx + dy) + (d2 - 2 * d1) * min(dx, dy)

    @staticmethod
    def euclidean(goal: GoalNode, node: Node):
        return math.hypot(goal.x - node.x, goal.y - node.y)

    @staticmethod
    def manhattan(goal: GoalNode, node: Node):
        return abs(goal.x - node.x) + abs(goal.y - node.y)

    def path(
        self,
        start: Vertex,
        goal: Vertex,
        intermediate_goal: list,
        space_threshold: float,
    ):
        raise NotImplementedError

    def calc_heuristic(self, goal: GoalNode, potential_node: Node, **kwargs) -> float:
        raise NotImplementedError

    def obstacle_check(self, grid, obstacle_map, node: Node) -> bool:
        raise NotImplementedError

    def final_path(self, grid, goal: GoalNode, closed_set: dict):
        raise NotImplementedError

    @staticmethod
    def generate_nodes(start: Vertex, goal: Vertex, intermediate_goals: list):
        start_node = StartNode(
            start.x,
            start.y,
            0.0,
            -1,
        )
        goal_node = GoalNode(
            goal.x,
            goal.y,
            0.0,
            -1,
        )
        intermediate_goal = list()

        for index, vertex in enumerate(intermediate_goals):
            goal = GoalNode(
                vertex.x,
                vertex.y,
                0.0,
                -1,
                is_intermediate=True,
            )
            intermediate_goal.append(goal)
        intermediate_goal.append(goal_node)
        return start_node, goal_node, intermediate_goal


class AStar(PathFinder):
    def __init__(self, grid, obstacle: Obstacle):
        super().__init__(grid, obstacle)

    def pre_compute_goal_heuristics(self):
        pass

    def path(
        self, start: Vertex, goal: Vertex, intermediate_goal: list, space_threshold=5
    ):
        start, goal, goal_collection = self.generate_nodes(
            start, goal, intermediate_goal
        )

        connectivity_meta = self.node_connectivity(start, goal_collection)

        open_set, closed_set = dict(), dict()
        open_set[self._grid.grid_index(start.x, start.y)] = start

        n_goal = goal_collection[0]
        start.parent_node = start
        parent_node = start

        previous_goal = n_goal
        # TODO SOLVE THE PROBLEM OF DIRECTION TO CHOOSE FOR NAVIGATION
        # TODO HEURISTIC FOR MULTI GOAL

        while 1:
            if len(open_set) == 0:
                print("Open set is empty..")
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
                    self.search_space(previous_goal, start, current)
                    > self.navigate_space
                    or self.search_space(n_goal, start, current) > self.navigate_space
                ):
                    del open_set[grid_id]
                    continue

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self._grid.x_pos(current.x), self._grid.y_pos(current.y), "xc")
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
                    self.navigate_space = self.search_space(
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
                if not self.obstacle_check(self._grid, self._obstacle.obs_map, node):
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

        rx, ry = self.final_path(self._grid, n_goal, closed_set)

        return rx, ry

    def calc_heuristic(self, goal: GoalNode, potential_node: Node, **kwargs) -> float:
        connectivity_meta = kwargs["connectivity_meta"]
        connectivity = connectivity_meta[potential_node.parent_node]["connectivity"]
        index_of_goal_node = connectivity.index(goal)

        missed_goal_cost = 0

        # TODO BAD HEURISTIC, CURRENT PROBLEM, COST ASSOCIATED WITH FIRST GOAL IS COMPUTED JUST CONSIDERING
        #  THE FIRST GOAL AS FINAL GOAL, WHICH CAUSES PROBlEM AS REST POTENTIAL NODES ARE COMPUTED CONSIDERING MISSED
        #  GOAL WHICH LEADS TO FIRST GOAL NODES TO BE ALWAYS WITH LOW COST

        # [USE CASE SPECIFIC]
        if index_of_goal_node != 0:
            # NOTE TO SELF - DON'T THINK THIS IS AN ADMISSIBLE HEURISTIC
            # RETHINK HEURISTIC TO MAKE IT ADMISSIBLE-

            # MISSED_GOAL_COST = HOPS_AWAY_TO MISSED_GOAL * DISTANCE(MISSED_GOAL, POTENTIAL_NODE)
            for missed_goal in connectivity[:index_of_goal_node]:
                missed_goal_cost += list(connectivity_meta.keys()).index(
                    missed_goal
                ) * self.diagonal(missed_goal, potential_node)

        return self.diagonal(goal, potential_node) + missed_goal_cost

    def obstacle_check(self, grid, obstacle_map, node: Node) -> bool:
        px = grid.x_pos(node.x)
        py = grid.y_pos(node.y)

        if px < grid.minx:
            return False
        elif py < grid.miny:
            return False
        elif px >= grid.maxx:
            return False
        elif py >= grid.maxy:
            return False
        # collision check
        if obstacle_map[node.x][node.y]:
            return False
        return True

    def final_path(self, grid, goal: GoalNode, closed_set: dict):
        rx, ry = [grid.x_pos(goal.x)], [grid.y_pos(goal.y)]
        parent_index = goal.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]

            rx.append(grid.x_pos(n.x))
            ry.append(grid.y_pos(n.y))

            parent_index = n.parent_index
        return rx, ry
