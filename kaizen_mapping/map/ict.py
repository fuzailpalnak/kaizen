import inspect
import itertools
from typing import Union, List, Tuple

import numpy as np
from geopandas import GeoDataFrame

from kaizen_mapping.map import refresh_print
from kaizen_mapping.map.grid import PixelGrid, Grid
from kaizen_mapping.map.matcher import Match
from kaizen_mapping.map.road import road_network_from_data_frame, RoadNetwork
from kaizen_mapping.map.trace import (
    traces_from_data_frame,
    Traces,
)
from kaizen_mapping.utils.gis import read_data_frame
from kaizen_mapping.utils.numerical import compute_center_of_mass, decompose_matrix


class Association:
    """
    Find association among source and target, i.e finding which point from target is the most likely point to which the
    source should transform in to.

    target can be thought of the ground truth, and source as input, and the task of association is to relate
    which point from source is a true match to the target, so when ict is performed on the association, minimum error
    is achieved.

    """

    @staticmethod
    def nearest_neighbour(
        target: np.ndarray, source: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        _batch_size = kwargs["batch_size"]
        d = np.linalg.norm(
            np.repeat(source, target.shape[2], axis=2)
            - np.tile(target, (1, source.shape[2])),
            axis=1,
        )
        indexes = np.argmin(
            d.reshape(_batch_size, source.shape[2], target.shape[2]), axis=2
        )

        return target[:, :, indexes][:, :, 0, :], source


class IterativeClosetPoint:
    """
    The task ot ICT is to find a rotation and translation matrix, which, when applied on source gives minimum error
    in simple terms transform source to target by some rotation (R) and translation (T)

    Algorithm
        a) Potentially SubSample Points
        b) Determine Corresponding Points (ASSOCIATION)
        c) Weight or reject pairs (FILTERING)
        d) Compute rotation (R) and translation (T)
        e) Apply rotation (R) and translation (T) on source
        f) Compute error E(R, t)
        g) While error decreased and error > error_tolerance
            1) Repeat from (b)
    """

    def __init__(self):
        self._homogeneous_matrix = None

        self._batch_size = None
        self._dimension = None

    def _update_homogeneous_matrix(self, homogeneous_matrix: np.ndarray):
        """
        :param homogeneous_matrix:
        :return:
        """

        if self._homogeneous_matrix is None:
            self._homogeneous_matrix = homogeneous_matrix
        else:
            self._homogeneous_matrix = np.tensordot(
                self._homogeneous_matrix, homogeneous_matrix, axes=([-1], [1])
            )[:, :, 0, :]

    @staticmethod
    def _error(target: np.ndarray, source: np.ndarray) -> np.ndarray:
        """

        :param target:
        :param source:
        :return:
        """
        assert type(target) is np.ndarray and type(source) is np.ndarray, (
            "Expected 'target' and 'source' to have type 'np.ndarray'" "got %s, %s",
            (
                type(target),
                type(source),
            ),
        )
        return np.mean(np.linalg.norm((target - source), axis=1), axis=-1)

    @staticmethod
    def _rotation(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """

        :param u:
        :param v:
        :return:
        """
        assert type(u) is np.ndarray and type(v) is np.ndarray, (
            "Expected 'u' and 'v' to have type 'np.ndarray'" "got %s, %s",
            (
                type(u),
                type(v),
            ),
        )
        return (u @ v).transpose((0, 2, 1))

    @staticmethod
    def _translation(
        rotation: np.ndarray, source_mean: np.ndarray, target_mean: np.ndarray
    ) -> np.ndarray:
        """

        :param rotation:
        :param source_mean:
        :param target_mean:
        :return:
        """

        return (
            target_mean - np.tensordot(rotation, source_mean, axes=([-1], [1]))[:, :, 0]
        )

    def _apply_transformation(
        self, input_array: np.ndarray, homogeneous_matrix: np.ndarray
    ) -> np.ndarray:
        """

        :param homogeneous_matrix:
        :param input_array:
        :return:
        """
        assert (
            type(input_array) is np.ndarray and type(homogeneous_matrix) is np.ndarray
        ), (
            "Expected 'input_array' and 'homogeneous_matrix' to have type 'np.ndarray'"
            "got %s, %s",
            (
                type(input_array),
                type(homogeneous_matrix),
            ),
        )
        dup = np.ones((self._batch_size, self._dimension + 1, input_array.shape[-1]))
        dup[:, : self._dimension, :] = input_array
        return np.tensordot(homogeneous_matrix, dup, axes=([-1], [1]))[
            :, : self._dimension, 0, :
        ]

    @staticmethod
    def _cross_covariance_matrix(target: np.ndarray, source: np.ndarray) -> np.ndarray:
        """

        :param target:
        :param source:
        :return:
        """
        assert type(target) is np.ndarray and type(source) is np.ndarray, (
            "Expected 'target' and 'source' to have type 'np.ndarray'" "got %s, %s",
            (
                type(target),
                type(source),
            ),
        )
        return np.tensordot(source, target.T, axes=([-1], [0]))[:, :, :, 0]

    def _generate_homogeneous_matrix(
        self, rotation: np.ndarray, translation: np.ndarray
    ) -> np.ndarray:
        """

        :param rotation:
        :param translation:
        :return:
        """
        assert type(rotation) is np.ndarray and type(translation) is np.ndarray, (
            "Expected 'rotation' and 'translation' to have type 'np.ndarray'"
            "got %s, %s",
            (
                type(rotation),
                type(translation),
            ),
        )
        iteration_homogeneous_matrix = np.identity(self._dimension + 1)[
            np.newaxis, :, :
        ].repeat(self._batch_size, axis=0)

        iteration_homogeneous_matrix[:, : self._dimension, : self._dimension] = rotation
        iteration_homogeneous_matrix[
            :, : self._dimension, self._dimension
        ] = translation

        return iteration_homogeneous_matrix

    @staticmethod
    def _shift(input_points: np.ndarray, input_mean: np.ndarray) -> np.ndarray:
        """

        :param input_points:
        :param input_mean:
        :return:
        """
        assert type(input_points) is np.ndarray and type(input_mean) is np.ndarray, (
            "Expected 'input_points' and 'input_mean' to have type 'np.ndarray'"
            "got %s, %s",
            (
                type(input_points),
                type(input_mean),
            ),
        )
        return input_points - input_mean[:, :, np.newaxis]

    def _compute_homogeneous_matrix(
        self, target: np.ndarray, source: np.ndarray
    ) -> np.ndarray:
        """

        :param target:
        :param source:
        :return:
        """

        assert type(target) is np.ndarray and type(source) is np.ndarray, (
            "Expected 'target' and 'source' to have type 'np.ndarray'" "got %s, %s",
            (
                type(target),
                type(source),
            ),
        )

        target_mean = compute_center_of_mass(target, axis=2)
        source_mean = compute_center_of_mass(source, axis=2)

        target_shift = self._shift(target, target_mean)
        source_shift = self._shift(source, source_mean)

        cross_covariance_matrix = self._cross_covariance_matrix(
            target_shift, source_shift
        )
        u, s, v = decompose_matrix(cross_covariance_matrix)

        rotation = self._rotation(u, v)
        translation = self._translation(rotation, source_mean, target_mean)

        iteration_homogeneous_matrix = self._generate_homogeneous_matrix(
            rotation, translation
        )

        return iteration_homogeneous_matrix

    def _ict(
        self,
        target: np.ndarray,
        source: np.ndarray,
        iteration: int,
        error_tolerance: float,
        association,
    ):
        """
        target must be of shape [batch, dimension, number of points]
        source must be of shape [batch, dimension, number of points]

        :param association:
        :param target:
        :param source:
        :param iteration:
        :param error_tolerance:
        :return:
        """
        assert type(target) == np.ndarray and type(source) == np.ndarray, (
            "Expected 'target' and 'source' to have type 'np.ndarray'" "got %s and %s",
            (
                type(target),
                type(source),
            ),
        )

        assert len(target.shape) == 3 and len(source.shape) == 3, (
            "Expected 'target' and 'shape' to have shape '[batch, dimension, number_of_coordinates]'"
            "got %s and %s",
            (
                target.shape,
                source.shape,
            ),
        )

        assert target.shape[0] == source.shape[0], (
            "Expected 'target' and 'source' to have same shape number of batch"
            "got %s and %s",
            (
                target.shape[0],
                source.shape[0],
            ),
        )

        assert target.shape[1] == source.shape[1], (
            "Expected 'target' and 'source' to have same shape number of dimension"
            "got %s and %s",
            (
                target.shape[1],
                source.shape[1],
            ),
        )

        previous_mean_batch_error = 0

        for i in range(iteration):
            current_mean_batch_error = np.mean(self._error(target, source))

            associated_target, associated_source = association(
                target, source, batch_size=self._batch_size
            )

            iteration_homogeneous_matrix = self._compute_homogeneous_matrix(
                associated_target, associated_source
            )

            source = self._apply_transformation(source, iteration_homogeneous_matrix)
            self._update_homogeneous_matrix(iteration_homogeneous_matrix)

            refresh_print(
                f"Iteration {i},"
                f" Loss {current_mean_batch_error},"
                f" Running Tolerance {np.abs(previous_mean_batch_error - current_mean_batch_error)}"
            )
            if (
                np.abs(previous_mean_batch_error - current_mean_batch_error)
                < error_tolerance
            ):
                # BREAK WHEN THE ERROR STOPS REDUCING
                break

            previous_mean_batch_error = current_mean_batch_error

        return source, self._homogeneous_matrix

    def ict_with_batch(
        self,
        target: np.ndarray,
        source: np.ndarray,
        iteration: int,
        error_tolerance: float,
        association: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        when ict is to be performed on multiple inputs at once

        this function is used when the input is of form
            source = [
                        [(s_b1_x1, s_b1_y1), (s_b1_x2, s_b1_y2), (s_b1_x3, s_b1_y3), ......., (s_b1_xn, s_b1_yn)]
                        [(s_b2_x1, s_b2_y1), (s_b2_x2, s_b2_y2), (s_b2_x3, s_b2_y3), ......., (s_b2_xn, s_b2_yn)]
                        .....
                        .....
                        [(s_bm_x1, s_bm_y1), (s_bm_x2, s_bm_y2), (s_bm_x3, s_bm_y3), ......., (s_bm_xn, s_bm_yn)]
                    ]

            target = [
                        [(t_b1_x1, t_b1_y1), (t_b1_x2, t_b1_y2), (t_b1_x3, t_b1_y3), ......., (t_b1_xn, t_b1_yn)]
                        [(t_b2_x1, t_b2_y1), (t_b2_x2, t_b2_y2), (t_b2_x3, t_b2_y3), ......., (t_b2_xn, t_b2_yn)]
                        .....
                        .....
                        [(t_bm_x1, t_bm_y1), (t_bm_x2, t_bm_y2), (t_bm_x3, t_bm_y3), ......., (t_bm_xn, t_bm_yn)]
                    ]

            _, _ = ict_with_batch(np.array(target)swapaxes(-1, 1),np.array(source)swapaxes(-1, 1))

            swap_axes is needed cause when list converted to array the shape is (m, n, dim) and the required shape is
            (m, dim, n)


        target must be of shape [batch, dimension, number of points]
        source must be of shape [batch, dimension, number of points]

        :param association:
        :param target:
        :param source:
        :param iteration:
        :param error_tolerance:
        :return:
        """
        assert type(target) == np.ndarray and type(source) == np.ndarray, (
            "Expected 'target' and 'source' to have type 'np.ndarray'" "got %s and %s",
            (
                type(target),
                type(source),
            ),
        )

        assert len(target.shape) == 3 and len(source.shape) == 3, (
            "Expected 'target' and 'shape' to have shape '[batch, dimension, number_of_coordinates]'"
            "got %s and %s",
            (
                target.shape,
                source.shape,
            ),
        )

        assert target.shape[0] == source.shape[0], (
            "Expected 'target' and 'source' to have same shape number of batch"
            "got %s and %s",
            (
                target.shape[0],
                source.shape[0],
            ),
        )
        self._batch_size = target.shape[0]

        assert target.shape[1] == source.shape[1], (
            "Expected 'target' and 'source' to have same shape number of dimension"
            "got %s and %s",
            (
                target.shape[1],
                source.shape[1],
            ),
        )
        self._dimension = target.shape[1]

        assert hasattr(Association, association), (
            "Expected association to be in %s" "got %s",
            (
                inspect.getmembers(Association, predicate=inspect.isfunction),
                association,
            ),
        )

        transformed_source, homogeneous_matrix = self._ict(
            target,
            source,
            iteration,
            error_tolerance,
            getattr(Association, association),
        )
        return transformed_source, homogeneous_matrix

    def ict_without_batch(
        self,
        target: np.ndarray,
        source: np.ndarray,
        iteration: int,
        error_tolerance: float,
        association: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        this function is used when the input is of form
            source = [(s_x1, s_y1), (s_x2, s_y2), (s_x3, s_y3), ......., (s_xn, s_yn)]
            target = [(t_x1, t_y1), (t_x2, t_y2), (t_x3, t_y3), ......., (t_xn, t_yn)]

            _, _ = ict_without_batch(np.array(target)swapaxes(-1, 0),np.array(source)swapaxes(-1, 0))

            swap_axes is needed cause when list converted to array the shape is (n, dim) and the required shape is
            (dim, n)

        target must be of shape [dimension, number of points]
        source must be of shape [dimension, number of points]

        :param association:
        :param target:
        :param source:
        :param iteration:
        :param error_tolerance:
        :return:
        """
        assert type(target) == np.ndarray and type(source) == np.ndarray, (
            "Expected 'target' and 'source' to have type 'np.ndarray'" "got %s and %s",
            (
                type(target),
                type(source),
            ),
        )

        assert len(target.shape) == 2 and len(source.shape) == 2, (
            "Expected 'target' and 'shape' to have shape '[dimension, number_of_coordinates]'"
            "got %s and %s",
            (
                target.shape,
                source.shape,
            ),
        )

        assert target.shape[0] == source.shape[0], (
            "Expected 'target' and 'source' to have same shape number of dimension"
            "got %s and %s",
            (
                target.shape[0],
                source.shape[0],
            ),
        )
        assert hasattr(Association, association), (
            "Expected association to be in %s" "got %s",
            (
                inspect.getmembers(Association, predicate=inspect.isfunction),
                association,
            ),
        )

        self._batch_size = 1
        self._dimension = target.shape[0]

        transformed_source, homogeneous_matrix = self._ict(
            target[np.newaxis, :, :],
            source[np.newaxis, :, :],
            iteration,
            error_tolerance,
            getattr(Association, association),
        )
        return transformed_source[0, :, :], homogeneous_matrix


class SpatialICT(IterativeClosetPoint):
    # TODO nearest neighbour is not a good association to minimize loss, look for a solution which is suitable for
    #  spatial coordinate use case

    def __init__(self, grid: Union[PixelGrid, Grid]):
        super().__init__()
        self.pixel_grid = grid

    def transform_spatial(
        self, input_list: list, homogeneous_matrix: np.ndarray
    ) -> List:
        return self.pixel_grid.spatial_path_from_x_y(
            *zip(
                *self._apply_transformation(
                    np.array(
                        self.pixel_grid.pixel_path_from_x_y(*zip(*input_list))
                    ).swapaxes(-1, 0)[np.newaxis, :, :],
                    homogeneous_matrix,
                )[0, :, :].T
            )
        )

    @staticmethod
    def _map_matched_target_association(
        referenced_data: RoadNetwork,
        source_trace: Traces,
        observation_error: int = 30,
    ) -> Tuple[List, List]:
        """
        This method finds the initial target association since the which source point belong to which target is unknown
        Initial Target association is found by map matching the source with reference (target), so accurate initial
        association is derived between source and target

        :param referenced_data:
        :param source_trace:
        :param observation_error:
        :return:
        """

        assert isinstance(source_trace, Traces), (
            "Expected 'source' to be of instance 'Traces'" "got %s",
            (type(source_trace),),
        )

        assert isinstance(referenced_data, RoadNetwork), (
            "Expected 'target' to be of instance 'RoadNetwork'" "got %s",
            (type(referenced_data),),
        )

        assert type(observation_error) is int, (
            "Expected 'observation_error' to be of type 'int'" "got %s",
            (type(observation_error),),
        )

        matcher = Match(referenced_data, observation_error=observation_error)

        target = list()
        source = list()
        for trace_id, trace in source_trace.items():
            _, _, reference_poi = matcher.match_trace(trace_id, trace)

            if len(reference_poi) > 0:
                source_trace_coordinates = source_trace.trace_point_to_coordinates(
                    trace
                )
                source.append(source_trace_coordinates)

                target.append([(poi.x, poi.y) for poi in reference_poi])

        return target, source

    def ict_map_matched_target(
        self,
        source: Traces,
        target: RoadNetwork,
        iteration: int = 100,
        error_tolerance: float = 0.001,
        association: str = "nearest_neighbour",
        observation_error: int = 30,
    ) -> List:
        """
        This method finds the initial target association since the which source point belong to which target is unknown
        Initial Target association is found by map matching the source with reference (target), so accurate initial
        association is derived between source and target

        :param association:
        :param error_tolerance:
        :param iteration:
        :param source:
        :param target:
        :param observation_error:
        :return:
        """

        assert isinstance(source, Traces), (
            "Expected 'source' to be of instance 'Traces'" "got %s",
            (type(source),),
        )

        assert isinstance(target, RoadNetwork), (
            "Expected 'target' to be of instance 'RoadNetwork'" "got %s",
            (type(target),),
        )

        assert type(iteration) is int, (
            "Expected 'iteration' to be of type 'int'" "got %s",
            (type(iteration),),
        )

        assert type(error_tolerance) is float, (
            "Expected 'error_tolerance' to be of type 'float'" "got %s",
            (type(error_tolerance),),
        )

        assert type(observation_error) is int, (
            "Expected 'observation_error' to be of type 'int'" "got %s",
            (type(observation_error),),
        )

        target_map_matched, source_map_matched = self._map_matched_target_association(
            target, source, observation_error=observation_error
        )

        _, homogeneous_matrix = self.ict_without_batch(
            np.array(
                self.pixel_grid.pixel_path_from_x_y(
                    *zip(*list(itertools.chain(*target_map_matched)))
                )
            ).swapaxes(-1, 0),
            np.array(
                self.pixel_grid.pixel_path_from_x_y(
                    *zip(*list(itertools.chain(*source_map_matched)))
                )
            ).swapaxes(-1, 0),
            iteration,
            error_tolerance,
            association,
        )

        source_transformed = list()

        if len(source_map_matched) > 1:
            for sub_elements in source_map_matched:
                source_transformed.append(
                    self.transform_spatial(sub_elements, homogeneous_matrix)
                )
        else:
            source_transformed.append(
                self.transform_spatial(source_map_matched, homogeneous_matrix)
            )

        return source_transformed

    def ict_map_matched_target_form_data_frame(
        self,
        source: Union[str, GeoDataFrame],
        target: Union[str, GeoDataFrame],
        iteration: int = 100,
        error_tolerance: float = 0.001,
        association: str = "nearest_neighbour",
        observation_error: int = 30,
    ) -> List:
        """
        This method finds the initial target association since the which source point belong to which target is unknown
        Initial Target association is found by map matching the source with reference (target), so accurate initial
        association is derived between source and target

        :param association:
        :param error_tolerance:
        :param iteration:
        :param source:
        :param target:
        :param observation_error:
        :return:
        """

        assert type(iteration) is int, (
            "Expected 'iteration' to be of type 'int'" "got %s",
            (type(iteration),),
        )

        assert type(error_tolerance) is float, (
            "Expected 'error_tolerance' to be of type 'float'" "got %s",
            (type(error_tolerance),),
        )

        assert type(observation_error) is int, (
            "Expected 'observation_error' to be of type 'int'" "got %s",
            (type(observation_error),),
        )

        if isinstance(source, GeoDataFrame) and isinstance(target, GeoDataFrame):
            return self.ict_map_matched_target(
                traces_from_data_frame(source),
                road_network_from_data_frame(target),
                iteration,
                error_tolerance,
                association,
                observation_error,
            )
        elif isinstance(source, str) and isinstance(target, str):
            source = read_data_frame(source)
            target = read_data_frame(target)
            return self.ict_map_matched_target(
                traces_from_data_frame(source),
                road_network_from_data_frame(target),
                iteration,
                error_tolerance,
                association,
                observation_error,
            )
        else:
            raise NotImplementedError(
                "Supported input '[target -> 'str', source -> 'str']' "
                "and '[target -> 'GeoDataFrame', source -> 'GeoDataFrame']'"
                "got target -> %s, source -> %s ",
                (
                    type(target),
                    type(source),
                ),
            )

    def ict_spatial_coordinates(
        self,
        source: list,
        target: list,
        iteration: int = 100,
        error_tolerance: float = 0.001,
        association: str = "nearest_neighbour",
    ) -> List:
        """
        When the association between source and target are known

        This will dissolve all relation and run as a single association, i.e point to point association

        :param source:
        :param target:
        :param iteration:
        :param error_tolerance:
        :param association:
        :return:
        """

        assert type(target) == list and type(source) == list, (
            "Expected 'target' and 'source' to have type 'list'" "got %s and %s",
            (
                type(target),
                type(source),
            ),
        )

        assert type(iteration) is int, (
            "Expected 'iteration' to be of type 'int'" "got %s",
            (type(iteration),),
        )

        assert type(error_tolerance) is float, (
            "Expected 'error_tolerance' to be of type 'float'" "got %s",
            (type(error_tolerance),),
        )

        _, homogeneous_matrix = self.ict_without_batch(
            np.array(
                self.pixel_grid.pixel_path_from_x_y(
                    *zip(*list(itertools.chain(*target)))
                )
            ).swapaxes(-1, 0),
            np.array(
                self.pixel_grid.pixel_path_from_x_y(
                    *zip(*list(itertools.chain(*source)))
                )
            ).swapaxes(-1, 0),
            iteration,
            error_tolerance,
            association,
        )

        source_transformed = list()
        if len(source) > 1:
            for sub_elements in source:
                source_transformed.append(
                    self.transform_spatial(sub_elements, homogeneous_matrix)
                )
        else:
            source_transformed.append(
                self.transform_spatial(source, homogeneous_matrix)
            )

        return source_transformed
