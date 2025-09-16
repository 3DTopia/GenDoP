from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from stonesoup.types.detection import Detection
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.initiator.simple import SimpleMeasurementInitiator
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.smoother.kalman import KalmanSmoother
from stonesoup.reader.base import DetectionReader
from stonesoup.base import Property
from stonesoup.buffered_generator import BufferedGenerator
from torchtyping import TensorType


# ------------------------------------------------------------------------------------- #

num_frames, num_verts = None, None

# ------------------------------------------------------------------------------------- #

LEFT_VERT_INDEX = 808  # Left hip vertex index
RIGHT_VERT_INDEX = 4297  # Right hip vertex index
FRONT_VERT_INDEX = 331  # Facing direction (331 front nose, 3146 front center hips)

# ------------------------------------------------------------------------------------- #

# https://github.com/movingpandas/movingpandas/blob/main/movingpandas/trajectory_cleaner.py # noqa


def extract_consecutive_pairs(array: List[int]) -> List[Tuple[int, int]]:
    if len(array) == 0:
        return []
    pairs = []
    start = array[0]
    end = array[0]

    for num in array[1:]:
        if num == end + 1:
            end = num
        else:
            pairs.append((start, end))
            start = end = num

    pairs.append((start, end))

    return pairs


def clean_outliers(
    translations: TensorType["num_frames", 3], alpha: float, min_chunk_length: int
) -> List[Tuple[int, int]]:
    # Compute velocities
    velocities = torch.linalg.norm(translations[:-1] - translations[1:], dim=-1)
    # Pad velocities with its first value (for index consistency)
    velocities = torch.cat([velocities[:1], velocities])

    # Compute the velocity threshold based on the 95th percentile
    v_max = alpha * torch.quantile(velocities, 0.95)
    accepted_indices = torch.nonzero(velocities < v_max).squeeze()
    chunk_indices = extract_consecutive_pairs(accepted_indices.numpy())

    chunks = [
        (start_index, end_index)
        for start_index, end_index in chunk_indices
        if end_index - start_index > min_chunk_length
    ]

    return chunks


def join_chunks(
    chunks_1: List[Tuple[int, int]], chunks_2: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    joined_chunks = []
    for start_1, end_1 in chunks_1:
        for start_2, end_2 in chunks_2:
            # Determine the overlap between chunks
            overlap_start = max(start_1, start_2)
            overlap_end = min(end_1, end_2)

            # If there is overlap, add it to the joined_chunks
            if overlap_start <= overlap_end:
                joined_chunks.append((overlap_start, overlap_end))

    return joined_chunks


# ------------------------------------------------------------------------------------- #

# https://github.com/movingpandas/movingpandas/blob/7bc03b76cfd326d8dd3846d4bcd06e593a17f38b/movingpandas/trajectory_smoother.py # noqa


class TrajectorySmoother(ABC):
    """
    TrajectorySmoother base class

    Base class for trajectory smoothers. This class is abstract and thus cannot be
    instantiated.
    """

    def __init__(self, traj):
        """
        Create TrajectorySmoother

        Parameters
        ----------
        traj : Trajectory or TrajectoryCollection
        """
        self.traj = traj

    def smooth(self, **kwargs):
        """
        Smooth the input Trajectory/TrajectoryCollection

        Parameters
        ----------
        kwargs : any type
            Keyword arguments, differs by smoother

        Returns
        -------
        Trajectory/TrajectoryCollection
            Smoothed Trajectory or TrajectoryCollection

        """
        return self._smooth_traj(self.traj, **kwargs)

    @abstractmethod
    def _smooth_traj(self, traj, **kwargs):
        raise NotImplementedError


class KalmanSmootherCV(TrajectorySmoother):
    """
    Smooths using a Kalman Filter with a Constant Velocity model.

    The Constant Velocity model assumes that the speed between consecutive locations is
    nearly constant. For trajectories where ``traj.is_latlon = True`` the smoother
    converts to EPSG:3395 (World Mercator) internally to perform filtering and smoothing

    .. note::
        This class makes use of
        `Stone Soup <https://stonesoup.readthedocs.io/en/latest/>`_, which is an
        optional dependency and not installed by default. To use this class, you need
        to install Stone Soup directly
        (see `here <https://stonesoup.readthedocs.io/en/latest/#installation>`_).
    """

    def smooth(self, process_noise_std=0.5, measurement_noise_std=1):
        """
        Smooth the input Trajectory/TrajectoryCollection

        Parameters
        ----------
        process_noise_std: float or sequence of floats of length 2, default is 0.5.
            The process (acceleration) noise standard deviation.

            If a sequence (e.g. list, tuple, etc.) is provided, the first index is used
            for the x coordinate, while the second is used for the y coordinate. If
            ``traj.is_latlon=True`` the values are applied to the  easting and northing
            coordinate (in EPSG:3395) respectively.

            Alternatively, a single float can be provided, which is assumed to be the
            same for both coordinates.

            This governs the uncertainty associated with the adherence of the new
            (smooth) trajectories to the CV model assumption; higher values relax
            the assumption, therefore leading to less-smooth trajectories,
            and vice-versa.

        measurement_noise_std: float or sequence of floats of length 2, default is 1.
            The measurement noise standard deviation.

            If a sequence (e.g. list, tuple, etc.) is provided, the first index is used
            for the x coordinate, while the second is used for the y coordinate. If
            ``traj.is_latlon=True`` the values are applied to the  easting and northing
            coordinate (in EPSG:3395) respectively.

            Alternatively, a single float can be provided, which is assumed to be the
            same for both coordinates.

            This controls the assumed error in the original trajectories; higher values
            dictate that the original trajectories are expected to be noisier
            (and therefore, less reliable), thus leading to smoother trajectories,
            and vice-versa.
        """
        return super().smooth(
            process_noise_std=process_noise_std,
            measurement_noise_std=measurement_noise_std,
        )

    def _smooth_traj(self, traj, process_noise_std=0.5, measurement_noise_std=1):
        # Get detector
        detector = self._get_detector(traj)

        # Models
        if not isinstance(process_noise_std, (list, tuple, np.ndarray)):
            process_noise_std = [
                process_noise_std,
                process_noise_std,
                process_noise_std,
            ]
        if not isinstance(measurement_noise_std, (list, tuple, np.ndarray)):
            measurement_noise_std = [
                measurement_noise_std,
                measurement_noise_std,
                measurement_noise_std,
            ]
        transition_model = CombinedLinearGaussianTransitionModel(
            [
                ConstantVelocity(process_noise_std[0] ** 2),
                ConstantVelocity(process_noise_std[1] ** 2),
                ConstantVelocity(process_noise_std[2] ** 2),  # New dimension
            ]
        )
        # Map the measurement to the corresponding state dimensions
        measurement_model = LinearGaussian(
            ndim_state=6,  # 3D state vector
            mapping=[0, 2, 4],
            noise_covar=np.diag(
                [
                    measurement_noise_std[0] ** 2,
                    measurement_noise_std[1] ** 2,
                    measurement_noise_std[2] ** 2,
                ]
            ),
        )
        # Predictor and updater
        predictor = KalmanPredictor(transition_model)
        updater = KalmanUpdater(measurement_model)
        # Initiator
        state_vector = StateVector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 3D state vector
        covar = CovarianceMatrix(np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        prior_state = GaussianStatePrediction(state_vector, covar)
        initiator = SimpleMeasurementInitiator(prior_state, measurement_model)
        # Filtering
        track = None
        for i, (timestamp, detections) in enumerate(detector):
            if i == 0:
                tracks = initiator.initiate(detections, timestamp)
                track = tracks.pop()
            else:
                detection = detections.pop()
                prediction = predictor.predict(track.state, timestamp=timestamp)
                hypothesis = SingleHypothesis(prediction, detection)
                posterior = updater.update(hypothesis)
                track.append(posterior)

        # Smoothing
        smoother = KalmanSmoother(transition_model)
        smooth_track = smoother.smooth(track)
        new_traj = np.array(
            [
                [x.state_vector[0], x.state_vector[2], x.state_vector[4]]
                for x in smooth_track
            ]
        )
        return new_traj

    @staticmethod
    def _get_detector(traj):
        class Detector(DetectionReader):
            trajectory: np.ndarray = Property(doc="")

            @BufferedGenerator.generator_method
            def detections_gen(self):
                for time, (x, y, z) in enumerate(traj):
                    t = pd.to_datetime(time, unit="s")
                    detection = Detection([x, y, z], timestamp=t)
                    yield t, {detection}

        return Detector(traj)


def smooth_chunks(
    raw_trans_chunks: List[TensorType["num_frames", 3]],
    process_noise_std: float,
    measurement_noise_std: int,
) -> List[Tuple[Tuple[int, int], TensorType["num_frames", 3]]]:
    smooth_trans_chunks = []
    for chunk_indices, raw_chunk in raw_trans_chunks:
        smooth_trans = KalmanSmootherCV(raw_chunk).smooth(
            process_noise_std, measurement_noise_std
        )
        smooth_trans_chunks.append((chunk_indices, torch.from_numpy(smooth_trans)))

    return smooth_trans_chunks


# ------------------------------------------------------------------------------------- #


def clean_trajectories(
    w2c_poses: TensorType["num_frames", 4, 4],
    alpha: float = 18.0,
    min_chunk_length: int = 5,
    process_noise_std: float = 0.5,
    measurement_noise_std: int = 1,
) -> List[Tuple[Tuple[int, int], TensorType["num_frames", 4, 4]]]:
    # Remove camera outliers
    camera_translations = w2c_poses[:, :3, 3]
    raw_chunk_indices = clean_outliers(camera_translations, alpha, min_chunk_length)

    # Smooth camera trajectory
    camera_trans_chunks = [
        ((start, end), camera_translations[start:end])
        for start, end in raw_chunk_indices
        if end - start > min_chunk_length
    ]
    smoothed_camera_trans_chunks = smooth_chunks(
        camera_trans_chunks, process_noise_std, measurement_noise_std
    )
    smoothed_camera_chunks = []
    for (start, end), smoothed_trans in smoothed_camera_trans_chunks:
        chunk_poses = w2c_poses[start:end].clone()
        chunk_poses[:, :3, 3] = smoothed_trans
        smoothed_camera_chunks.append(((start, end), chunk_poses))

    return smoothed_camera_chunks