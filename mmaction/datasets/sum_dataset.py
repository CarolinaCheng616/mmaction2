import copy
import os
import os.path as osp
from os import PathLike
from pathlib import Path
import warnings

import mmcv
import numpy as np
import h5py
import yaml

from ..core import average_recall_at_avg_proposals
from .base import BaseDataset
from .registry import DATASETS

from ortools.algorithms.pywrapknapsack_solver import KnapsackSolver


@DATASETS.register_module()
class SumDataset(BaseDataset):
    def __init__(self, ann_file, split_idx, pipeline, data_prefix=None, test_mode=False, keyshot_proportion=0.15):
        super().__init__(ann_file, pipeline, data_prefix, test_mode)
        self.split_idx = split_idx
        self.video_paths = self.video_infos
        self.keyshot_proportion = keyshot_proportion

    @staticmethod
    def _load_yaml(path):
        with open(path) as f:
            obj = yaml.safe_load(f)
        return obj

    @staticmethod
    def _get_datasets(keys):
        dataset_paths = {str(Path(key).parent) for key in keys}
        datasets = {path: h5py.File(path, 'r') for path in dataset_paths}
        return datasets

    @staticmethod
    def _knapsack(values, weights, capacity):
        """Solve 0/1 knapsack problem using dynamic programming.

        :param values: Values of each items. Sized [N].
        :param weights: Weights of each items. Sized [N].
        :param capacity: Total capacity of the knapsack.
        :return: List of packed item indices.
        """
        knapsack_solver = KnapsackSolver(
            KnapsackSolver.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER, 'test'
        )

        values = list(values)
        weights = list(weights)
        capacity = int(capacity)

        knapsack_solver.Init(values, [weights], [capacity])
        knapsack_solver.Solve()
        packed_items = [x for x in range(0, len(weights))
                        if knapsack_solver.BestSolutionContains(x)]

        return packed_items

    def _get_keyshot_summ(self, score, change_points, n_frames, n_frame_per_seg, picks, proportion=0.15):
        """Generate keyshot-based video summary i.e. a binary vector.

        :param score: Predicted importance scores.
        :param change_points: Change points, 2D matrix, each row contains a segment.
        :param n_frames: Original number of frames.
        :param n_frame_per_seg: Number of frames per segment.
        :param picks: Positions of subsampled frames in the original video.
        :param proportion: Max length of video summary compared to original length.
        :return: Generated keyshot-based summary.
        """
        assert score.shape == picks.shape
        picks = np.asarray(picks, dtype=np.int32)

        # Get original frame scores from downsampled sequence
        frame_scores = np.zeros(n_frames, dtype=np.float32)
        for i in range(len(picks)):
            pos_lo = picks[i]
            pos_hi = picks[i + 1] if i + 1 < len(picks) else n_frames
            frame_scores[pos_lo:pos_hi] = score[i]

        # Assign scores to video shots as the average of the frames.
        seg_scores = np.zeros(len(change_points), dtype=np.int32)
        for seg_idx, (first, last) in enumerate(change_points):
            scores = frame_scores[first:last + 1]
            seg_scores[seg_idx] = int(1000 * scores.mean())

        # Apply knapsack algorithm to find the best shots
        limits = int(n_frames * proportion)
        packed = self._knapsack(seg_scores, n_frame_per_seg, limits)

        # Get key-shot based summary
        summary = np.zeros(n_frames, dtype=np.bool)
        for seg_idx in packed:
            first, last = change_points[seg_idx]
            summary[first:last + 1] = True

        return summary

    def load_annotations(self):
        obj = self._load_yaml(self.ann_file)
        split = obj[self.split_idx]
        if self.test_mode:
            keys = split['test_keys']
        else:
            keys = split['train_keys']
        datasets = self._get_datasets(keys)
        video_infos = []
        for key in keys:
            video_path = Path(key)
            dataset_name = str(video_path.parent)
            video_name = video_path.name
            video_file = datasets[dataset_name][video_name]

            features = video_file['features'][...].astype(np.float32)
            change_points = video_file['change_points'][...].astype(np.int32)
            n_frames = video_file['n_frames'][...].astype(np.int32)
            n_frame_per_seg = video_file['n_frame_per_seg'][...].astype(np.int32)
            picks = video_file['picks'][...].astype(np.int32)

            if not self.test_mode:
                gtscore = video_file['gtscore'][...].astype(np.float32)
                summary = self._get_keyshot_summ(gtscore, change_points, n_frames, n_frame_per_seg, picks,
                                                 self.keyshot_proportion)
                # segments
            else:
                summary = None
                if 'user_summary' in video_file:
                    summary = video_file['user_summary'][...].astype(np.float32)
                # segments

            # video_info = dict(video_name=key, features=features, segments=segments)
            # video_infos.append(video_info)
        return video_infos

    def prepare_train_frames(self, idx):
        return self.video_infos[idx]

    def prepare_test_frames(self, idx):
        return self.video_infos[idx]

    def evaluate(self):
        pass
