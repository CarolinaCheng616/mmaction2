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
    def __init__(
        self,
        split_idx,
        snippet_length=7,
        pos_neg_ratio=1.0,
        keyshot_proportion=0.15,
        *args,
        **kwargs
    ):
        self.split_idx = split_idx
        self.snippet_length = snippet_length
        self.pos_neg_ratio = pos_neg_ratio
        self.keyshot_proportion = keyshot_proportion
        self.test = kwargs.pop('test', False)
        super().__init__(*args, **kwargs)

    @staticmethod
    def _load_yaml(path):
        with open(path) as f:
            obj = yaml.safe_load(f)
        return obj

    @staticmethod
    def _get_datasets(keys):
        dataset_paths = {str(Path(key).parent) for key in keys}
        datasets = {path: h5py.File(path, "r") for path in dataset_paths}
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
            KnapsackSolver.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER, "test"
        )

        values = list(values)
        weights = list(weights)
        capacity = int(capacity)

        knapsack_solver.Init(values, [weights], [capacity])
        knapsack_solver.Solve()
        packed_items = [
            x for x in range(0, len(weights)) if knapsack_solver.BestSolutionContains(x)
        ]

        return packed_items

    def _get_keyshot_summ(
        self, score, change_points, n_frames, n_frame_per_seg, picks, proportion=0.15
    ):
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
            scores = frame_scores[first : last + 1]
            seg_scores[seg_idx] = int(1000 * scores.mean())

        # Apply knapsack algorithm to find the best shots
        limits = int(n_frames * proportion)
        packed = self._knapsack(seg_scores, n_frame_per_seg, limits)

        # Get key-shot based summary
        # summary = np.zeros(n_frames)
        # for seg_idx in packed:
        #     first, last = change_points[seg_idx]
        #     summary[first: last + 1] = 1
        # within_indicator = np.zeros(n_frames + 1)
        # within_indicator[1:n_frames] = summary[1:] - summary[:-1]
        # within_indicator[0] = summary[0]
        # within_indicator[-1] = 0 - summary[-1]
        # segments = []
        # segment = []
        # for i, indicator in enumerate(within_indicator):
        #     if indicator == 1:
        #         segment.append(i)
        #     elif indicator == -1:
        #         segment.append(i - 1)
        #         segments.append(segment)
        #         segment = []
        # segments = np.array(segments)
        # assert (
        #     segments.shape[1] == 2
        # ), "something wrong in sum_dataset.py: _get_keyshot_summ"
        segments = np.zeros(n_frames, dtype=np.bool)
        for seg_idx in packed:
            first, last = change_points[seg_idx]
            segments[first : last + 1] = True

        return segments

    def load_annotations(self):
        obj = self._load_yaml(self.ann_file)
        split = obj[self.split_idx]
        if self.test:
            keys = split["test_keys"]
        else:
            keys = split["train_keys"]
        keys = [osp.join(self.data_prefix, key) for key in keys]
        datasets = self._get_datasets(keys)
        # trunet like data

        video_infos = list()
        self.video_frames = dict()
        for key in keys:
            video_path = Path(key)
            dataset_name = str(video_path.parent)  # whole path for dataset
            video_name = video_path.name  # video_1
            video_file = datasets[dataset_name][video_name]

            tmp_name = '#'.join([osp.basename(dataset_name), video_name])

            features = video_file["features"][...].astype(np.float32)                # 被采样的那些帧的feature
            change_points = video_file["change_points"][...].astype(np.int32)        # 范围是所有帧,shape=[num, 2]
            n_frames = video_file["n_frames"][...].astype(np.int32)                  # 所有帧的数量
            n_frame_per_seg = video_file["n_frame_per_seg"][...].astype(np.int32)    # 每两个change_points之间的帧数量,shape=(num,)
            picks = video_file["picks"][...].astype(np.int32)                        # 被采样的那些帧的idx

            if not self.test:
                gtscore = video_file["gtscore"][...].astype(np.float32)
                summary = self._get_keyshot_summ(
                    gtscore,
                    change_points,
                    n_frames,
                    n_frame_per_seg,
                    picks,
                    self.keyshot_proportion,
                )
                label_action = np.zeros(len(features))
                for i, pick in enumerate(picks):
                    if summary[pick]:
                        label_action[i] = 1.0
            else:
                summary = None
                if "user_summary" in video_file:
                    summary = video_file["user_summary"][...].astype(np.float32)
                label_action = None

            video_info = dict(
                video_name=tmp_name,
                features=features,
                label_action=label_action,
                segments=summary,
            )
            self.video_frames[tmp_name] = [n_frames, picks]
            video_infos.append(video_info)
        return video_infos

    def prepare_train_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        return self.pipeline(results)

    def dump_results(self, results, out, output_format, version='VERSION 1.3'):
        """Dump data to json/csv files."""
        if output_format == 'json':
            result_dict = self.proposals2json(results)
            mmcv.dump(result_dict, out)
        elif output_format == 'csv':
            os.makedirs(out, exist_ok=True)
            for result in results:
                video_name, outputs = result
                header = f'frame,action,n_frames {self.video_frames[video_name][0]}'
                picks = self.video_frames[video_name][1].reshape(-1, 1).astype(np.int)
                outputs = np.concatenate((picks, outputs), axis=1)
                output_path = osp.join(out, video_name + '.csv')
                np.savetxt(
                    output_path,
                    outputs,
                    header=header,
                    delimiter=',',
                    comments='')
        else:
            raise ValueError(
                f'The output format {output_format} is not supported.')

    def evaluate(self):
        pass
