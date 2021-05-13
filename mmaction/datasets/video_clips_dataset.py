import copy
import os.path as osp

import numpy as np

# import torch

from .base import BaseDataset
from .registry import DATASETS

# import warnings
# from mmcv.utils import print_log
from ..core import eval_retrieval_metrics


@DATASETS.register_module()
class VideoClipDataset(BaseDataset):
    """VideoClips dataset for matcher.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors.

    The ann_file is a text file with multiple lines, and each line indicates
    the clips of the same video, which are split with a whitespace.
    Example of a annotation file:

    .. code-block:: txt

        some/video1_clip1.avi some/video_1clip2.avi some/video_1clip3.avi
        some/video2_clip1.avi some/video2_clip2.avi some/video2_clip3.avi
        some/video3_clip1.avi some/video3_clip2.avi some/video3_clip3.avi
        some/video4_clip1.avi some/video4_clip2.avi some/video4_clip3.avi
        some/video5_clip1.avi some/video5_clip2.avi some/video5_clip3.avi

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str | None): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 1.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
            Default: 'RGB'.
        power (float | None): We support sampling data with the probability
            proportional to the power of its label frequency (freq ^ power)
            when sampling data. `power == 1` indicates uniformly sampling all
            data; `power == 0` indicates uniformly sampling all classes.
            Default: None.
    """

    def __init__(
        self,
        ann_file,
        pipeline,
        data_prefix=None,
        test_mode=False,
        start_index=0,
        modality="RGB",
        power=None,
    ):
        super().__init__(
            ann_file,
            pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            start_index=start_index,
            modality=modality,
            power=power,
            multi_class=False,
            num_classes=None,
            sample_by_class=False,
        )

    def load_annotations(self):
        """Load annotation file to get video and text information."""
        video_infos = []
        with open(self.ann_file, "r") as fin:
            for line in fin:
                clips = line.strip().split()
                video_info = []
                for clip in clips:
                    video_info.append(clip[2:])
                video_infos.append(video_info)
        return video_infos

    # def evaluate(
    #     self,
    #     results,
    #     metrics=["vt_retrieval_metrics_full", "tv_retrieval_metrics_full"],
    #     logger=None,
    #     **deprecated_kwargs,
    # ):
    #     """Perform evaluation for common datasets.
    #
    #     Args:
    #         results (list): Output results.
    #         metrics (str | sequence[str]): Metrics to be performed.
    #             Defaults: 'top_k_accuracy'.
    #         metric_options (dict): Dict for metric options. Options are
    #             ``topk`` for ``top_k_accuracy``.
    #             Default: ``dict(top_k_accuracy=dict(topk=(1, 5)))``.
    #         logger (logging.Logger | None): Logger for recording.
    #             Default: None.
    #         deprecated_kwargs (dict): Used for containing deprecated arguments.
    #             See 'https://github.com/open-mmlab/mmaction2/pull/286'.
    #
    #     Returns:
    #         dict: Evaluation results dict.
    #     """
    #     # Protect ``metric_options`` since it uses mutable value as default
    #
    #     if deprecated_kwargs != {}:
    #         warnings.warn(
    #             "Option arguments for metrics has been changed to "
    #             "`metric_options`, See 'https://github.com/open-mmlab/mmaction2/pull/286' "  # noqa: E501
    #             "for more details"
    #         )
    #
    #     if not isinstance(results, list):
    #         raise TypeError(f"results must be a list, but got {type(results)}")
    #     assert len(results) == len(self), (
    #         f"The length of results is not equal to the dataset len: "
    #         f"{len(results)} != {len(self)}"
    #     )
    #
    #     v_feat = np.array([result[0] for result in results])
    #     t_feat = np.array([result[1] for result in results])
    #     t_feat = t_feat.reshape(t_feat.shape[0], -1)
    #     eval_results = eval_retrieval_metrics(v_feat, t_feat)
    #     return eval_results

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        clips = copy.deepcopy(self.video_infos[idx])
        img1_dix, img2_idx = np.random.choice(len(clips), 2, replace=False)
        img1_path, img2_path = clips[img1_dix], clips[img2_idx]

        results = dict()
        results1, results2 = dict(), dict()

        results1["filename"] = osp.join(self.data_prefix, img1_path)
        results1["start_index"] = self.start_index
        results1["modality"] = self.modality

        results2["filename"] = osp.join(self.data_prefix, img2_path)
        results2["start_index"] = self.start_index
        results2["modality"] = self.modality

        results["imgs1"] = self.pipeline(results1)["imgs"]
        results["imgs2"] = self.pipeline(results2)["imgs"]

        return results

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        clips = copy.deepcopy(self.video_infos[idx])
        img1_dix, img2_idx = np.random.choice(len(clips), 2, replace=False)
        img1_path, img2_path = clips[img1_dix], clips[img2_idx]

        results = dict()
        results1, results2 = dict(), dict()

        results1["filename"] = osp.join(self.data_prefix, img1_path)
        results1["start_index"] = self.start_index
        results1["modality"] = self.modality

        results2["filename"] = osp.join(self.data_prefix, img2_path)
        results2["start_index"] = self.start_index
        results2["modality"] = self.modality

        results["imgs1"] = self.pipeline(results1)["imgs"]
        results["imgs2"] = self.pipeline(results2)["imgs"]

        return results
