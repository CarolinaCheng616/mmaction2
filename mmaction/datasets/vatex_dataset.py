import os.path as osp
import os

from .base import BaseDataset
from .registry import DATASETS

import json


@DATASETS.register_module()
class VATEXDataset(BaseDataset):
    """Video dataset for action recognition.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    .. code-block:: json

        [{"videoID":"xxx"}, {"videoID":"xxx"}]

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 0.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(
        self,
        ann_file,
        feature_prefix,
        pipeline,
        start_index=0,
        feature_suffix=".npy",
        **kwargs
    ):
        files = ann_file.split(" ")
        self.train_ann_file = files[0]
        self.val_ann_file = files[1]
        self.feature_prefix = feature_prefix
        self.feature_suffix = feature_suffix
        os.makedirs(self.feature_prefix, exist_ok=True)
        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)

    def load_annotations(self):
        """Load annotation file to get video information."""
        with open(self.train_ann_file, "r", encoding="utf-8") as f:
            video_infos = [
                dict(
                    filename=osp.join(self.data_prefix, video_info["videoID"]) + ".mp4",
                    featurepath=osp.join(self.feature_prefix, video_info["videoID"])
                    + self.feature_suffix,
                )
                for video_info in json.load(f)
            ]
        with open(self.val_ann_file, "r", encoding="utf-8") as f:
            video_infos += [
                dict(
                    filename=osp.join(self.data_prefix, video_info["videoID"]) + ".mp4",
                    featurepath=osp.join(self.feature_prefix, video_info["videoID"])
                    + self.feature_suffix,
                )
                for video_info in json.load(f)
            ]
        return video_infos

    def evaluate(
        self,
        results,
        metrics=None,
        metric_options=None,
        logger=None,
        **deprecated_kwargs
    ):
        return 0
