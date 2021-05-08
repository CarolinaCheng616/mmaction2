import copy
import os.path as osp

import numpy as np
import torch

from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module()
class RawframeTextDataset(BaseDataset):
    """Rawframe dataset with text description.

    The dataset loads raw frames and text and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    the directory to frames of a video, total frames of the video and
    text description of a video, which are split with a whitespace.
    Example of a annotation file:

    .. code-block:: txt

        some/directory-1 163 some/text1
        some/directory-2 122 some/text2
        some/directory-3 258 some/text3
        some/directory-4 234 some/text4
        some/directory-5 295 some/text5
        some/directory-6 121 some/text6

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str | None): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        filename_tmpl (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
            Default: 'RGB'.
    """

    def __init__(
        self,
        ann_file,
        pipeline,
        data_prefix=None,
        test_mode=False,
        filename_tmpl="{:05}.jpg",
        start_index=1,
        modality="RGB",
    ):
        self.filename_tmpl = filename_tmpl
        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode,
            multi_class=False,
            num_classes=None,
            start_index=start_index,
            modality=modality,
            sample_by_class=False,
            power=None,
        )

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith(".json"):
            return self.load_json_annotations()
        video_infos = []
        with open(self.ann_file, "r") as fin:
            for line in fin:
                line_split = line.strip().split()
                video_info = {}
                # idx for frame_dir, absolute path
                video_info["frame_dir"] = line_split[0]
                video_info["total_frames"] = int(line_split[1])
                video_info["text_path"] = line_split[2]
                video_infos.append(video_info)

        return video_infos

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results["filename_tmpl"] = self.filename_tmpl
        results["modality"] = self.modality
        results["start_index"] = self.start_index

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results["filename_tmpl"] = self.filename_tmpl
        results["modality"] = self.modality
        results["start_index"] = self.start_index

        return self.pipeline(results)
