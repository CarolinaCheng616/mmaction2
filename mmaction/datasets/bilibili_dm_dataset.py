import os
import os.path as osp
import copy

import mmcv
import numpy as np

from .registry import DATASETS
from .base import BaseDataset


@DATASETS.register_module()
class DmDataset(BaseDataset):
    def __init__(self, ann_file, pipeline, data_prefix, test_mode=False):
        super().__init__(ann_file, pipeline, data_prefix, test_mode)

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.video_infos)

    def load_annotations(self):
        print("load dataset annotations")
        if self.ann_file is not None and self.data_prefix is None:  # for liwei code
            with open(self.ann_file, "r") as f:
                video_infos = [dict(path=line.strip()) for line in f]
        elif self.ann_file is None and self.data_prefix is not None:  # for haoyue code
            path_list = set()
            path_list.add(self.data_prefix)
            for i in range(3):
                tmp_list = set()
                for path in path_list:
                    for subdir in os.listdir(path):
                        tmp_list.add(osp.join(path, subdir))
                path_list = tmp_list
            video_infos = list()
            for path in path_list:
                files = [dict(path=osp.join(path, file)) for file in os.listdir(path)]
                video_infos += files
        else:
            raise ValueError("something wrong in ann_file and data_prefix")
        return video_infos

    def prepare_train_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        import pdb

        pdb.set_trace()
        results = copy.deepcopy(self.video_infos[idx])
        print("load data")
        return self.pipeline(results)

    def dump_results(self, results, out, output_format):
        pass
