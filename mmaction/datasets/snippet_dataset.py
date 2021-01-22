import copy
import os
import os.path as osp
from random import shuffle as shuf

import mmcv
import numpy as np

from .registry import DATASETS
from .trunet_dataset import TruNetDataset


@DATASETS.register_module()
class SnippetDataset(TruNetDataset):

    def __init__(self, snippet_length=7, pos_neg_ratio=1., *args, **kwargs):
        self.snippet_length = snippet_length
        self.pos_neg_ratio = pos_neg_ratio
        super().__init__(*args, **kwargs)
        # if self.test_mode:
        self.snippet_infos = self.load_snippet_annotations()
        if self.test_mode:
            self.filter_neg()

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.snippet_infos)

    def load_snippet_annotations(self):
        """Load the annotation according to ann_file into video_infos."""
        snippet_infos = list()
        import pdb
        pdb.set_trace()
        # for v_id, v_info in self.video_infos.items():
        for v_info in self.video_infos:
            v_id = v_info['video_name']
            # for i in range(v_info['duration_seconds'] - self.snippet_length):
            for i in range(
                    -(self.snippet_length // 2),
                    v_info['duration_second'] - (self.snippet_length) // 2):
                # snippet_infos[f'{v_id}_{i}'] = self._assign(i, i + self.snippet_length, v_info)  # noqa
                snippet = self._assign(i, i + self.snippet_length, v_info)
                snippet['video_name'] = f'{v_id}_{i}'
                snippet['duration_second'] = v_info['duration_second']
                snippet_infos.append(snippet)

        return snippet_infos

    def filter_neg(self):
        """Filter out too many negative snippets."""
        self.filtered = True
        self.pos_snippets = []
        self.neg_snippets = []
        self.snippet_infos = shuf(self.snippet_infos)
        for snippet in self.snippet_infos:
            if snippet['neg']:
                self.neg_snippets.append(snippet)
            else:
                self.pos_snippets.append(snippet)
        self.neg_snippets = shuf(
            self.neg_snippets
        )[:int(len(self.pos_snippets) * self.pos_neg_ratio)]
        self.snippet_infos = shuf(self.neg_snippets + self.pos_snippets)
        import pdb
        pdb.set_trace()

    def dump_results(self, results, out, output_format, version='VERSION 1.3'):
        """Dump data to json/csv files."""
        if output_format == 'json':
            result_dict = self.proposals2json(results)
            # output_dict = {
            #     # 'version': version,
            #     'results': result_dict
            #     # 'external_data': {}
            # }
            mmcv.dump(result_dict, out)
        elif output_format == 'csv':
            # TODO: add csv handler to mmcv and use mmcv.dump
            os.makedirs(out, exist_ok=True)
            header = 'action,start,end,tmin,tmax'
            from collections import defaultdict
            all_videos = defaultdict(dict)
            for result in results:
                # video_name, outputs = result
                # output_path = osp.join(out, video_name + '.csv')
                # np.savetxt(
                #     output_path,
                #     outputs,
                #     header=header,
                #     delimiter=',',
                #     comments='')
                video_name, outputs = result
                name_tokens = video_name.split('_')
                name, idx = '_'.join(name_tokens[:-1]), int(name_tokens[-1])
                all_videos[name][idx] = outputs
            for name in all_videos:
                tmp_list = sorted([(idx, all_videos[name][idx])
                                   for idx in all_videos[name]],
                                  key=lambda x: x[0])
                output_array = np.array([item[1] for item in tmp_list])
                output_path = osp.join(out, name + '.csv')
                np.savetxt(
                    output_path,
                    output_array,
                    header=header,
                    delimiter=',',
                    comments='')
        else:
            raise ValueError(
                f'The output format {output_format} is not supported.')

    @staticmethod
    def _assign(start, end, video_info):
        label = {
            'label_action': 0.0,
            'label_start': 0.0,
            'label_end': 0.0,
            'neg': True
        }
        for segment in video_info['annotations']:
            center = (start + end) // 2
            if center == int(round(segment[0])):
                label['label_start'] = 1.0
            elif center == int(round(segment[1])):
                label['label_end'] = 1.0
            elif int(round(segment[0])) < center < int(round(segment[1])):
                label['action'] = 1.0
            if any(label['label_start'] != 0., label['label_end'] != 0.,
                   label['label_action'] != 0):
                label['neg'] = False
                break

        return label

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.snippet_infos[idx])
        results['data_prefix'] = self.data_prefix
        results['snippet_length'] = self.snippet_length
        return self.pipeline(results)

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.snippet_infos[idx])
        results['data_prefix'] = self.data_prefix
        results['snippet_length'] = self.snippet_length
        return self.pipeline(results)
