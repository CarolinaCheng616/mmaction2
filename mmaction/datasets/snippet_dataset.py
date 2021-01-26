import copy
import os
import os.path as osp
# import time
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
        self.snippet_infos, self.pos_snippets, self.neg_snippets = \
            self.load_snippet_annotations()
        if not self.test_mode:
            self.filter_neg()
        else:
            self.filtered_snippet_infos = self.snippet_infos

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.filtered_snippet_infos)

    def load_snippet_annotations(self):
        """Load the annotation according to ann_file into video_infos."""
        snippet_infos = list()
        for v_info in self.video_infos:
            v_id = v_info['video_name']
            video_snippets = list()
            for i in range(
                    -(self.snippet_length // 2),
                    v_info['duration_second'] - (self.snippet_length) // 2):
                # snippet = self._assign(i, i + self.snippet_length, v_info)
                snippet = dict(
                    label_action=0.0,
                    label_start=0.0,
                    label_end=0.0,
                    neg=True,
                    video_name=f'{v_id}_{i}',
                    duration_second=v_info['duration_second'])
                video_snippets.append(snippet)
            self._assign_label(video_snippets, v_info)
            snippet_infos += video_snippets
        pos_snippets = []
        neg_snippets = []
        for snippet in snippet_infos:
            if snippet['neg']:
                neg_snippets.append(snippet)
            else:
                pos_snippets.append(snippet)

        return snippet_infos, pos_snippets, neg_snippets

    def filter_neg(self):
        import pdb
        pdb.set_trace()
        """Filter out too many negative snippets."""
        shuf(self.neg_snippets)
        neg_snippets = self.neg_snippets[:int(
            len(self.pos_snippets) * self.pos_neg_ratio)]
        filtered_snippet_infos = self.pos_snippets + neg_snippets
        shuf(filtered_snippet_infos)
        self.filtered_snippet_infos = sorted(
            filtered_snippet_infos,
            key=lambda x: '_'.join(x['video_name'].split('_')[:-1]))

    def dump_results(self, results, out, output_format, version='VERSION 1.3'):
        """Dump data to json/csv files."""
        # import pdb
        # pdb.set_trace()
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
                output_array = np.vstack([item[1] for item in tmp_list])
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
    def _assign_label(video_snippets, v_info):
        duration = v_info['duration_second']
        assert duration == len(
            video_snippets), 'something wrong in load_snippet_annotations'
        for anno in v_info['annotations']:
            segment = anno['segment']
            start, end = int(round(segment[0])), int(round(segment[1]))
            start = min(max(start, 0), duration - 1)
            end = min(max(end, 0), duration - 1)
            video_snippets[start]['label_start'] = 1.0
            video_snippets[end]['label_end'] = 1.0
            video_snippets[start]['neg'] = False
            video_snippets[end]['neg'] = False
            for i in range(start + 1, end):
                video_snippets[i]['label_action'] = 1.0
                video_snippets[i]['neg'] = False

    @staticmethod
    def _assign(start, end, video_info):
        label = {
            'label_action': 0.0,
            'label_start': 0.0,
            'label_end': 0.0,
            'neg': True
        }
        for anno in video_info['annotations']:
            segment = anno['segment']
            center = (start + end) // 2
            if center == int(round(segment[0])):
                label['label_start'] = 1.0
            elif center == int(round(segment[1])):
                label['label_end'] = 1.0
            elif int(round(segment[0])) < center < int(round(segment[1])):
                label['label_action'] = 1.0
            if any((label['label_start'] != 0., label['label_end'] != 0.,
                    label['label_action'] != 0)):
                label['neg'] = False
                break

        return label

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.filtered_snippet_infos[idx])
        results['data_prefix'] = self.data_prefix
        results['snippet_length'] = self.snippet_length
        return self.pipeline(results)

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.filtered_snippet_infos[idx])
        results['data_prefix'] = self.data_prefix
        results['snippet_length'] = self.snippet_length
        return self.pipeline(results)
