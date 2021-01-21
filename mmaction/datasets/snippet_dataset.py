import copy
from random import shuffle as shuf

from .trunet_dataset import TruNetDataset


class SnippetDataset(TruNetDataset):

    def __init__(self, snippet_length=7, pos_neg_ratio=1., *args, **kwargs):
        self.snippet_length = snippet_length
        self.pos_neg_ratio = pos_neg_ratio
        super().__init__(*args, **kwargs)
        self.snippet_infos = self.load_snippet_annotations()
        self.filter_neg()

    def load_snippet_annotations(self):
        """Load the annotation according to ann_file into video_infos."""
        snippet_infos = list()
        for v_id, v_info in self.video_infos.items():
            for i in range(v_info['duration_seconds'] - self.snippet_length):
                # snippet_infos[f'{v_id}_{i}'] = self._assign(i, i + self.snippet_length, v_info)  # noqa
                snippet = self._assign(i, i + self.snippet_length, v_info)
                snippet['video_name'] = f'{v_id}_{i}'
                snippet_infos.append(snippet)

        return snippet_infos

    def filter_neg(self):
        """Filter out too many negative snippets."""
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
