import torch.nn as nn
import torch
import torch.distributed as dist
import numpy as np
import os
import os.path as osp

from ..backbones import bert
from ..registry import EXTRACTORS
from transformers import BertTokenizer
from .. import builder


@EXTRACTORS.register_module()
class BertExtractor(nn.Module):
    def __init__(self, bert_path, bert_backbone, new_path):
        super().__init__()
        self.bert_path = bert_path
        self.bert = builder.build_backbone(bert_backbone)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.new_path = new_path
        self.conv = nn.Conv1d(
            in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, groups=1
        )
        self.bert.init_weights()

    def forward_train(self, times, dms, path, video_meta):
        base_path = osp.basename(path).split()[0]
        # for dm
        import pdb

        pdb.set_trace()
        number_per_iter = 500
        nums = (len(dms) + number_per_iter - 1) // number_per_iter
        for i in range(nums):
            sub_dm = dms[i * number_per_iter : (i + 1) * number_per_iter]
            sub_tokens = self.tokenizer(
                sub_dm, truncation=True, padding="max_length", return_tensors="pt"
            )
            for key in sub_tokens:
                sub_tokens[key] = sub_tokens[key].cuda()
            sub_feat = self.bert(sub_tokens)

        # for video name
        tokens = self.tokenizer(
            base_path, truncation=True, padding="max_length", return_tensors="pt"
        )
        for key in tokens:
            tokens[key] = tokens[key].cuda()
        title_feat = self.bert(tokens)
        # save npy file

        return 0

    def forward_test(self, times, dms, path, video_meta):
        import pdb

        pdb.set_trace()
        # base_path = osp.basename(video_meta["path"]).split()[0]
        base_path = osp.splitext(osp.basename(path))[0]
        # for dm
        number_per_iter = 500
        nums = (len(dms) + number_per_iter - 1) // number_per_iter
        for i in range(nums):
            sub_dm = dms[i * number_per_iter : (i + 1) * number_per_iter]
            sub_tokens = self.tokenizer(
                sub_dm, truncation=True, padding="max_length", return_tensors="pt"
            )
            for key in sub_tokens:
                sub_tokens[key] = sub_tokens[key].cuda()
            sub_feat = self.bert(sub_tokens)

        # for video name
        tokens = self.tokenizer(
            base_path, truncation=True, padding="max_length", return_tensors="pt"
        )
        for key in tokens:
            tokens[key] = tokens[key].cuda()
        title_feat = self.bert(tokens)
        # save npy file

        return None

    def forward(self, times, dms, path, video_meta=None, return_loss=False):
        if return_loss:
            return self.forward_train(times, dms, path, video_meta)
        else:
            return self.forward_test(times, dms, path, video_meta)

    # def
