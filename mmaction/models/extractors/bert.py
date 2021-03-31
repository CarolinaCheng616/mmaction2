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
        self.bert = builder.build_backbone(bert_backbone)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.new_path = new_path

    def forward_train(self):
        raise ValueError("training is forbidden.")

    def forward_test(self, times, dms, video_path):
        base_path = osp.basename(video_path).split()[0]
        # for dm
        number_per_iter = 500
        nums = (len(dms) + number_per_iter - 1) // number_per_iter
        for i in range(nums):
            sub_dm = dms[i * number_per_iter: (i + 1) * number_per_iter]
            sub_tokens = self.tokenizer(sub_dm, truncation=True, padding='max_length', return_tensors="pt")
            for key in sub_tokens:
                sub_tokens[key] = sub_tokens[key].cuda()
            sub_feat = self.bert(sub_tokens)

        # for video name
        tokens = self.tokenizer(base_path, truncation=True, padding='max_length', return_tensors='pt')
        for key in tokens:
            tokens[key] = tokens[key].cuda()
        title_feat = self.bert(tokens)
        # save npy file


    def forward(self, times, dms, video_path):

        pass

    # def
