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

    def forward_train(self):
        raise ValueError("training is forbidden.")
        # return None

    def forward_test(self, times, dms, path):
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

        return None

    def forward(self, times, dms, path, return_loss=False):
        if return_loss:
            return self.forward_train()
        else:
            return self.forward_test(times, dms, path)

    # def
