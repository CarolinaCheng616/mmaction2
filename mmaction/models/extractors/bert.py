import torch.nn as nn
import torch
import torch.distributed as dist
import numpy as np
import os
import os.path as osp
import re

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
        print("begin initializing model.")
        self.bert.init_weights()
        print("finish initializing model.")

    def forward_train(self, video_meta):
        for video in video_meta:
            times, dms, path = video["times"], video["dms"], video["path"]
            times = np.array(times)
            base_path = osp.splitext(osp.basename(path))[0]
            pattern = r".*bilibili\w?\/"
            new_path = re.sub(pattern, self.new_path, osp.dirname(path))
            os.makedirs(new_path, exist_ok=True)
            # for dm
            number_per_iter = 500
            nums = (len(dms) + number_per_iter - 1) // number_per_iter
            features = []
            for i in range(nums):
                sub_dm = dms[i * number_per_iter : (i + 1) * number_per_iter]
                sub_tokens = self.tokenizer(
                    sub_dm, truncation=True, padding="max_length", return_tensors="pt"
                )
                for key in sub_tokens:
                    sub_tokens[key] = sub_tokens[key].cuda()
                sub_feat = self.bert(sub_tokens).cpu().numpy()
                features.append(sub_feat)
            if len(features) > 0:
                features = np.concatenate(features, axis=0)
            else:
                features = np.array(features)
            # save npz file
            np.savez(
                osp.join(new_path, base_path + "_dm.npz"),
                times=times,
                features=features,
            )

            # for video name
            tokens = self.tokenizer(
                base_path, truncation=True, padding="max_length", return_tensors="pt"
            )
            for key in tokens:
                tokens[key] = tokens[key].cuda()
            title_feat = self.bert(tokens).cpu().numpy()
            # save npy file
            np.save(osp.join(new_path, base_path + "_title.npy"), title_feat)

        return 0

    def forward_test(self, video_meta):
        for video in video_meta:
            times, dms, path = video["times"], video["dms"], video["path"]
            times = np.array(times)
            base_path = osp.splitext(osp.basename(path))[0]
            pattern = r".*bilibili_parse_xml\/"
            new_dir = re.sub(pattern, self.new_path, osp.dirname(path))
            os.makedirs(new_dir, exist_ok=True)
            # for dm
            dm_file = osp.join(new_dir, base_path + "_dm.npz")
            if not osp.exists(dm_file):
                number_per_iter = 500
                nums = (len(dms) + number_per_iter - 1) // number_per_iter
                features = []
                for i in range(nums):
                    sub_dm = dms[i * number_per_iter : (i + 1) * number_per_iter]
                    sub_tokens = self.tokenizer(
                        sub_dm,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                    )
                    for key in sub_tokens:
                        sub_tokens[key] = sub_tokens[key].cuda()
                    sub_feat = self.bert(sub_tokens).cpu().numpy()
                    features.append(sub_feat)
                if len(features) > 0:
                    features = np.concatenate(features, axis=0)
                else:
                    features = np.array(features)
                # save npz file
                np.savez(dm_file, times=times, features=features)

            # for video name
            title_file = osp.join(new_dir, base_path + "_title.npy")
            if not osp.exists(title_file):
                tokens = self.tokenizer(
                    base_path,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                for key in tokens:
                    tokens[key] = tokens[key].cuda()
                title_feat = self.bert(tokens).cpu().numpy()
                # save npy file
                np.save(title_file, title_feat)

        return None

    def forward(self, video_meta=None, return_loss=False):
        if return_loss:
            return self.forward_train(video_meta)
        else:
            return self.forward_test(video_meta)

    # def
