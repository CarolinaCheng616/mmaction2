import os.path as osp
import os

import numpy as np

# from transformers import BertTokenizer
#
from transformers import BertTokenizer
from transformers import AutoModel
from transformers import pipeline
import torch.nn as nn
import torch

# import jieba
# import jieba.posseg as pseg


forbidden_list = ["e", "m", "o", "x", "y", "z"]


# def filter_meaningless_text(text_list, time_array, feature_array):
#     idxes = []
#     filtered_text_list = []
#     for i, text in enumerate(text_list):
#         words = [flag for word, flag in pseg.cut(text)]
#         print(words)
#         # if not all(words):
#         #     print(text)
#         # idxes.append(i)
#         # filtered_text_list.append(text)
#     # idxes = np.array(idxes)
#     # for i, text in enumerate(text_list):
#     #     words = [flag for _, flag in pseg.cut(text)]
#     #     print(words)
#     # return filtered_text_list, time_array[idxes], feature_array[idxes]


def filter_meaningless_text(text_list, time_array, feature_array):
    if len(text_list) == 0:
        return text_list, time_array, feature_array
    idxes = []
    filtered_text_list = []
    for i, text in enumerate(text_list):
        words = [
            flag[0] in forbidden_list and flag != "eng" for word, flag in pseg.cut(text)
        ]
        if not all(words):
            idxes.append(i)
            filtered_text_list.append(text)
    idxes = np.array(idxes)
    return filtered_text_list, time_array[idxes], feature_array[idxes]


def save_denoised_file(new_path, time_array, text_list, save_idx, weight):
    os.makedirs(osp.dirname(new_path), exist_ok=True)
    lines = []
    for i, idx in enumerate(save_idx):
        lines.append(
            str(time_array[idx]) + "#*," + text_list[idx] + "#*," + str(weight[i])
        )
    with open(new_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def modify(file, wfile, new_root):
    new_lines = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = "/".join(line.strip().split("/")[5:])
            new_lines.append("/".join([new_root, line]))
    with open(wfile, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))


def remove_invalid_dm_file(file, wfile):
    new_lines = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if (
                line.endswith("_dm.npz") or line.endswith("_title.npy")
            ) and "bilibili_dm" in line:
                if osp.exists(line):
                    os.remove(line)
            else:
                new_lines.append(line)
    with open(wfile, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))


def remove_invalid_numpy_file(file, wfile):
    new_lines = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.endswith(".txt") and "bilibili_text_feature" in line:
                if osp.exists(line):
                    os.remove(line)
            else:
                new_lines.append(line)
    with open(wfile, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))


def find_invalid_dm_file(file, wfile):
    new_lines = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if (
                line.endswith("_dm.npz") or line.endswith("_title.npy")
            ) and "bilibili_dm" in line:
                new_lines.append(line)
    with open(wfile, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))


def list_possible_invalid_dm_file(file, wfile):
    files = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = osp.dirname(line.strip()).replace(
                "bilibili_dm", "bilibili_intra_denoise/bilibili_dm", 1
            )
            if osp.exists(line):
                files += [osp.join(line, file) for file in os.listdir(line)]
    with open(wfile, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(files)))


def find_duplicated_files(file, dup_file, uniq_file):
    uniq_files = []
    duplicated_files = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line in uniq_files:
                duplicated_files.append(line)
            else:
                uniq_files.append(line)
    with open(dup_file, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(duplicated_files)))
    with open(uniq_file, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(uniq_files)))


class BERT(nn.Module):
    """BERT backbone.
    """

    def __init__(self, pretrained=None, freeze=True):
        super(BERT, self).__init__()
        self.pretrained = pretrained
        self.freeze = freeze
        self.init_weights()

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            self.model = AutoModel.from_pretrained(self.pretrained).to("cuda")
            self.model.train()
        else:
            raise TypeError("pretrained must be a str")

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                text_out = self.model(**x).pooler_output
        else:
            text_out = self.model(**x).pooler_output
        return text_out


def get_feature(file):
    bert_path = "/mnt/lustre/chenghaoyue/projects/mmaction2/work_dirs/bert_model"
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    bert = BERT(bert_path)
    ori_root = "bilibili_intra_denoise"
    feature_root = "bilibili_intra_denoise_feature"
    from mmcv import ProgressBar

    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        pb = ProgressBar(len(lines))
        pb.start()
        for line in lines:
            dm_path = line.strip()
            new_path = (
                osp.splitext(dm_path.replace(ori_root, feature_root, 1))[0] + "_dm.npz"
            )
            if osp.exists(new_path):
                continue
            os.makedirs(osp.dirname(new_path), exist_ok=True)

            text_list = []
            time_array = []
            with open(dm_path, "r", encoding="utf-8") as dm_file:
                for dm in dm_file:
                    try:
                        tokens = dm.strip().split("#*,")
                        time = float(tokens[0])
                        text = tokens[1]
                        time_array.append(time)
                        text_list.append(text)
                    except (ValueError, IndexError):
                        pass
            time_array = np.array(time_array)

            number_per_iter = 200
            nums = (len(text_list) + number_per_iter - 1) // number_per_iter
            features = []
            for i in range(nums):
                sub_dm = text_list[i * number_per_iter : (i + 1) * number_per_iter]
                sub_tokens = tokenizer(
                    sub_dm, truncation=True, padding="max_length", return_tensors="pt"
                )
                for key in sub_tokens:
                    sub_tokens[key] = sub_tokens[key].cuda()
                sub_feat = bert(sub_tokens).cpu().numpy()

                features.append(sub_feat)
            if len(features) > 0:
                features = np.concatenate(features, axis=0)
            else:
                features = np.array(features)
            np.savez(new_path, times=time_array, features=features)
            pb.update()


def test_feature_distance():
    text_list = ["哈哈哈哈哈哈哈哈哈哈哈哈好", "哈哈哈哈哈哈哈哈哈草", "呵呵哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈", "哈哈哈哈嗝哈哈哈哈哈哈哈哈哈"]
    bert_path = "work_dirs/bert_model"
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    bert = BERT(bert_path)

    number_per_iter = 200
    nums = (len(text_list) + number_per_iter - 1) // number_per_iter
    features = []
    for i in range(nums):
        sub_dm = text_list[i * number_per_iter : (i + 1) * number_per_iter]
        sub_tokens = tokenizer(
            sub_dm, truncation=True, padding="max_length", return_tensors="pt"
        )
        for key in sub_tokens:
            sub_tokens[key] = sub_tokens[key].cuda()
        sub_feat = bert(sub_tokens).cpu().numpy()

        features.append(sub_feat)
    if len(features) > 0:
        features = np.concatenate(features, axis=0)
    else:
        features = np.array(features)
    from sklearn.metrics.pairwise import (
        cosine_distances,
        euclidean_distances,
        cosine_similarity,
    )
    import pdb

    pdb.set_trace()
    from sklearn.feature_extraction.text import TfidfVectorizer

    sim1 = cosine_similarity(features)
    dis1 = 1 - sim1
    sim2 = np.exp(sim1 / 0.1)
    smin, smax = np.min(sim2), np.max(sim2)
    if smin != smax:
        sim2 = (sim2 - smin) / (smax - smin)
    elif smin != 0:
        sim2 = sim2 / smin
    dis2 = 1 - sim2

    vectorizer = TfidfVectorizer(stop_words=None)
    tf_idf = vectorizer.fit_transform(text_list)
    distance = cosine_distances(tf_idf)


if __name__ == "__main__":
    # dm_file = "data/bilibili/dm_files.txt"
    # dm_dup_file = "data/bilibili/dm_duplicated_files.txt"
    # dm_uniq_file = "data/bilibili/dm_uniq_files.txt"
    # file = "/mnt/lustrenew/DATAshare/bilibili/intra_denoise_files.txt"
    # get_feature(file)
    test_feature_distance()
