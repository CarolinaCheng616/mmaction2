import os.path as osp
import os
import sys
import time

import Levenshtein as ed

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import (
    cosine_distances,
    euclidean_distances,
    cosine_similarity,
)
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

from collections import defaultdict

from transformers import BertTokenizer, AutoModel
import torch.nn as nn
import torch

import argparse

from mmcv import ProgressBar

# from collections import defaultdict

import jieba.posseg as pseg


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
    text_list = [
        "哈哈哈哈哈哈哈哈哈哈哈哈好",
        "哈哈哈哈哈哈哈哈哈草",
        "呵呵哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈",
        "哈哈哈哈嗝哈哈哈哈哈哈哈哈哈",
        "读取每个类别下num个视频的弹幕",
    ]
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
    import Levenshtein as ed
    import jieba.posseg as pseg

    sim1 = cosine_similarity(features)
    dis1 = 1 - sim1
    sim2 = np.exp(sim1 / 0.1)
    smin, smax = np.min(sim2), np.max(sim2)
    if smin != smax:
        sim2 = (sim2 - smin) / (smax - smin)
    elif smin != 0:
        sim2 = sim2 / smin
    dis2 = 1 - sim2

    token_list = []
    for text in text_list:
        words = " ".join([word for word, _ in pseg.cut(text)])
        token_list.append(words)

    vectorizer = TfidfVectorizer(stop_words=None)
    tf_idf = vectorizer.fit_transform(token_list)
    distance = cosine_distances(tf_idf)

    length = len(text_list)
    distance = np.zeros((length, length))
    for i in range(length):
        texti = text_list[i]
        for j in range(i + 1, length):
            distance[i][j] = ed.distance(texti, text_list[j])
    distance = distance + distance.T
    dmin, dmax = np.min(distance), np.max(distance)
    if dmin != dmax:
        distance = (distance - dmin) / (dmax - dmin)
    elif dmin != 0:
        distance = distance / dmin
    import pdb

    pdb.set_trace()


def test_real_exmaple_distance(dm_path):
    text_list = []
    time_array = []
    with open(dm_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                tokens = line.strip().split("#*,")
                time = float(tokens[0])
                text = tokens[1]
                text_list.append(text)
                time_array.append(time)
            except (ValueError, IndexError):
                pass
    time_array = np.array(time_array)

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

    from sklearn.feature_extraction.text import TfidfVectorizer
    import Levenshtein as ed
    import jieba.posseg as pseg
    from sklearn.metrics.pairwise import (
        cosine_distances,
        euclidean_distances,
        cosine_similarity,
    )

    import pdb

    pdb.set_trace()

    sim1 = cosine_similarity(features)
    dis1 = 1 - sim1
    sim2 = np.exp(sim1 / 0.1)
    smin, smax = np.min(sim2), np.max(sim2)
    if smin != smax:
        sim2 = (sim2 - smin) / (smax - smin)
    elif smin != 0:
        sim2 = sim2 / smin
    dis2 = 1 - sim2

    token_list = []
    for text in text_list:
        words = " ".join([word for word, _ in pseg.cut(text)])
        token_list.append(words)

    vectorizer = TfidfVectorizer(stop_words=None)
    tf_idf = vectorizer.fit_transform(token_list)
    distance = cosine_distances(tf_idf)

    length = len(text_list)
    distance = np.zeros((length, length))
    for i in range(length):
        texti = text_list[i]
        for j in range(i + 1, length):
            distance[i][j] = ed.distance(texti, text_list[j])
    distance = distance + distance.T
    dmin, dmax = np.min(distance), np.max(distance)
    if dmin != dmax:
        distance = (distance - dmin) / (dmax - dmin)
    elif dmin != 0:
        distance = distance / dmin


def tf_idf_distance(text_list):
    """
    :param text_list:
    :param metric:      e: Euclidean distance  c: Cosine distance
    :return:
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_distances
    import jieba.posseg as pseg

    token_list = []
    for text in text_list:
        words = " ".join([word for word, _ in pseg.cut(text)]) + " "
        token_list.append(words)
    # token_list = text_list
    vectorizer = TfidfVectorizer(stop_words=None)
    try:
        tf_idf = vectorizer.fit_transform(token_list)
        distance = cosine_distances(tf_idf)
    except ValueError:
        distance = np.ones((len(text_list), len(text_list)))
    return distance


def evaluate_cluster(
    cover_cat_num,
    total_cat_num,
    dm_num,
    total_dm_num,
    cat_distribute,
    num_per_cat,
    num_per_video,
):
    # global num_per_cat
    # 覆盖的类别个数，弹幕总个数，每个类别弹幕数的方差
    # maxnum = 5
    # value1 = np.exp(cover_cat_num / total_cat_num * maxnum)
    # value2 = dm_num
    # dm_num_per_cat = []
    # for cat in cat_distribute:
    #     dm_num_per_cat.append(cat_distribute[cat])
    # dm_num_per_cat = np.array(dm_num_per_cat)
    # var = np.std(dm_num_per_cat)
    # if var < 10:
    #     var = 10
    # value3 = 1 / var
    # # value3 = 1 / (np.std(dm_num_per_cat) + 10) * np.mean(dm_num_per_cat)
    # import pdb
    # pdb.set_trace()
    # return value1 * value2 * value3

    # 直接按照个数来
    dm_num_per_cat = []
    for cat in cat_distribute:
        dm_num_per_cat.append(cat_distribute[cat])
    # dm_num_per_cat = np.array(dm_num_per_cat)
    boderline = num_per_cat * num_per_video // 100
    valid_dm_num_per_cat = []
    for num in dm_num_per_cat:
        if num >= boderline:
            valid_dm_num_per_cat.append(num)
    # return len(valid_dm_num_per_cat) * sum(valid_dm_num_per_cat)
    return len(valid_dm_num_per_cat), sum(valid_dm_num_per_cat)


def analysis_stop_sentenses(file, wfile):
    text_cat_label_list = []
    unique_cats_dict = dict()
    unique_labels_dict = dict()
    # cover_weight = 0.5
    # dm_prop_weight = 0.2
    # var_weight = 0.3
    # threshold = 0.5
    num_per_cat = 50
    num_per_video = 20
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            tokens = line.strip().split("#*,")
            # if i == 0:
            #     num_per_cat, num_per_video = int(tokens[0]), int(tokens[1])
            # else:
            text, cat, label = tokens[0], tokens[1], int(tokens[2])
            text_cat_label_list.append((text, cat, label))
            unique_cats_dict[cat] = True
            if label != -1:
                unique_labels_dict[label] = True

    total_dm_number = len(text_cat_label_list)
    total_cat_number = len(unique_cats_dict.keys())
    total_label_number = len(unique_labels_dict.keys())

    label2dm_idxes = defaultdict(list)
    cat_distribute_dict = defaultdict(dict)
    label_value_dict = defaultdict(float)

    for i, (text, cat, label) in enumerate(text_cat_label_list):
        if label != -1:
            label2dm_idxes[label].append(i)
            if cat not in cat_distribute_dict[label]:
                cat_distribute_dict[label][cat] = 0
            cat_distribute_dict[label][cat] += 1
    for label in cat_distribute_dict.keys():
        cover_cat_num = len(cat_distribute_dict[label])
        dm_num = len(label2dm_idxes[label])
        cat_distribute = cat_distribute_dict[label]
        valid_cat_num, valid_dm_num = evaluate_cluster(
            cover_cat_num,
            total_cat_number,
            dm_num,
            total_dm_number,
            cat_distribute,
            num_per_cat,
            num_per_video,
        )
        if valid_cat_num >= 10:
            label_value_dict[label] = valid_cat_num * valid_dm_num
    label_value = [(label, value) for label, value in label_value_dict.items()]
    labels = np.array([item[0] for item in label_value])
    values = np.array([item[1] for item in label_value])
    idxes = np.argsort(values)[::-1]
    labels = labels[idxes]
    values = values[idxes]
    final_text_list = []
    for i, label in enumerate(labels):
        final_text_list.append(str(values[i]))
        dm_num_per_cluster = len(label2dm_idxes[label])
        sample_num = min(dm_num_per_cluster, 10)
        sample_idxes = np.random.choice(dm_num_per_cluster, sample_num, replace=False)
        for idx in sample_idxes:
            final_text_list.append(text_cat_label_list[label2dm_idxes[label][idx]][0])
        final_text_list.append("\n")
    with open(wfile, "w", encoding="utf-8") as f:
        f.write("\n".join(final_text_list))


if __name__ == "__main__":
    # dm_file = "data/bilibili/dm_files.txt"
    # dm_dup_file = "data/bilibili/dm_duplicated_files.txt"
    # dm_uniq_file = "data/bilibili/dm_uniq_files.txt"
    # file = "/mnt/lustrenew/DATAshare/bilibili/intra_denoise_files.txt"
    # get_feature(file)
    # dm_path = "data/《出发吧师傅》周深亮嗓惊艳众评委，笑出框笑得直不起腰的深深 (P9. 花絮之小机灵鬼：表情包深深).txt"
    # test_real_exmaple_distance(dm_path)
    # tf_idf_distance("草", "草", "笑出强益达")
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--wfile", type=str, required=True)
    args = parser.parse_args()
    file = args.file
    wfile = args.wfile
    analysis_stop_sentenses(file, wfile)
