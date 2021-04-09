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

import jieba.posseg as pseg


bert_path = "/mnt/lustre/chenghaoyue/projects/mmaction2/work_dirs/bert_model"
# bert_path = "work_dirs/bert_model"
tokenizer = None
bert = None
new_root = "/mnt/lustrenew/DATAshare/bilibili/bilibili_intra_denoise"
# new_root = "data/bilibili_intra_denoise"
feature_root = "/mnt/lustrenew/DATAshare/bilibili/bilibili_intra_denoise_feature"
# feature_root = "data/bilibili_intra_denoise_feature"

############################################# init bert ##################################################


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


def init_global():
    global tokenizer, bert
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    bert = BERT(bert_path)


############################################# get file directory ##########################################


def get_paths(ROOT, depth=4):
    path_list = set()
    path_list.add(ROOT)
    for i in range(depth):
        tmp_list = set()
        for path in path_list:
            for subdir in os.listdir(path):
                tmp_list.add(osp.join(path, subdir))
        path_list = tmp_list
    return path_list


def read_tree_dir_files_to_file(path, wfile, depth=4):
    """
    read root dir path in depth and write files path into wfile

    :param path:    root directory
    :param wfile:   write the files in the root directory's depth's into wfile
    :param depth:
    :return:
    """
    path_list = get_paths(path, depth)
    with open(wfile, "w", encoding="utf-8") as f:
        f.write("\n".join(path_list))


# def save_denoised_file(new_path, time_array, text_list, save_idx, weight):
#     os.makedirs(osp.dirname(new_path), exist_ok=True)
#     lines = []
#     for i, idx in enumerate(save_idx):
#         lines.append(
#             str(time_array[idx]) + "#*," + text_list[idx] + "#*," + str(weight[i])
#         )
#     with open(new_path, "w", encoding="utf-8") as f:
#         f.write("\n".join(lines))


def get_cat_videos_dict(dm_file):
    cat_videos = defaultdict(list)
    with open(dm_file, "r", encoding="utf-8") as f:
        for line in f:
            path = line.strip()
            cat = path[
                path.find("bilibili_intra_denoise/") + len("bilibili_intra_denoise/") :
            ].split("/")[0]
            cat_videos[cat].append(path)
    return cat_videos


############################################# read file ###################################################


# read dm file
def read_dm_file(file_name):
    time_list = []
    text_list = []
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split("#*,")
            try:
                time = float(tokens[0])
                text = tokens[1]
                time_list.append(time)
                text_list.append(text)
            except (ValueError, IndexError):
                pass
    return np.array(time_list), text_list


# read feature file
def get_feature_and_save(time_array, text_list, dm_path):
    new_path = osp.splitext(dm_path.replace(new_root, feature_root, 1))[0] + "_dm.npz"
    if osp.exists(new_path):
        features = np.load(new_path)["features"]
        return features
    os.makedirs(osp.dirname(new_path), exist_ok=True)
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
        print("using bert")
        # torch.cuda.empty_cache()

        features.append(sub_feat)
    if len(features) > 0:
        features = np.concatenate(features, axis=0)
    else:
        features = np.array(features)
    np.savez(new_path, times=time_array, features=features)
    return features


############################################# compute distance #############################################


def edit_distance(text_list):
    """
    text pairwise edit distance
    :param text_list:    list of text
    :return:
    """
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
    return distance


def tf_idf_distance(text_list):
    """
    :param text_list:
    :param metric:      e: Euclidean distance  c: Cosine distance
    :return:
    """
    token_list = []
    for text in text_list:
        words = " ".join([word for word, _ in pseg.cut(text)])
        token_list.append(words)
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    try:
        tf_idf = vectorizer.fit_transform(token_list)
        distance = cosine_distances(tf_idf)
    except ValueError:
        distance = np.ones((len(text_list), len(text_list)))
    return distance


def tgap_distance(time_array):
    """
    given text time stamp numpy array, return pairwise distance
    :param time_list(float numpy.array):  sorted time stamp list
    :return:
    """
    time_array_copy = time_array - time_array[0]
    distance = abs(time_array_copy.reshape(-1, 1) - time_array_copy.reshape(1, -1))
    tmax = time_array_copy[-1]
    if tmax != 0:
        distance = distance / tmax
    return distance


def feature_distance(feature_array, temperature=0.1):
    """
    given features array, return cosine distances
    :param feature_array(numpy.array):
    :param temperature(float): exp^(dis/temperature)
    :return:
    """
    similarity = cosine_similarity(feature_array)
    similarity = np.exp(similarity / temperature)
    smin, smax = np.min(similarity), np.max(similarity)
    if smin != smax:
        similarity = (similarity - smin) / (smax - smin)
    elif smin != 0:
        similarity = similarity / smin
    distance = 1 - similarity
    return distance


############################################# cluster #########################################################


class Filter:
    def __init__(self, distance_list, distance_weight_list, num_per_cat, num_per_video):
        self.disfunc_list = []
        for dis in distance_list:
            if not hasattr(sys.modules[__name__], dis):
                raise ValueError(f"no distance function {dis}!")
            self.disfunc_list.append(getattr(sys.modules[__name__], dis))
        self.distance_weight_list = distance_weight_list
        self.num_per_cat = num_per_cat
        self.num_per_video = num_per_video

    def change_weight_list(self, distance_weight_list):
        self.distance_weight_list = distance_weight_list

    def cluster(
        self,
        eps,
        num_samples,
        text_list,
        cat_list,
        write_cluster_file,
        time_array=None,
        feature_array=None,
    ):
        start = time.time()
        distance_list = []
        for i, dis in enumerate(self.disfunc_list):
            if self.distance_weight_list[i] == 0:
                continue
            if dis.__name__ == "edit_distance" or dis.__name__ == "tf_idf_distance":
                distance_list.append(dis(text_list))
            elif dis.__name__ == "tgap_distance":
                distance_list.append(dis(time_array))
            elif dis.__name__ == "feature_distance":
                distance_list.append(dis(feature_array))
        distance = sum(
            [
                dis * weight
                for dis, weight, in zip(distance_list, self.distance_weight_list)
            ]
        )
        # import pdb
        #
        # pdb.set_trace()
        clusters = DBSCAN(eps=eps, metric="precomputed", min_samples=num_samples).fit(
            distance
        )

        # kmeans = KMeans(n_clusters=20, random_state=0).fit(feature_array)

        end = time.time()
        print(f"cluster time: {end - start}")

        lines = [f"{num_per_cat}#*,{num_per_video}"]
        for i, label in enumerate(clusters.labels_):
            lines.append("#*,".join([text_list[i], cat_list[i], str(label)]))
        with open(write_cluster_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


############################################# evaluation ###################################################


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
    boderline = num_per_cat * num_per_video // 50
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
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            tokens = line.strip().split("#*,")
            if i == 0:
                num_per_cat, num_per_video = int(tokens[0]), int(tokens[1])
            else:
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


############################################## main ###########################################################


def collect_by_cat(cat_videos, num_per_cat, num_per_video):
    cat_list = []
    text_list = []
    time_array = []
    feature_array = []
    pb = ProgressBar(len(cat_videos))
    pb.start()
    for cat in cat_videos:
        # paths = cat_videos[cat][:num_per_cat]
        sample_num = min(len(cat_videos[cat]), num_per_cat)
        sample_idxes = np.random.choice(len(cat_videos[cat]), sample_num, replace=False)
        paths = []
        for sample_idx in sample_idxes:
            paths.append(cat_videos[cat][sample_idx])
        for path in paths:
            time, text = read_dm_file(path)
            # feature = get_feature_and_save(time, text, path)
            feature = np.zeros((len(text), 768))
            if len(time) == 0 or len(text) == 0 or len(feature) == 0:
                continue
            assert len(time) == len(text) and len(time) == len(
                feature
            ), f"not match for {path}"
            idxes = np.random.choice(
                len(text), min(num_per_video, len(text)), replace=False
            )
            for idx in idxes:
                text_list.append(text[idx])
            cat_list += [cat] * len(idxes)
            time_array.append(time[idxes])
            feature_array.append(feature[idxes])
        pb.update()
    time_array = np.concatenate(time_array, axis=0)
    feature_array = np.concatenate(feature_array, axis=0)
    return text_list, cat_list, time_array, feature_array


def parse_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument(
        "--num_per_category", type=int, default=50, help="number of videos per category"
    )
    parser.add_argument("--num_per_video", type=int, default=10)
    parser.add_argument("--write_cluster_file", type=str, required=True)
    parser.add_argument("--weight_list", type=float, nargs="+")
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()
    num_per_cat = args.num_per_category
    num_per_video = args.num_per_video
    write_cluster_file = args.write_cluster_file
    weight_list = args.weight_list
    eps = args.eps
    num_samples = args.num_samples
    return num_per_cat, num_per_video, write_cluster_file, weight_list, eps, num_samples


if __name__ == "__main__":
    # path = "data/bilibili_intra_denoise"
    # wfile = "data/intra_denoise_files.txt"
    # read_tree_dir_files_to_file(path, wfile)

    num_per_cat, num_per_video, write_cluster_file, weight_list, eps, num_samples = (
        parse_args()
    )
    # init_global()

    ####################################  load dataset  ######################################
    text_files = "/mnt/lustre/chenghaoyue/projects/mmaction2/data/bilibili/intra_denoise_files.txt"
    # text_files = "data/intra_denoise_files.txt"
    cat_videos = get_cat_videos_dict(text_files)
    text_list, cat_list, time_array, feature_array = collect_by_cat(
        cat_videos, num_per_cat, num_per_video
    )

    #################################### cluster ##############################################
    distance_list = ["edit_distance", "tf_idf_distance", "feature_distance"]
    # distance_weight_list = [0.1, 0.15, 0.75]
    weight_list = np.array(weight_list) / sum(weight_list)
    filter = Filter(distance_list, weight_list, num_per_cat, num_per_video)

    filter.cluster(
        eps,
        num_samples,
        text_list,
        cat_list,
        write_cluster_file,
        time_array,
        feature_array,
    )

    # # #################################### analysis inter noise sentences ##################################
    # analysis_stop_sentenses(
    #     "data/dbscan_clusters2.txt", "data/inter_stop_sentences2.txt"
    # )
