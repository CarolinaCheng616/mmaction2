import os.path as osp
import os
import sys

import Levenshtein as ed

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import (
    cosine_distances,
    euclidean_distances,
    cosine_similarity,
)
from sklearn.cluster import DBSCAN

from collections import defaultdict

from transformers import BertTokenizer

from transformers import BertTokenizer
from transformers import AutoModel
from transformers import pipeline
import torch.nn as nn
import torch

from multiprocessing import Process


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
    return time_list, text_list


# read feature file
def get_feature(feature_file, time_array, text_list):
    data = np.load(feature_file)
    features = data["features"]
    # if len(features) != len(text_list):
    #     tokenizer = BertTokenizer.from_pretrained(bert_path)
    #     features = []
    #     number_per_iter = 500
    #     nums = (len(text_list) + number_per_iter - 1) // number_per_iter
    #     features = []
    #     for i in range(nums):
    #         sub_dm = text_list[i * number_per_iter: (i + 1) * number_per_iter]
    #         sub_tokens = tokenizer(
    #             sub_dm,
    #             truncation=True,
    #             padding="max_length",
    #             return_tensors="pt",
    #         )
    #         for key in sub_tokens:
    #             sub_tokens[key] = sub_tokens[key].cuda()
    #         sub_feat = bert(sub_tokens).cpu().numpy()
    #         features.append(sub_feat)
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


def tf_idf_distance(text_list, metric="c"):
    """
    :param text_list:
    :param metric:      e: Euclidean distance  c: Cosine distance
    :return:
    """
    vectorizer = TfidfVectorizer()
    tf_idf = vectorizer.fit_transform(text_list)
    if metric == "c":
        distance = cosine_distances(tf_idf)
    elif metric == "e":
        distance = euclidean_distances(tf_idf)
    else:
        raise ValueError("metric parameter should be e or c")
    return distance


def tgap_distance(time_array):
    """
    given text time stamp numpy array, return pairwise distance
    :param time_list(float numpy.array):  sorted time stamp list
    :return:
    """
    tmin = time_array[0]
    time_array_copy = time_array - tmin
    distance = abs(time_array_copy.reshape(-1, 1) - time_array_copy.reshape(1, -1))
    tmax = time_array_copy[-1]
    if tmax != 0:
        distance = distance / tmax
    return distance


def feature_distance(feature_array, temperature=0.01):
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
    distance = 1 - similarity
    return distance


############################################# cluster #########################################################


class IntraFilter:
    def __init__(self, distance_list, distance_weight_list):
        self.disfunc_list = []
        for dis in distance_list:
            if not hasattr(sys.modules[__name__], dis):
                raise ValueError(f"no distance function {dis}!")
            self.disfunc_list.append(getattr(sys.modules[__name__], dis))
        self.distance_weight_list = distance_weight_list

    def change_weight_list(self, distance_weight_list):
        self.distance_weight_list = distance_weight_list

    def cluster(self, text_list, time_array=None, feature_array=None):
        distance_list = []
        for dis in self.disfunc_list:
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
        db = DBSCAN(eps=0.5, metric="precomputed", min_samples=1).fit(distance)

        dic = defaultdict(list)
        for i, label in enumerate(db.labels_):
            if label != -1:
                dic[label].append(i)
        centers = []
        center_weight = []
        for cluster in dic.keys():
            centers.append(np.random.choice(dic[cluster], 1))
            center_weight.append(len(dic[cluster]))
        centers = np.array(centers)
        center_weight = np.array(center_weight)
        idxes = np.argsort(centers)
        centers = centers[idxes]
        center_weight = center_weight[idxes]
        return centers, center_weight


############################################## main ###########################################################


if __name__ == "__main__":
    # dm_root = "/mnt/lustre/chenghaoyue/bilibili_dm"                   # absolute path or file
    # feature_root = "/mnt/lustre/share_data/bilibili_text_feature"     # absolute path or file
    root1 = "/mnt/lustre/share_data/bilibili_text_feature"
    wfile1 = "/mnt/lustre/chenghaoyue/text_feature_files.txt"
    proc1 = Process(target=read_tree_dir_files_to_file, args=(root1, wfile1))
    proc1.start()
    # read_tree_dir_files_to_file(root, wfile)
    # dm_file_paths = get_paths(dm_root)
    root2 = "/mnt/lustre/share_data/bilibili_parse_xml"
    wfile2 = "/mnt/lustre/chenghaoyue/dm_files.txt"
    proc2 = Process(target=read_tree_dir_files_to_file, args=(root2, wfile2))
    proc2.start()
    proc1.join()
    proc2.join()
    # read_tree_dir_files_to_file(root, wfile)
