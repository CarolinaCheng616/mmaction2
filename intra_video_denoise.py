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

from transformers import BertTokenizer, AutoModel
import torch.nn as nn
import torch

from multiprocessing import Process

import jieba.posseg as pseg

from mmcv import ProgressBar


bert_path = "/mnt/lustre/chenghaoyue/projects/mmaction2/work_dirs/bert_model"
# bert_path = "data/bert_model"
tokenizer = None
bert = None
new_root = "/mnt/lustrenew/DATAshare/bilibili/bilibili_intra_denoise"
# new_root = "data/bilibili_intra_denoise"

forbidden_list = ["e", "m", "o", "x", "y", "z"]


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


def save_denoised_file(new_path, time_array, text_list, save_idx, weight):
    os.makedirs(osp.dirname(new_path), exist_ok=True)
    lines = []
    for i, idx in enumerate(save_idx):
        lines.append(
            str(time_array[idx]) + "#*," + text_list[idx] + "#*," + str(weight[i])
        )
    with open(new_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


class DataSet:
    def __init__(self, dm_file, feature_file):
        with open(dm_file, "r", encoding="utf-8") as f:
            self.dm_paths = [line.strip() for line in f]
        with open(feature_file, "r", encoding="utf-8") as f:
            self.feature_paths = [line.strip() for line in f if "_dm.npz" in line]
        self.path_idx = defaultdict(list)
        self.length = 0
        for i, path in enumerate(self.dm_paths):
            self.path_idx[osp.splitext(osp.basename(path))[0]].append(i)
        for i, path in enumerate(self.feature_paths):
            name = osp.basename(path)[: -len("_dm.npz")]
            if len(self.path_idx[name]) == 1:
                self.path_idx[name].append(i)
                self.length += 1
            else:
                del self.path_idx[name]
        self.keys = sorted(list(self.path_idx.keys()))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx1, idx2 = self.path_idx[self.keys[idx]]
        return self.dm_paths[idx1], self.feature_paths[idx2]


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
def get_feature(feature_file, text_list):
    data = np.load(feature_file)
    features = data["features"]
    if len(features) != len(text_list):
        number_per_iter = 500
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
    return features


############################################# filter meaningless text #####################################


def filter_meaningless_text(text_list, time_array, feature_array):
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
        db = DBSCAN(eps=0.4, metric="precomputed", min_samples=1).fit(distance)
        # import pdb
        #
        # pdb.set_trace()
        dic = defaultdict(list)
        for i, label in enumerate(db.labels_):
            if label != -1:
                dic[label].append(i)
        centers = []
        center_weight = []
        for cluster in dic.keys():
            centers.append(*np.random.choice(dic[cluster], 1))
            center_weight.append(len(dic[cluster]))
        centers = np.array(centers)
        center_weight = np.array(center_weight)
        idxes = np.argsort(centers)
        centers = centers[idxes]
        center_weight = center_weight[idxes]
        return centers, center_weight


############################################## main ###########################################################


def multi_cluster(dataset, idxes):
    pb = ProgressBar(len(idxes))
    pb.start()
    for idx in idxes:
        dm_path, feature_path = dataset[idx]

        base_name = osp.splitext(osp.basename(dm_path))[0] + ".txt"
        new_name = "/".join(
            [*dm_path[dm_path.find("bilibili") :].split("/")[1:-1], base_name]
        )
        new_path = osp.join(new_root, new_name)
        if osp.exists(new_path):
            continue

        time_array, text_list = read_dm_file(dm_path)
        feature_array = get_feature(feature_path, text_list)
        text_list, time_array, feature_array = filter_meaningless_text(
            text_list, time_array, feature_array
        )
        centers, center_weight = filter.cluster(text_list, time_array, feature_array)
        save_denoised_file(new_path, time_array, text_list, centers, center_weight)
        pb.update()


if __name__ == "__main__":
    ############################### generate paths file #######################################
    # root1 = "/mnt/lustrenew/DATAshare/bilibili/bilibili_dm"
    # wfile1 = "/mnt/lustre/chenghaoyue/dm_files.txt"
    # # root1 = "/home/chenghaoyue/chenghaoyue/code/mmaction2/data/bilibili_text_feature"
    # # wfile1 = "/home/chenghaoyue/chenghaoyue/code/mmaction2/data/text_feature_files.txt"
    # proc1 = Process(target=read_tree_dir_files_to_file, args=(root1, wfile1))
    # proc1.start()
    # root2 = "/mnt/lustrenew/DATAshare/bilibili/bilibili_text_feature"
    # wfile2 = "/mnt/lustre/chenghaoyue/text_feature_files.txt"
    # # root2 = "/home/chenghaoyue/chenghaoyue/code/mmaction2/data/bilibili_parse_xml"
    # # wfile2 = "/home/chenghaoyue/chenghaoyue/code/mmaction2/data/dm_files.txt"
    # proc2 = Process(target=read_tree_dir_files_to_file, args=(root2, wfile2))
    # proc2.start()
    # proc1.join()
    # proc2.join()

    ####################################  load dataset  ######################################
    # feature_files = "/mnt/lustre/chenghaoyue/text_feature_files.txt"
    # text_files = "/mnt/lustre/chenghaoyue/dm_files.txt"
    feature_files = "/mnt/lustre/chenghaoyue/projects/mmaction2/data/bilibili/text_feature_files.txt"
    text_files = "/mnt/lustre/chenghaoyue/projects/mmaction2/data/bilibili/dm_files.txt"
    dataset = DataSet(text_files, feature_files)

    #################################### cluster ##############################################
    distance_list = [
        "edit_distance",
        "tf_idf_distance",
        "tgap_distance",
        "feature_distance",
    ]
    distance_weight_list = [0.1, 0.15, 0.15, 0.6]
    filter = IntraFilter(distance_list, distance_weight_list)

    proc_num = 16
    procs = []
    data_num_per_proc = (len(dataset) + proc_num - 1) // proc_num
    idxes = list(range(len(dataset)))

    for i in range(proc_num):
        proc = Process(
            target=multi_cluster,
            args=(dataset, idxes[i * data_num_per_proc : (i + 1) * data_num_per_proc]),
        )
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()
