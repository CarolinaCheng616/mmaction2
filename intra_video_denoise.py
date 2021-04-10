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

# from multiprocessing import Process, Manager
from torch.multiprocessing import Process, Manager, set_start_method

import jieba.posseg as pseg

from mmcv import ProgressBar


bert_path = "work_dirs/bert_model"
# tokenizer = None
# bert = None
forbidden_list = ["e", "m", "o", "x", "y", "z"]


############################################# init bert ##################################################


class BERT(nn.Module):
    """BERT backbone.
    """

    def __init__(self, pretrained):
        super(BERT, self).__init__()
        self.model = AutoModel.from_pretrained(pretrained).to("cuda")
        self.model.eval()

    def forward(self, x):
        with torch.no_grad():
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
    path_list = sorted(get_paths(path, depth))
    with open(wfile, "w", encoding="utf-8") as f:
        f.write("\n".join(path_list))


def save_denoised_file(new_path, text_list, time_array, weight):
    os.makedirs(osp.dirname(new_path), exist_ok=True)
    lines = []
    for i, text in enumerate(text_list):
        lines.append("#*,".join([str(time_array[i]), text, str(weight[i])]))
    with open(new_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_denoised_feature(new_path, time_array, feature_array):
    os.makedirs(osp.dirname(new_path), exist_ok=True)
    np.savez(new_path, times=time_array, features=feature_array)


############################################ dataset ######################################################


class DataSet:
    # def __init__(self, dm_file, start=0, end=-1):
    def __init__(self, dm_file, number=-1):
        with open(dm_file, "r", encoding="utf-8") as f:
            self.dm_paths = [line.strip() for line in f]
        # if end != -1:
        #     self.dm_paths = self.dm_paths[start: end]
        # else:
        #     self.dm_paths = self.dm_paths[start:]
        if number != -1:
            self.dm_paths = self.dm_paths[:number]

        self.length = len(self.dm_paths)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # idx1, idx2 = self.path_idx[self.keys[idx]]
        # return self.dm_paths[idx1], self.feature_paths[idx2]
        return self.dm_paths[idx]


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
            except:
                pass
    return np.array(time_list), text_list


# read feature file
def get_feature(tokenizer, bert, text_list):
    # global tokenizer, bert
    if len(text_list) == 0:
        return np.array([])
    number_per_iter = 100
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
    features = np.concatenate(features, axis=0)
    return features


############################################# filter meaningless text #####################################


def filter_meaningless_text(text_list, time_array):
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
    if len(idxes) == 0:
        return [], np.array([])
    else:
        return filtered_text_list, time_array[idxes]


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


class Cluster:
    def __init__(self, distance_list, distance_weight_list):
        self.disfunc_list = []
        for dis in distance_list:
            if not hasattr(sys.modules[__name__], dis):
                raise ValueError(f"no distance function {dis}!")
            self.disfunc_list.append(getattr(sys.modules[__name__], dis))
        self.distance_weight_list = distance_weight_list

    def change_weight_list(self, distance_weight_list):
        self.distance_weight_list = distance_weight_list

    def _cluster(
        self, eps, min_samples, text_list, time_array=None, feature_array=None
    ):
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
        db = DBSCAN(eps=eps, metric="precomputed", min_samples=min_samples).fit(
            distance
        )
        return db

    def cluster(self, eps, min_samples, text_list, time_array=None, feature_array=None):
        db = self._cluster(eps, min_samples, text_list, time_array, feature_array)

        dic = defaultdict(list)
        for i, label in enumerate(db.labels_):
            if label != -1:
                dic[label].append(i)
        return db.labels_, dic  # 每个文本的类别; 按照类别的文本字典


############################################## main ###########################################################


def multi_cluster(
    tokenizer,
    bert,
    dataset,
    idxes,
    intra_denoise_root,
    intra_denoise_feature_root,
    cluster,
    eps,
    min_samples,
    wrong_list,
):
    pb = ProgressBar(len(idxes))
    pb.start()
    for idx in idxes:
        dm_path = dataset[idx]

        new_name = osp.splitext(
            dm_path[dm_path.find("bilibili_dm/") + len("bilibili_dm/") :]
        )[0]
        intra_denoise_path = osp.join(intra_denoise_root, new_name + ".txt")
        intra_denoise_feature_path = osp.join(
            intra_denoise_feature_root, new_name + "_dm.npz"
        )
        if osp.exists(intra_denoise_path) and osp.exists(intra_denoise_feature_path):
            continue

        time_array, text_list = read_dm_file(dm_path)
        text_list, time_array = filter_meaningless_text(text_list, time_array)
        feature_array = get_feature(tokenizer, bert, text_list)
        if len(text_list) != len(time_array) or len(text_list) != len(feature_array):
            wrong_list.append(new_name)
            continue

        # # cluster
        # if len(text_list) != 0:
        #     labels, dic = cluster.cluster(
        #         eps, min_samples, text_list, time_array, feature_array
        #     )
        #     os.makedirs(osp.dirname(intra_denoise_path), exist_ok=True)
        #     with open(intra_denoise_path, "w", encoding="utf-8") as f:
        #         for label in dic:
        #             label_list = dic[label]
        #             for idx in label_list:
        #                 f.write(str(time_array[idx]) + "#*," + text_list[idx] + "\n")
        #             f.write("\n")

        # filter
        centers = []
        centers_weight = []
        if len(text_list) != 0:
            labels, dic = cluster.cluster(
                eps, min_samples, text_list, time_array, feature_array
            )
            for label in dic:
                if label != -1:
                    centers.append(*np.random.choice(dic[label], 1))
                    centers_weight.append(len(dic[label]))
            centers = np.array(centers)
            filtered_text_list = []
            for center in centers:
                filtered_text_list.append(text_list[center])
            text_list = filtered_text_list
            time_array = time_array[centers]
            feature_array = feature_array[centers]
        save_denoised_file(intra_denoise_path, text_list, time_array, centers_weight)
        save_denoised_feature(intra_denoise_feature_path, time_array, feature_array)
        pb.update()


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--start_idx", type=int, default=0)
    # parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--dataset_length", type=str, default=-1)
    parser.add_argument("--distance_weight_list", nargs="+", type=float, required=True)
    parser.add_argument("--intra_denoise_root", type=str, required=True)
    parser.add_argument("--intra_denoise_feature_root", type=str, required=True)
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--min_samples", type=int, required=True)
    parser.add_argument("--process_number", type=int, default=8)
    parser.add_argument(
        "--wrong_log_file",
        type=str,
        default="data/bilibili/wrong_intra_denoise_files.txt",
        help="file to log error dm files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    ################################### parse args ###########################################
    args = parse_args()
    # start_idx = args.start_idx
    # end_idx = args.end_idx
    dataset_length = args.dataset_length
    distance_weight_list = args.distance_weight_list
    distance_weight_list = np.array(distance_weight_list) / sum(distance_weight_list)
    intra_denoise_root = args.intra_denoise_root
    intra_denoise_feature_root = args.intra_denoise_feature_root
    eps = args.eps
    min_samples = args.min_samples
    process_number = args.process_number
    wrong_log_file = args.wrong_log_file

    ############################### generate paths file #######################################
    root2 = "data/bilibili/bilibili_dm"
    wfile2 = "data/bilibili/dm_files.txt"
    read_tree_dir_files_to_file(root2, wfile2)

    ####################################  load dataset  ######################################
    text_files = "data/bilibili/dm_files.txt"
    # dataset = DataSet(text_files, start_idx, end_idx)
    # dataset_length = len(dataset)
    dataset = DataSet(text_files, dataset_length)

    #################################### cluster ##############################################
    distance_list = [
        "edit_distance",
        "tf_idf_distance",
        "tgap_distance",
        "feature_distance",
    ]
    cluster = Cluster(distance_list, distance_weight_list)

    #################################### init bert ###########################################
    set_start_method("spawn")
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    bert = BERT(bert_path)
    bert.share_memory()
    # init_global()

    #################################### multiprocess run ##########################################
    procs = []
    data_num_per_proc = (len(dataset) + process_number - 1) // process_number
    idxes = list(range(len(dataset)))
    wrong_list = Manager().list()
    for i in range(process_number):
        proc = Process(
            target=multi_cluster,
            args=(
                tokenizer,
                bert,
                dataset,
                idxes[i * data_num_per_proc : (i + 1) * data_num_per_proc],
                intra_denoise_root,
                intra_denoise_feature_root,
                cluster,
                eps,
                min_samples,
                wrong_list,
            ),
        )
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()
    # wrong_list = []
    # multi_cluster(dataset, list(range(dataset_length)), intra_denoise_root, intra_denoise_feature_root, cluster, wrong_list)

    if len(wrong_list) > 0:
        with open(wrong_log_file, "w", encoding="utf-8") as f:
            f.write("\n".join(wrong_list))
