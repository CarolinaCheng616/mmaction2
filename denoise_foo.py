import os.path as osp
import os

import numpy as np

# from transformers import BertTokenizer
#
# from transformers import BertTokenizer
# from transformers import AutoModel
# from transformers import pipeline
# import torch.nn as nn
# import torch

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
            line = line.strip().replace(
                "bilibili_dm", "bilibili_intra_denoise/bilibili_dm", 1
            )
            files += [osp.join(line, file) for file in os.listdir(line)]
    with open(wfile, "w", encoding="utf-8") as f:
        f.write("\n".join(files))


if __name__ == "__main__":
    # dm_file = "data/bilibili/ori_dm_files.txt"
    dm_file = "data/bilibili/invalid_dm_files.txt"
    wfile = "data/bilibili/invalid_dir_dm_files.txt"
    list_possible_invalid_dm_file(dm_file, wfile)
    # find_invalid_dm_file(dm_file, dm_wfile)
    # feature_file = "data/bilibili/text_feature_files.txt"
    # feature_wfile = "data/bilibili/text_feature_files2.txt"
    # remove_invalid_numpy_file(feature_file, feature_wfile)
    # new_path = "/home/chy/projects/mmaction2/test.txt"
    # time_array = np.array([])
    # text_list = []
    # save_idx = time_array
    # weight = time_array
    # save_denoised_file(new_path, time_array, text_list, save_idx, weight)
