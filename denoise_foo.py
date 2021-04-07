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


if __name__ == "__main__":
    # root1 = "/mnt/lustrenew/DATAshare/bilibili/bilibili_dm"
    # # wfile1 = "/mnt/lustre/chenghaoyue/dm_files.txt"
    # # # root1 = "/home/chenghaoyue/chenghaoyue/code/mmaction2/data/bilibili_text_feature"
    # # # wfile1 = "/home/chenghaoyue/chenghaoyue/code/mmaction2/data/text_feature_files.txt"
    # # proc1 = Process(target=read_tree_dir_files_to_file, args=(root1, wfile1))
    # # proc1.start()
    # # root2 = "/mnt/lustrenew/DATAshare/bilibili/bilibili_text_feature"
    # # wfile2 = "/mnt/lustre/chenghaoyue/text_feature_files.txt"
    # # # root2 = "/home/chenghaoyue/chenghaoyue/code/mmaction2/data/bilibili_parse_xml"
    # # # wfile2 = "/home/chenghaoyue/chenghaoyue/code/mmaction2/data/dm_files.txt"
    # # proc2 = Process(target=read_tree_dir_files_to_file, args=(root2, wfile2))
    # # proc2.start()
    # # proc1.join()
    # # proc2.join()
    # # from multiprocessing import Process
    # #
    # # file1 = "/mnt/lustre/chenghaoyue/projects/mmaction2/data/bilibili/dm_files.txt"
    # # wfile1 = "/mnt/lustre/chenghaoyue/projects/mmaction2/data/bilibili/dm_files2.txt"
    # # new_root1 = "/mnt/lustrenew/DATAshare/bilibili/bilibili_dm"
    # # proc1 = Process(target=modify, args=(file1, wfile1, new_root1))
    # # proc1.start()
    # #
    # # file2 = "/mnt/lustre/chenghaoyue/projects/mmaction2/data/bilibili/text_feature_files.txt"
    # # wfile2 = "/mnt/lustre/chenghaoyue/projects/mmaction2/data/bilibili/text_feature_files2.txt"
    # # new_root2 = "/mnt/lustrenew/DATAshare/bilibili/bilibili_text_feature"
    # # proc2 = Process(target=modify, args=(file2, wfile2, new_root2))
    # # proc2.start()
    # #
    # # proc1.join()
    # # proc2.join()
    #
    # text_list = []
    # time_array = np.array([])
    # feature_array = np.array([])
    # filter_meaningless_text(text_list, time_array, feature_array)
    # print(text_list)
    # print(time_array)
    # print(feature_array)
    new_path = "/home/chy/projects/mmaction2/test.txt"
    time_array = np.array([])
    text_list = []
    save_idx = time_array
    weight = time_array
    save_denoised_file(new_path, time_array, text_list, save_idx, weight)
