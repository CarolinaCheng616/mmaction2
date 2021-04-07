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

import jieba
import jieba.posseg as pseg


forbidden_list = ["e", "m", "o", "x", "y", "z"]


def filter_meaningless_text(text_list, time_array, feature_array):
    idxes = []
    filtered_text_list = []
    for i, text in enumerate(text_list):
        words = [flag for word, flag in pseg.cut(text)]
        print(words)
        # if not all(words):
        #     print(text)
        # idxes.append(i)
        # filtered_text_list.append(text)
    # idxes = np.array(idxes)
    # for i, text in enumerate(text_list):
    #     words = [flag for _, flag in pseg.cut(text)]
    #     print(words)
    # return filtered_text_list, time_array[idxes], feature_array[idxes]


if __name__ == "__main__":
    text_list = ["yeah", "一头牛", "哈哈哈哈哈", "靠", "的"]
    time_array = np.array([1, 2, 3])
    feature_array = np.array([[1, 2], [2, 3], [3, 4]])
    filter_meaningless_text(text_list, time_array, feature_array)
    # print(text_list)
    # print(time_array)
    # print(feature_array)
