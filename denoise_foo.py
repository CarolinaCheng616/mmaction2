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
        words = [flag[0] in forbidden_list for word, flag in pseg.cut(text)]
        if not all(words):
            idxes.append(i)
            filtered_text_list.append(text)
    idxes = np.array(idxes)
    return filtered_text_list, time_array[idxes], feature_array[idxes]


if __name__ == "__main__":
    text_list = ["啊啊啊啊啊啊啊", "一头牛", "哈哈哈哈哈"]
    time_array = np.array([1, 2, 3])
    feature_array = np.array([[1, 2], [2, 3], [3, 4]])
    text_list, time_array, feature_array = filter_meaningless_text(
        text_list, time_array, feature_array
    )
    print(text_list)
    print(time_array)
    print(feature_array)
