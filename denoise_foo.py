import os.path as osp
import os

import numpy as np

from transformers import BertTokenizer

from transformers import BertTokenizer
from transformers import AutoModel
from transformers import pipeline
import torch.nn as nn
import torch


bert_path = ""
tokenizer = None
bert = None


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


def init_bert():
    global tokenizer, bert
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    bert = BERT(bert_path)


def get_bert_feature(text_list):
    number_per_iter = 500
    nums = (len(text_list) + number_per_iter - 1) // number_per_iter
    features = []
    import pdb

    pdb.set_trace()
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


if __name__ == "__main__":
    text_list = [
        "方法过于简单；任务没有创新；单标签分类在应用中不现实(可能是多标签也可能没有任何一种标签)；已有很多数据集是多模态的，给出USV的统计量说明本数据集跟其他数据集在多模态上有很大的优势；"
        "话题识别任务如何帮助处理动作识别任务；其他数据集像Youtube-8M为什么被判定为topical标签；对这六个模块各自起的作用做一个说明；V+A+T已经达到最优了，没有必要用六个模块；"
        "对于数据的选择，依据什么标准选择这些视频来组成数据集；数据集分别有多少个视频含有这六个模块的数据的具体的统计量；提供mean class accuracy的准确定义；"
        "user-generated 和 professionally-generated的界限不明确，Youtube上也有user-generated，抖音上也有专业内容；"
        "标签是non-visual-only的，youtube-8M也有non-visual-only的标签，给出一些non-visual-only的例子；爬虫版权问题；标签体系跟其他数据集的不同；"
        "multi-task跟single classification loss有冲突；多语言，如何处理多种语言的BERT；话题识别与理解视频创作者的意图的关系；"
    ] * 513
