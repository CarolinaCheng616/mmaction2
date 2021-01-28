import argparse
import json
import os
import os.path as osp
from multiprocessing import Manager, Process

import numpy as np

# from mmaction.localization.tag_utils import *


def multi_statistic(meta, pgm_proposal, iou, dic, if_train=True):
    with open(meta, 'r', encoding='utf-8') as f:
        train = json.load(f)
    train_files = [
        osp.join(pgm_proposal, key + '.csv') for key in train.keys()
    ]
    train_pos, train_neg, train_num = 0, 0, len(train_files)
    for train_file in train_files:
        proposals = np.loadtxt(
            train_file, dtype=np.float32, delimiter=',', skiprows=1)
        match_iou = proposals[:, 5]
        train_pos += np.sum(match_iou > iou)
        train_neg += np.sum(match_iou <= iou)
    if if_train:
        dic['train'] = [train_num, train_pos, train_neg]
    else:
        dic['val'] = [train_num, train_pos, train_neg]


def statistic():
    # 统计proposals在给定的阈值下有多少个正负样本
    # 分训练集和测试集
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train', type=str, help='train_meta.json annotation path')
    parser.add_argument(
        '--val', type=str, help='val_meta.json annotation path')
    parser.add_argument('--pgm_proposal', type=str, help='pgm_proposal path')
    parser.add_argument('--iou', type=float, help='iou threshold')
    args = parser.parse_args()
    train_meta, val_meta, pgm_proposal, iou = args.train, args.val, \
        args.pgm_proposal, args.iou
    dic = Manager().dict()
    proc1 = Process(
        target=multi_statistic, args=(train_meta, pgm_proposal, iou, dic))
    proc2 = Process(
        target=multi_statistic, args=(val_meta, pgm_proposal, iou, dic, False))
    proc1.start()
    proc2.start()
    proc1.join()
    proc2.join()
    train_num, train_pos, train_neg = dic['train']
    val_num, val_pos, val_neg = dic['val']
    print(f'train positive per video: {train_pos / train_num}, '
          f'ratio: {train_pos / (train_pos + train_neg)}')
    print(f'train negative per video: {train_neg / train_num}, '
          f'ratio: {train_neg / (train_pos + train_neg)}')
    print(f'val positive per video: {val_pos / val_num}, '
          f'ratio: {val_pos / (val_pos + val_neg)}')
    print(f'val negative per video: {val_neg / val_num}, '
          f'ratio: {val_neg / (val_pos + val_neg)}')


def train_pem():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float)
    pgm_cmd = 'python tools/bsn_proposal_generation.py ' \
              'configs/localization/bsn/' \
              'bsn_pgm_2000x4096_8x5_trunet_feature.py ' \
              '--mode train'
    os.system(pgm_cmd)
    # pem_cmd = ''


# def test_tag_proposal():

if __name__ == '__main__':
    statistic()
