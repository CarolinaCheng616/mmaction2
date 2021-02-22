import json
import os
import os.path as osp
import argparse

import numpy as np
from matplotlib import pyplot as plt


def draw(direct, train_meta):
    # tem_results 文件所在上级目录  xxx.csv
    tem_results = osp.join(direct, 'tem_results')
    figure_dir = osp.join(direct, 'tem_figure')
    if not osp.exists(figure_dir):
        os.makedirs(figure_dir)
    with open(train_meta, 'r') as f:
        dic = json.load(f)
    csv_files = os.listdir(tem_results)
    files = [file for file in dic.keys() if file + '.csv' in csv_files][:100]
    for file in files:
        info = dic[file]
        file = osp.join(tem_results, file + '.csv')
        result = np.loadtxt(file, dtype=np.float32, delimiter=',', skiprows=1)
        action, start, end = result[:, 0], result[:, 1], result[:, 2]
        length = len(action)
        duration = float(info['duration_second'])
        annos = np.array([anno['segment'] for anno in info['annotations']])
        annos = (annos / duration * length).astype(int)
        # print(annos)
        ann_start, ann_end, ann_action = np.zeros(length), np.zeros(
            length), np.zeros(length)
        # ann_start[np.maximum(0, np.minimum(annos[:, 0], 99))] = 1
        # ann_end[np.maximum(0, np.minimum(annos[:, 1], 99))] = 1
        ann_start[np.clip(annos[:, 0], 0, length - 1)] = 1
        ann_end[np.clip(annos[:, 1], 0, length - 1)] = 1
        # print(ann_start)
        # print(np.clip(annos[:, 0], 0, 99))
        for a in annos:
            ann_action[a[0]:a[1]] = 1
        # if 'activitynet' in direct:
        action_file = osp.join(
            figure_dir,
            osp.splitext(osp.basename(file))[0] + '_action.png')
        plt.figure()
        plt.plot(np.array(range(length)), action)
        plt.plot(np.array(range(length)), ann_action)
        plt.savefig(action_file)

        start_file = osp.join(
            figure_dir,
            osp.splitext(osp.basename(file))[0] + '_start.png')
        plt.figure()
        plt.plot(np.array(range(length)), start)
        plt.plot(np.array(range(length)), ann_start)
        plt.savefig(start_file)

        end_file = osp.join(figure_dir,
                            osp.splitext(osp.basename(file))[0] + '_end.png')
        plt.figure()
        plt.plot(np.array(range(length)), end)
        plt.plot(np.array(range(length)), ann_end)
        plt.savefig(end_file)


if __name__ == '__main__':
    # draw('work_dirs/bsn_2000x4096_8x5_trunet_feature',
    #      'data/TruNet/train_meta.json')
    # draw('work_dirs/bsn_400x100_32x3_20e_activitynet_feature',
    #      'data/ActivityNet/anet_anno_train.json')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    parser.add_argument('--anno', type=str)
    args = parser.parse_args()
    direct, anno = args.dir, args.anno
    draw(direct, anno)
