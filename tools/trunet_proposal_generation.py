import numpy as np
import argparse
import json
import os.path as osp
import os
from matplotlib import pyplot as plt


def trunet_proposal_gen():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tem_result')
    parser.add_argument('--trunet_result')
    parser.add_argument('--anno_file')
    args = parser.parse_args()
    tem_result_dir, trunet_result_dir = args.tem_result, args.trunet_result
    anno_file = args.anno_file
    os.makedirs(trunet_result_dir, exist_ok=True)
    with open(anno_file, 'r') as f:
        dic = json.load(f)
    videos = list(dic.keys())
    for video in videos:
        result_path = osp.join(tem_result_dir, video + '.csv')
        with open(result_path, 'r') as f:
            for header in f:
                break
        result = np.loadtxt(result_path, dtype=np.float32, delimiter=',', skiprows=1)
        category = result[:, :4]  # action, start, end, bg
        new_result = np.zeros(category.shape)
        max_indexes = np.argmax(category, axis=1)
        new_result[np.arange(new_result.shape[0]), max_indexes] = category[np.arange(new_result.shape[0]), max_indexes]
        min_max = result[:, 4:]
        new_result = np.concatenate((new_result, min_max), axis=1)
        new_result_path = osp.join(trunet_result_dir, video + '.csv')
        np.savetxt(
            new_result_path,
            new_result,
            header=header,
            delimiter=',',
            comments='')


def trunet_proposal_visualize():
    # trunet_results 文件所在上级目录  xxx.csv
    # draw('work_dirs/bsn_2000x4096_8x5_trunet_feature',
    #      'data/TruNet/train_meta.json')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir')
    parser.add_argument('--anno')
    args = parser.parse_args()
    direct, train_meta = args.dir, args.anno

    tem_results = osp.join(direct, 'trunet_results')
    figure_dir = osp.join(direct, 'trunet_figure')
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
        ann_start, ann_end, ann_action = np.zeros(length), np.zeros(
            length), np.zeros(length)
        ann_start[np.clip(annos[:, 0], 0, length - 1)] = 1
        ann_end[np.clip(annos[:, 1], 0, length - 1)] = 1
        for a in annos:
            ann_action[a[0]:a[1]] = 1
        action_file = osp.join(
            figure_dir,
            osp.splitext(osp.basename(file))[0] + '_action.png')
        plt.figure()
        plt.plot(np.array(range(length)), action)
        plt.plot(np.array(range(length)), ann_action)
        plt.savefig(action_file)

        # start_file = osp.join(
        #     figure_dir,
        #     osp.splitext(osp.basename(file))[0] + '_start.png')
        # plt.figure()
        # plt.plot(np.array(range(length)), start)
        # plt.plot(np.array(range(length)), ann_start)
        # plt.savefig(start_file)

        # end_file = osp.join(figure_dir,
        #                     osp.splitext(osp.basename(file))[0] + '_end.png')
        # plt.figure()
        # plt.plot(np.array(range(length)), end)
        # plt.plot(np.array(range(length)), ann_end)
        # plt.savefig(end_file)


if __name__ == '__main__':
    trunet_proposal_visualize()
