import numpy as np
import argparse
import json
import os.path as osp
import os


def trunet_proposal_gen():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tem_result')
    parser.add_argument('--trunet_result')
    parser.add_argument('--anno_file')
    args = parser.parse_args()
    tem_result_dir, trunet_result_dir = args.tem_result, args.trunet_result
    anno_file = args.anno_file
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
    pass


if __name__ == '__main__':
    trunet_proposal_gen()
