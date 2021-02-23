import os
import os.path as osp
import argparse
import json
import numpy as np
import mmcv


def restrict_proposal_length():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proposal_dir')
    parser.add_argument('--new_proposal_dir')
    parser.add_argument('--anno_file')
    parser.add_argument('--feature_dir', default=None)
    parser.add_argument('--new_feature_dir', default=None)
    args = parser.parse_args()
    proposal_dir, result_dir = args.proposal_dir, args.new_proposal_dir
    anno_file, feature_dir, new_feature_dir = args.anno_file, args.feature_dir, args.new_feature_dir
    proposal_proportion = 3 / 80
    length_range = (proposal_proportion / 4, proposal_proportion * 4)
    os.makedirs(result_dir, exist_ok=True)
    if new_feature_dir is not None:
        os.makedirs(new_feature_dir, exist_ok=True)
    with open(anno_file, 'r') as f:
        dic = json.load(f)
    videos = list(dic.keys())
    prog_bar = mmcv.ProgressBar(len(videos))
    prog_bar.start()
    for video in videos:
        video_path = osp.join(proposal_dir, video + '.csv')
        with open(video_path, 'r') as f:
            for header in f:
                header = header.strip()
                break
        proposals = np.loadtxt(video_path, dtype=np.float32, delimiter=',', skiprows=1)
        tgap = proposals[:, 1] - proposals[:, 0]
        # mask = length_range[0] <= tgap <= length_range[1]
        mask1 = tgap >= length_range[0]
        mask2 = tgap <= length_range[1]
        mask = mask1 * mask2
        new_proposals = proposals[mask]
        np.savetxt(osp.join(result_dir, video + '.csv'), new_proposals, header=header, delimiter=',', comments='')
        if feature_dir is not None and new_feature_dir is not None:
            feature = np.load(osp.join(feature_dir, video + '.csv'))
            new_feature = feature[mask]
            np.save(osp.join(new_feature, video + '.csv'))
        prog_bar.update()


if __name__ == '__main__':
    restrict_proposal_length()
