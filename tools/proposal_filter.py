import os
import os.path as osp
import argparse
import json
import numpy as np
import mmcv
from random import shuffle as shuf


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


# def pem_training_pos_neg_ratio(proposal_dir, new_proposal_dir, anno_file,
#                                low_threshold, high_threshold, neg_pos_ratio,
#                                score_idx, file_extend='.csv', feature_dir=None,
#                                new_feature_dir=None):
#     os.makedirs(new_proposal_dir, exist_ok=True)
#     with open(anno_file, 'r') as f:
#         dic = json.load(f)
#     videos = list(dic.keys())
#     for video in videos:
#         proposal_path = osp.join(proposal_dir, video + file_extend)
#         with open(proposal_path, 'r') as f:
#             for header in f:
#                 header = header.strip()
#                 break
#         proposals = np.loadtxt(proposal_path, dtype=np.float32, delimiter=',', skiprows=1)
#         if len(proposals.shape) == 1:
#             proposals = proposals[np.newaxis, :]
#         pos_proposals = proposals[proposals[:, score_idx] >= high_threshold]
#         neg_proposals = proposals[proposals[:, score_idx] <= low_threshold]
#         # shuf(pos_proposals)
#         # shuf(neg_proposals)
#         pos_number, neg_number = max(len(pos_proposals), 1), max(len(neg_proposals), 1)
#         neg_number = max(min(int(pos_number * neg_pos_ratio), neg_number), 1)
#         pos_number = max(min(int(neg_number / neg_pos_ratio), pos_number), 1)
#         pos_proposals = pos_proposals[:pos_number]
#         neg_proposals = neg_proposals[:neg_number]
#         if len(pos_proposals) == 0 and len(neg_proposals) == 0:
#             new_proposals = proposals
#         elif len(pos_proposals) == 0:
#             new_proposals = neg_proposals
#         elif len(neg_proposals) == 0:
#             new_proposals = pos_proposals
#         else:
#             new_proposals = pos_proposals + neg_proposals
#         new_proposal_path = osp.join(new_proposal_dir, video + file_extend)
#         np.savetxt(new_proposal_path, new_proposals, header=header, delimiter=',', comments='')
#         if feature_dir is not None:


def pem_training_pos_neg_ratio():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proposal_dir')
    parser.add_argument('--new_proposal_dir')
    parser.add_argument('--anno_file')
    parser.add_argument('--low_threshold', type=float, default=0.3)
    parser.add_argument('--high_threshold', type=float, default=0.7)
    parser.add_argument('--score_idx', type=int)
    parser.add_argument('--feature_dir', default=None)
    parser.add_argument('--new_feature_dir', default=None)
    args = parser.parse_args()
    proposal_dir = args.proposal_dir
    new_proposal_dir = args.new_proposal_dir
    anno_file = args.anno_file
    low_threshold = args.low_threshold
    high_threshold = args.high_threshold
    score_idx = args.score_idx
    feature_dir = args.feature_dir
    new_feature_dir = args.new_feature_dir
    os.makedirs(new_proposal_dir, exist_ok=True)
    if new_feature_dir is not None:
        os.makedirs(new_feature_dir, exist_ok=True)
    with open(anno_file, 'r') as f:
        dic = json.load(f)
    videos = list(dic.keys())
    prog_bar = mmcv.ProgressBar(len(videos))
    prog_bar.start()
    for video in videos:
        proposal_path = osp.join(proposal_dir, video + '.csv')
        with open(proposal_path, 'r') as f:
            for header in f:
                header = header.strip()
                break
        proposals = np.loadtxt(proposal_path, dtype=np.float32, delimiter=',', skiprows=1)
        if len(proposals.shape) == 1:
            proposals = proposals[np.newaxis, :]
        pos_idx = proposals[:, score_idx] >= high_threshold
        neg_idx = proposals[:, score_idx] <= low_threshold
        pos_proposals = proposals[pos_idx]
        neg_proposals = proposals[neg_idx]
        if len(pos_proposals) == 0 and len(neg_proposals) == 0:
            new_proposals = proposals
        elif len(pos_proposals) == 0:
            new_proposals = neg_proposals
        elif len(neg_proposals) == 0:
            new_proposals = pos_proposals
        else:
            new_proposals = np.concatenate((pos_proposals, neg_proposals), axis=0)
        new_proposal_path = osp.join(new_proposal_dir, video + '.csv')
        np.savetxt(new_proposal_path, new_proposals, header=header, delimiter=',', comments='')
        if feature_dir is not None and new_feature_dir is not None:
            feature_path = osp.join(feature_dir, video + '.npy')
            features = np.load(feature_path)
            pos_features = features[pos_idx]
            neg_features = features[neg_idx]
            if len(pos_features) == 0 and len(neg_features) == 0:
                new_features = features
            elif len(pos_features) == 0:
                new_features = neg_features
            elif len(neg_features) == 0:
                new_features = pos_features
            else:
                new_features = np.concatenate((pos_features, neg_features), axis=0)
            new_feature_path = osp.join(new_feature_dir, video + '.npy')
            np.save(new_feature_path, new_features)
        prog_bar.update()


if __name__ == '__main__':
    pem_training_pos_neg_ratio()
