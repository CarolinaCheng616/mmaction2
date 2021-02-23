import argparse
import os
import os.path as osp

import mmcv
import numpy as np
import torch.multiprocessing as mp

from mmaction.localization import (generate_tag_feature,
                                   generate_tag_proposals,
                                   generate_tag_proposals_offset)


def load_video_infos(ann_file):
    """Load the video annotations.

    Args:
        ann_file (str): A json file path of the annotation file.

    Returns:
        list[dict]: A list containing annotations for videos.
    """
    video_infos = []
    anno_database = mmcv.load(ann_file)
    for video_name in anno_database:
        video_info = anno_database[video_name]
        video_info['video_name'] = video_name
        video_infos.append(video_info)
    return video_infos


def generate_proposals(ann_file, tem_results_dir, pgm_proposals_dir,
                       pgm_proposals_thread, **kwargs):
    """Generate proposals using multi-process.

    Args:
        ann_file (str): A json file path of the annotation file for
            all videos to be processed.
        tem_results_dir (str): Directory to read tem results
        pgm_proposals_dir (str): Directory to save generated proposals.
        pgm_proposals_thread (int): Total number of threads.
        kwargs (dict): Keyword arguments for "generate_candidate_proposals".
    """
    video_infos = load_video_infos(ann_file)
    num_videos = len(video_infos)
    num_videos_per_thread = num_videos // pgm_proposals_thread
    processes = []
    manager = mp.Manager()
    result_dict = manager.dict()
    kwargs['result_dict'] = result_dict
    for tid in range(pgm_proposals_thread - 1):
        tmp_video_list = range(tid * num_videos_per_thread,
                               (tid + 1) * num_videos_per_thread)
        p = mp.Process(
            target=generate_tag_proposals,
            args=(
                tmp_video_list,
                video_infos,
                tem_results_dir,
            ),
            kwargs=kwargs)
        p.start()
        processes.append(p)

    tmp_video_list = range((pgm_proposals_thread - 1) * num_videos_per_thread,
                           num_videos)
    p = mp.Process(
        target=generate_tag_proposals,
        args=(
            tmp_video_list,
            video_infos,
            tem_results_dir,
        ),
        kwargs=kwargs)
    p.start()
    processes.append(p)

    for p in processes:
        p.join()

    # save results
    os.makedirs(pgm_proposals_dir, exist_ok=True)
    prog_bar = mmcv.ProgressBar(num_videos)
    # tmin, tmax, action_score, match_iou, match_ioa
    header = 'tmin,tmax,action_score,match_iou,match_ioa'
    for video_name in result_dict:
        proposals = result_dict[video_name]
        proposal_path = osp.join(pgm_proposals_dir, video_name + '.csv')
        np.savetxt(
            proposal_path,
            proposals,
            header=header,
            delimiter=',',
            comments='')
        prog_bar.update()


def generate_proposals_offset(ann_file, tem_results_dir, pgm_proposals_dir,
                              pgm_proposals_thread, **kwargs):
    """Generate proposals using multi-process.

    Args:
        ann_file (str): A json file path of the annotation file for
            all videos to be processed.
        tem_results_dir (str): Directory to read tem results
        pgm_proposals_dir (str): Directory to save generated proposals.
        pgm_proposals_thread (int): Total number of threads.
        kwargs (dict): Keyword arguments for "generate_candidate_proposals".
    """
    video_infos = load_video_infos(ann_file)
    num_videos = len(video_infos)
    num_videos_per_thread = num_videos // pgm_proposals_thread
    processes = []
    manager = mp.Manager()
    result_dict = manager.dict()
    kwargs['result_dict'] = result_dict
    for tid in range(pgm_proposals_thread - 1):
        tmp_video_list = range(tid * num_videos_per_thread,
                               (tid + 1) * num_videos_per_thread)
        p = mp.Process(
            target=generate_tag_proposals_offset,
            args=(
                tmp_video_list,
                video_infos,
                tem_results_dir,
            ),
            kwargs=kwargs)
        p.start()
        processes.append(p)

    tmp_video_list = range((pgm_proposals_thread - 1) * num_videos_per_thread,
                           num_videos)
    p = mp.Process(
        target=generate_tag_proposals_offset,
        args=(
            tmp_video_list,
            video_infos,
            tem_results_dir,
        ),
        kwargs=kwargs)
    p.start()
    processes.append(p)

    for p in processes:
        p.join()

    # save results
    os.makedirs(pgm_proposals_dir, exist_ok=True)
    prog_bar = mmcv.ProgressBar(num_videos)
    # tmin, tmax, action_score, match_iou, match_ioa, tmin_offset, tmax_offset
    header = 'tmin,tmax,action_score,match_iou,' \
             'match_ioa,tmin_offset,tmax_offset'
    for video_name in result_dict:
        proposals = result_dict[video_name]
        proposal_path = osp.join(pgm_proposals_dir, video_name + '.csv')
        np.savetxt(
            proposal_path,
            proposals,
            header=header,
            delimiter=',',
            comments='')
        prog_bar.update()


def generate_features(ann_file, tem_results_dir, pgm_proposals_dir,
                      pgm_features_dir, pgm_features_thread, **kwargs):
    """Generate proposals features using multi-process.

    Args:
        ann_file (str): A json file path of the annotation file for
            all videos to be processed.
        tem_results_dir (str): Directory to read tem results.
        pgm_proposals_dir (str): Directory to read generated proposals.
        pgm_features_dir (str): Directory to save generated features.
        pgm_features_thread (int): Total number of threads.
        kwargs (dict): Keyword arguments for "generate_bsp_feature".
    """
    video_infos = load_video_infos(ann_file)
    num_videos = len(video_infos)
    num_videos_per_thread = num_videos // pgm_features_thread
    processes = []
    manager = mp.Manager()
    feature_return_dict = manager.dict()
    kwargs['result_dict'] = feature_return_dict
    for tid in range(pgm_features_thread - 1):
        tmp_video_list = range(tid * num_videos_per_thread,
                               (tid + 1) * num_videos_per_thread)
        p = mp.Process(
            target=generate_tag_feature,
            args=(
                tmp_video_list,
                video_infos,
                tem_results_dir,
                pgm_proposals_dir,
            ),
            kwargs=kwargs)
        p.start()
        processes.append(p)
    tmp_video_list = range((pgm_features_thread - 1) * num_videos_per_thread,
                           num_videos)
    p = mp.Process(
        target=generate_tag_feature,
        args=(
            tmp_video_list,
            video_infos,
            tem_results_dir,
            pgm_proposals_dir,
        ),
        kwargs=kwargs)
    p.start()
    processes.append(p)

    for p in processes:
        p.join()

    # save results
    os.makedirs(pgm_features_dir, exist_ok=True)
    prog_bar = mmcv.ProgressBar(num_videos)
    print(f'len(feature_return_dict: {len(feature_return_dict)}\n')
    for video_name in feature_return_dict.keys():
        bsp_feature = feature_return_dict[video_name]
        feature_path = osp.join(pgm_features_dir, video_name + '.npy')
        np.save(feature_path, bsp_feature)
        prog_bar.update()


def parse_args():
    parser = argparse.ArgumentParser(description='Proposal generation module')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--mode', choices=['train', 'test'], help='train or test')
    parser.add_argument(
        '--proposal',
        choices=['no_offset', 'offset'],
        help='whether to generate ground truth offset')
    args = parser.parse_args()
    if args.mode == 'test' and args.proposal == 'offset':
        raise ValueError(
            '--mode and --proposal cannot be test and offset at the same.')
    return args


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    tem_results_dir = cfg.tem_results_dir
    pgm_proposals_dir = cfg.pgm_proposals_dir
    pgm_features_dir = cfg.pgm_features_dir  # noqa
    if args.proposal == 'offset' and cfg.offset is False or args.proposal == 'no_offset' and cfg.offset:
        raise ValueError(
            'non matching config offset and args proposal.'
        )
    if args.mode == 'test':
        ann_file = cfg.ann_file_val
        pgm_features_config = cfg.pgm_features_test_cfg  # noqa
    else:
        ann_file = cfg.ann_file_train
        pgm_features_config = cfg.pgm_features_train_cfg  # noqa
    print('\nBegin Proposal Generation')
    if args.proposal == 'offset':
        generate_proposals_offset(ann_file, tem_results_dir, pgm_proposals_dir,
                                  **cfg.pgm_proposals_cfg)
    else:
        generate_proposals(ann_file, tem_results_dir, pgm_proposals_dir,
                           **cfg.pgm_proposals_cfg)
    print('\nFinish Proposal Generation')
    print('\nBegin Feature Generation')
    # generate_features(ann_file, tem_results_dir, pgm_proposals_dir, pgm_features_dir, **pgm_features_config)  # noqa
    print('\nFinish Feature Generation')


if __name__ == '__main__':
    main()