# import argparse
import copy
import os
import os.path as osp
from multiprocessing import Manager, Process

import mmcv
import numpy as np

from .proposal_utils import temporal_iop, temporal_iou


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


def generate_tag_proposals(video_list,
                           video_infos,
                           tem_results_dir,
                           alpha_list=[.5, .55, .6, .65, .7, .75, .8, .85, .9],
                           beta_list=[0.05, .1, .2, .3, .4, .5, .6, 0.8, 1.0],
                           tem_results_ext='.csv',
                           result_dict=None):
    """Generate Candidate Proposals with given temporal evalutation results.
    Each proposal file will contain:
    'tmin,tmax,action_score,match_iou,match_ioa'.

    Args:
        video_list (list[int]): List of video indexs to generate proposals.
        video_infos (list[dict]): List of video_info dict that contains
            'video_name', 'duration_frame', 'duration_second',
            'feature_frame', and 'annotations'.
        tem_results_dir (str): Directory to load temporal evaluation
            results.
        alpha_list (tuple): The threshold of max action score for label assign.
        beta_list (tuple): The threshold for merge proposals
        tem_results_ext (str): File extension for temporal evaluation
            model output. Default: '.csv'.
        result_dict (dict | None): The dict to save the results. Default: None.

    Returns:
        dict: A dict contains video_name as keys and proposal list as value.
            If result_dict is not None, save the results to it.
    """
    if tem_results_ext != '.csv':
        raise NotImplementedError('Only support csv format now.')

    proposal_dict = {}
    for video_index in video_list:
        video_name = video_infos[video_index]['video_name']
        tem_path = osp.join(tem_results_dir, video_name + tem_results_ext)
        tem_results = np.loadtxt(
            tem_path, dtype=np.float32, delimiter=',', skiprows=1)
        # action, start, end, tmin, tmax
        action_scores = tem_results[:, 0]
        # start_scores = tem_results[:, 1]
        # end_scores = tem_results[:, 2]
        length = len(tem_results)
        tgap = 1. / length
        max_action = max(action_scores)

        new_props = []
        labels = []
        for alpha in alpha_list:
            label = (action_scores > alpha * max_action).astype(int)
            labels.append(label)
        for label in labels:
            diff = np.empty(length + 1)
            diff[1:-1] = label[1:].astype(int) - label[:-1].astype(int)
            diff[0] = float(label[0])
            diff[-1] = 0 - float(label[-1])  # 每个位置与前一个位置的差值
            cs = np.cumsum(1 - label)  # cs[i]表示第i个位置以及之前有多少0(即actionness低于阈值)
            offset = np.arange(0, length, 1)

            up = np.nonzero(diff == 1)[0]  # segment开始位置
            down = np.nonzero(diff == -1)[0]  # segment结束位置

            assert len(up) == len(down), '{} != {}'.format(len(up), len(down))
            for i, t in enumerate(
                    beta_list):  # t为给定的阈值, t * offset=每个位置允许的0的个数
                signal = cs - t * offset
                for x in range(len(up)):
                    s = signal[up[x]]  # 在每个segment开始位置的信号
                    for y in range(x + 1, len(up)):  # x segment后面的segment
                        if y < len(down) and signal[up[
                                y]] > s:  # y segment 开始位置的信号大于x segment开始位置的信号
                            new_props.append(
                                (up[x] * tgap, down[y - 1] * tgap,
                                 np.mean(action_scores[up[x]:down[y - 1]])))
                            break
                    else:  # x 是最后一个segment 或 y 是最后一个segment
                        new_props.append(
                            (up[x] * tgap, down[-1] * tgap,
                             np.mean(action_scores[up[x]:down[-1]])))

                for x in range(len(down) - 1, -1, -1):  # 反过来做一遍
                    # s = signal[down[x]-1] if down[x] < length else signal[-1] - t  # noqa
                    s = signal[down[x] - 1]
                    for y in range(x - 1, -1, -1):
                        if y >= 0 and signal[down[y] - 1] < s:
                            new_props.append(
                                (up[y + 1] * tgap, down[x] * tgap,
                                 np.mean(action_scores[up[y + 1]:down[x]])))
                            break
                    else:
                        new_props.append((up[0] * tgap, down[x] * tgap,
                                          np.mean(action_scores[0:down[x]])))
        new_props = np.stack(new_props)
        # tmin, tmax, action_score
        new_props = new_props[new_props[:, 2].argsort()[::-1]]
        video_info = video_infos[video_index]
        corrected_second = float(video_info['duration_second'])

        gt_tmins = []
        gt_tmaxs = []
        for annotations in video_info['annotations']:
            gt_tmins.append(annotations['segment'][0] / corrected_second)
            gt_tmaxs.append(annotations['segment'][1] / corrected_second)

        new_iou_list = []
        new_ioa_list = []
        for new_prop in new_props:
            new_iou = max(
                temporal_iou(new_prop[0], new_prop[1], gt_tmins, gt_tmaxs))
            new_ioa = max(
                temporal_iop(new_prop[0], new_prop[1], gt_tmins, gt_tmaxs))
            new_iou_list.append(new_iou)
            new_ioa_list.append(new_ioa)

        new_iou_list = np.array(new_iou_list).reshape(-1, 1)
        new_ioa_list = np.array(new_ioa_list).reshape(-1, 1)
        new_props = np.concatenate((new_props, new_iou_list), axis=1)
        new_props = np.concatenate((new_props, new_ioa_list), axis=1)
        proposal_dict[video_name] = new_props
        # tmin, tmax, action_score, iou, iop
        if result_dict is not None:
            result_dict[video_name] = new_props
    return proposal_dict


def generate_tag_feature(video_list,
                         video_infos,
                         tem_results_dir,
                         pgm_proposals_dir,
                         top_k=1000,
                         bsp_boundary_ratio=0.2,
                         num_sample_start=8,
                         num_sample_end=8,
                         num_sample_action=16,
                         num_sample_interp=3,
                         tem_results_ext='.csv',
                         pgm_proposal_ext='.csv',
                         result_dict=None):
    """Generate Boundary-Sensitive Proposal Feature with given proposals.

    Args:
        video_list (list[int]): List of video indexs to generate bsp_feature.
        video_infos (list[dict]): List of video_info dict that contains
            'video_name'.
        tem_results_dir (str): Directory to load temporal evaluation
            results.
        pgm_proposals_dir (str): Directory to load proposals.
        top_k (int): Number of proposals to be considered. Default: 1000
        bsp_boundary_ratio (float): Ratio for proposal boundary
            (start/end). Default: 0.2.
        num_sample_start (int): Num of samples for actionness in
            start region. Default: 8.
        num_sample_end (int): Num of samples for actionness in end region.
            Default: 8.
        num_sample_action (int): Num of samples for actionness in center
            region. Default: 16.
        num_sample_interp (int): Num of samples for interpolation for
            each sample point. Default: 3.
        tem_results_ext (str): File extension for temporal evaluation
            model output. Default: '.csv'.
        pgm_proposal_ext (str): File extension for proposals. Default: '.csv'.
        result_dict (dict | None): The dict to save the results. Default: None.

    Returns:
        bsp_feature_dict (dict): A dict contains video_name as keys and
            bsp_feature as value. If result_dict is not None, save the
            results to it.
    """
    if tem_results_ext != '.csv' or pgm_proposal_ext != '.csv':
        raise NotImplementedError('Only support csv format now.')

    bsp_feature_dict = {}
    for video_index in video_list:
        video_name = video_infos[video_index]['video_name']

        # Load temporal evaluation results
        tem_path = osp.join(tem_results_dir, video_name + tem_results_ext)
        tem_results = np.loadtxt(
            tem_path, dtype=np.float32, delimiter=',', skiprows=1)
        # action, start, end, tmin, tmax
        score_action = tem_results[:, 0]
        seg_tmins = tem_results[:, 3]
        seg_tmaxs = tem_results[:, 4]
        video_scale = len(tem_results)
        video_gap = seg_tmaxs[0] - seg_tmins[0]
        video_extend = int(video_scale / 4 + 10)

        # Load proposals results
        proposal_path = osp.join(pgm_proposals_dir,
                                 video_name + pgm_proposal_ext)
        pgm_proposals = np.loadtxt(
            proposal_path, dtype=np.float32, delimiter=',', skiprows=1)
        pgm_proposals = pgm_proposals[:top_k]

        # Generate temporal sample points
        boundary_zeros = np.zeros([video_extend])
        score_action = np.concatenate(
            (boundary_zeros, score_action, boundary_zeros))
        begin_tp = []
        middle_tp = []
        end_tp = []
        for i in range(video_extend):
            begin_tp.append(-video_gap / 2 -
                            (video_extend - 1 - i) * video_gap)
            end_tp.append(video_gap / 2 + seg_tmaxs[-1] + i * video_gap)
        for i in range(video_scale):
            middle_tp.append(video_gap / 2 + i * video_gap)
        t_points = begin_tp + middle_tp + end_tp

        bsp_feature = []
        for pgm_proposal in pgm_proposals:
            # tmin, tmax, action_score, iou, iop
            tmin = pgm_proposal[0]
            tmax = pgm_proposal[1]

            tlen = tmax - tmin
            # Temporal range for start
            tmin_0 = tmin - tlen * bsp_boundary_ratio
            tmin_1 = tmin + tlen * bsp_boundary_ratio
            # Temporal range for end
            tmax_0 = tmax - tlen * bsp_boundary_ratio
            tmax_1 = tmax + tlen * bsp_boundary_ratio

            # Generate features at start boundary
            tlen_start = (tmin_1 - tmin_0) / (num_sample_start - 1)
            tlen_start_sample = tlen_start / num_sample_interp
            t_new = [
                tmin_0 - tlen_start / 2 + tlen_start_sample * i
                for i in range(num_sample_start * num_sample_interp + 1)
            ]
            y_new_start_action = np.interp(t_new, t_points, score_action)
            y_new_start = [
                np.mean(y_new_start_action[i * num_sample_interp:(i + 1) *
                                           num_sample_interp + 1])
                for i in range(num_sample_start)
            ]
            # Generate features at end boundary
            tlen_end = (tmax_1 - tmax_0) / (num_sample_end - 1)
            tlen_end_sample = tlen_end / num_sample_interp
            t_new = [
                tmax_0 - tlen_end / 2 + tlen_end_sample * i
                for i in range(num_sample_end * num_sample_interp + 1)
            ]
            y_new_end_action = np.interp(t_new, t_points, score_action)
            y_new_end = [
                np.mean(y_new_end_action[i * num_sample_interp:(i + 1) *
                                         num_sample_interp + 1])
                for i in range(num_sample_end)
            ]
            # Generate features for action
            tlen_action = (tmax - tmin) / (num_sample_action - 1)
            tlen_action_sample = tlen_action / num_sample_interp
            t_new = [
                tmin - tlen_action / 2 + tlen_action_sample * i
                for i in range(num_sample_action * num_sample_interp + 1)
            ]
            y_new_action = np.interp(t_new, t_points, score_action)
            y_new_action = [
                np.mean(y_new_action[i * num_sample_interp:(i + 1) *
                                     num_sample_interp + 1])
                for i in range(num_sample_action)
            ]
            feature = np.concatenate([y_new_action, y_new_start, y_new_end])
            bsp_feature.append(feature)
        bsp_feature = np.array(bsp_feature)
        bsp_feature_dict[video_name] = bsp_feature
        if result_dict is not None:
            result_dict[video_name] = bsp_feature
    return bsp_feature_dict


def tag_soft_nms(proposals, alpha, low_threshold, high_threshold, top_k,
                 score_idx):
    """Soft NMS for tag temporal proposals.

    Args:
        proposals (np.ndarray): Proposals generated by network.
        alpha (float): Alpha value of Gaussian decaying function.
        low_threshold (float): Low threshold for soft nms.
        high_threshold (float): High threshold for soft nms.
        top_k (int): Top k values to be considered.

    Returns:
        np.ndarray: The updated proposals.
    """
    # tmin, tmax, action_score, match_iou, match_iop
    proposals = proposals[proposals[:, score_idx].argsort()[::-1]]
    tscores = copy.copy(list(proposals[:, score_idx]))
    tstart = list(proposals[:, 0])
    tend = list(proposals[:, 1])
    tscore = list(proposals[:, 2])
    tiou = list(proposals[:, 3])
    tiop = list(proposals[:, 4])
    rscores = []
    rstart = []
    rend = []
    rscore = []
    riou = []
    riop = []

    while len(tscores) > 0 and len(rscores) <= top_k:
        max_index = np.argmax(tscores)
        max_width = tend[max_index] - tstart[max_index]
        iou_list = temporal_iou(tstart[max_index], tend[max_index],
                                np.array(tstart), np.array(tend))
        iou_exp_list = np.exp(-np.square(iou_list) / alpha)

        for idx, _ in enumerate(tscores):
            if idx != max_index:
                current_iou = iou_list[idx]
                if current_iou > low_threshold + (high_threshold -
                                                  low_threshold) * max_width:
                    tscores[idx] = tscores[idx] * iou_exp_list[idx]

        rscores.append(tscores[max_index])
        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        riou.append(tiou[max_index])
        riop.append(tiop[max_index])
        tscores.pop(max_index)
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)
        tiou.pop(max_index)
        tiop.pop(max_index)

    rstart = np.array(rstart).reshape(-1, 1)
    rend = np.array(rend).reshape(-1, 1)
    rscore = np.array(rscore).reshape(-1, 1)
    riou = np.array(riou).reshape(-1, 1)
    riop = np.array(riop).reshape(-1, 1)
    new_proposals = np.concatenate((rstart, rend, rscore, riou, riop), axis=1)
    return new_proposals


def tag_post_processing(result, video_info, score_idx, soft_nms_alpha,
                        soft_nms_low_threshold, soft_nms_high_threshold,
                        post_process_top_k):
    """Post process for tag temporal proposals generation.

    Args:
        result (np.ndarray): Proposals generated by network.
        video_info (dict): Meta data of video. Required keys are
            'duration_frame', 'duration_second'.
        soft_nms_alpha (float): Alpha value of Gaussian decaying function.
        soft_nms_low_threshold (float): Low threshold for soft nms.
        soft_nms_high_threshold (float): High threshold for soft nms.
        post_process_top_k (int): Top k values to be considered.
        feature_extraction_interval (int): Interval used in feature extraction.

    Returns:
        list[dict]: The updated proposals, e.g.
            [{'score': 0.9, 'segment': [0, 1]},
             {'score': 0.8, 'segment': [0, 2]},
            ...].
    """
    if len(result) > 1:
        result = tag_soft_nms(result, soft_nms_alpha, soft_nms_low_threshold,
                              soft_nms_high_threshold, post_process_top_k,
                              score_idx)
    result = result[result[:, score_idx].argsort()[::-1]][:post_process_top_k]
    # start, end, score, iou, iop
    video_duration = float(video_info['duration_second'])
    proposal_list = []

    for j in range(len(result)):
        proposal = dict()
        proposal['score'] = float(result[j, score_idx])
        proposal['segment'] = [
            max(0, result[j, 0]) * video_duration,
            min(1, result[j, 1]) * video_duration
        ]
        proposal_list.append(proposal)
    return proposal_list, result


def multithread_dump_results(video_infos, pgm_proposals_dir,
                             tag_pgm_result_dir, result_dict, kwargs):
    prog_bar = mmcv.ProgressBar(len(video_infos))
    prog_bar.start()
    for vinfo in video_infos:
        video_name = vinfo['video_name']
        file_name = osp.join(pgm_proposals_dir, video_name + '.csv')
        proposal = np.loadtxt(
            file_name, dtype=np.float32, delimiter=',', skiprows=1)
        proposal_list, post_proposal = tag_post_processing(
            proposal, vinfo, score_idx=2, **kwargs)
        tag_pgm_file = osp.join(tag_pgm_result_dir, video_name + '.csv')
        header = 'tmin,tmax,action_score,match_iou,match_ioa'
        np.savetxt(
            tag_pgm_file,
            post_proposal,
            header=header,
            delimiter=',',
            comments='')
        result_dict[video_name] = proposal_list
        prog_bar.update()


def dump_results(pgm_proposals_dir, tag_pgm_result_dir, ann_file, out,
                 **kwargs):
    os.makedirs(tag_pgm_result_dir, exist_ok=True)
    video_infos = load_video_infos(ann_file)
    thread_num = kwargs.pop('threads', 1)
    videos_per_thread = (len(video_infos) + thread_num - 1) // thread_num
    jobs = []
    result_dict = Manager().dict()
    for i in range(thread_num):
        proc = Process(
            target=multithread_dump_results,
            args=(video_infos[i * videos_per_thread:(i + 1) *
                              videos_per_thread], pgm_proposals_dir,
                  tag_pgm_result_dir, result_dict, kwargs))
        proc.start()
        jobs.append(proc)
    for job in jobs:
        job.join()
    mmcv.dump(result_dict.copy(), out)


def multithread_dump_highest_iou_results(video_infos, pgm_proposals_dir,
                                         tag_pgm_result_dir, result_dict,
                                         kwargs):
    # prog_bar = mmcv.ProgressBar(len(video_infos))
    # prog_bar.start()
    # for vinfo in video_infos:
    #     video_name = vinfo['video_name']
    #     video_duration = vinfo['duration_second']
    #     file_name = osp.join(pgm_proposals_dir, video_name + '.csv')
    #     proposal = np.loadtxt(
    #         file_name, dtype=np.float32, delimiter=',', skiprows=1)
    #     # tmin, tmax, score, iou, iop
    #     post_proposal = proposal[proposal[:, 3].argsort()
    #                              [::-1]][:kwargs['post_process_top_k']]
    #     tag_pgm_file = osp.join(tag_pgm_result_dir, video_name + '.csv')
    #     header = 'tmin,tmax,action_score,match_iou,match_ioa'
    #     np.savetxt(
    #         tag_pgm_file,
    #         post_proposal,
    #         header=header,
    #         delimiter=',',
    #         comments='')
    #     proposal_list = []
    #     for result in post_proposal:
    #         proposal = dict()
    #         proposal['score'] = float(result[3])
    #         proposal['segment'] = [
    #             max(0, result[0]) * video_duration,
    #             min(1, result[1]) * video_duration
    #         ]
    #         proposal_list.append(proposal)
    #     result_dict[video_name] = proposal_list
    #     prog_bar.update()
    prog_bar = mmcv.ProgressBar(len(video_infos))
    prog_bar.start()
    for vinfo in video_infos:
        video_name = vinfo['video_name']
        file_name = osp.join(pgm_proposals_dir, video_name + '.csv')
        proposal = np.loadtxt(
            file_name, dtype=np.float32, delimiter=',', skiprows=1)
        proposal_list, post_proposal = tag_post_processing(
            proposal, vinfo, score_idx=3, **kwargs)
        tag_pgm_file = osp.join(tag_pgm_result_dir, video_name + '.csv')
        header = 'tmin,tmax,action_score,match_iou,match_ioa'
        np.savetxt(
            tag_pgm_file,
            post_proposal,
            header=header,
            delimiter=',',
            comments='')
        result_dict[video_name] = proposal_list
        prog_bar.update()


def dump_highest_iou_results(pgm_proposals_dir, tag_pgm_result_dir, ann_file,
                             out, **kwargs):
    os.makedirs(tag_pgm_result_dir, exist_ok=True)
    video_infos = load_video_infos(ann_file)
    thread_num = kwargs.pop('threads', 1)
    videos_per_thread = (len(video_infos) + thread_num - 1) // thread_num
    jobs = []
    result_dict = Manager().dict()
    for i in range(thread_num):
        proc = Process(
            target=multithread_dump_highest_iou_results,
            args=(video_infos[i * videos_per_thread:(i + 1) *
                              videos_per_thread], pgm_proposals_dir,
                  tag_pgm_result_dir, result_dict, kwargs))
        proc.start()
        jobs.append(proc)
    for job in jobs:
        job.join()
    mmcv.dump(result_dict.copy(), out)
