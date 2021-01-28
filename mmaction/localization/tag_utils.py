import os.path as osp

import numpy as np

from .proposal_utils import temporal_iop, temporal_iou


def generate_tag_proposals(video_list,
                           video_infos,
                           tem_results_dir,
                           alpha_list=[0.01, 0.05, 0.1, .15, 0.25, .4, .5, .6, .7, .8, .9, .95, ],
                           beta_list=[0.05, .1, .2, .3, .4, .5, .6, 0.8, 1.0],
                           tem_results_ext='.csv',
                           result_dict=None):
    """Generate Candidate Proposals with given temporal evalutation results.
    Each proposal file will contain:
    'tmin,tmax,mean_action,tmin_score,tmax_score,score,match_iou,match_ioa'.

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
        length = len(tem_results)
        tgap = 1. / length
        action_scores = tem_results[:, 0]
        start_scores = tem_results[:, 1]
        end_scores = tem_results[:, 2]
        max_action = max(action_scores)

        new_props = []
        labels = []
        for alpha in alpha_list:
            label = np.nonzero(action_scores > alpha * max_action)[0]
            labels.append(label)
        for label in labels:
            diff = np.empty(length + 1)
            diff[1:-1] = label[1:].astype(int) - label[:-1].astype(int)
            diff[0] = float(label[0])
            diff[length] = 0 - float(label[-1])  # 每个位置与前一个位置的差值
            cs = np.cumsum(1 - label)  # cs[i]表示第i个位置以及之前有多少0(即actionness低于阈值)
            offset = np.arange(0, length, 1)

            up = np.nonzero(diff == 1)[0]  # segment开始位置
            down = np.nonzero(diff == -1)[0]  # segment结束位置

            assert len(up) == len(down), "{} != {}".format(len(up), len(down))
            for i, t in enumerate(beta_list):  # t为给定的阈值, t * offset=每个位置允许的0的个数
                signal = cs - t * offset
                for x in range(len(up)):
                    s = signal[up[x]]  # 在每个segment开始位置的信号
                    for y in range(x + 1, len(up)):  # x segment后面的segment
                        if y < len(down) and signal[up[y]] > s:  # y segment 开始位置的信号大于x segment开始位置的信号
                            new_props.append((up[x] * tgap, down[y - 1] * tgap,
                                              np.mean(action_scores[up[x]:down[y - 1]]),
                                              start_scores[up[x]], end_scores[down[y - 1] - 1]))
                            break
                    else:  # x 是最后一个segment 或 y 是最后一个segment
                        new_props.append((up[x] * tgap, down[-1] * tgap,
                                          np.mean(action_scores[up[x]:down[-1]]),
                                          start_scores[up[x]], end_scores[down[-1] - 1]))

                for x in range(len(down) - 1, -1, -1):  # 反过来做一遍
                    # s = signal[down[x]-1] if down[x] < length else signal[-1] - t
                    s = signal[down[x] - 1]
                    for y in range(x - 1, -1, -1):
                        if y >= 0 and signal[down[y]-1] < s:
                            new_props.append((up[y + 1] * tgap, down[x] * tgap,
                                              np.mean(action_scores[up[y + 1]:down[x]]),
                                              start_scores[up[y + 1]], end_scores[down[x] - 1]))
                            break
                    else:
                        new_props.append((up[0] * tgap, down[x] * tgap,
                                          np.mean(action_scores[0: down[x]]),
                                          start_scores[up[0]], end_scores[down[x] - 1]))
        new_props = np.stack(new_props)
        # tmin, tmax, action_score, start_score, end_score
        new_props = new_props[new_props[:, 2].argsort()[::-1]]
        score_list = (new_props[:, 3] * new_props[:, 4]).reshape(-1, 1)
        new_props = np.concatenate((new_props, score_list), axis=1)
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
        # tmin, tmax, action_score, start_score, end_score, score, iou, iop
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
