import copy
import os
import os.path as osp
import warnings

import mmcv
import numpy as np

from ..core import average_recall_at_avg_proposals
from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module()
class TruNetDataset(BaseDataset):
    """TruNet dataset for temporal proposal generation.

    The dataset loads raw features and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a json file with multiple objects, and each object has a
    key of the name of a video, and value of total duration of the video,
    annotations of a video. Example of a annotation file:
    .. code-block:: JSON

    {
        "219131500": {
            "duration_second": 5938,
            "annotations": [
                {
                    "segment": [
                        139.734487,
                        254.652057
                    ]
                },
                {
                    "segment": [
                        263.529267,
                        356.486561
                    ]
                }
            ]
        }
    }

    note: fps is always 1


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str | None): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
    """

    def __init__(self, ann_file, pipeline, data_prefix=None, test_mode=False):
        super().__init__(ann_file, pipeline, data_prefix, test_mode)

    def load_annotations(self):
        """Load the annotation according to ann_file into video_infos."""
        video_infos = []
        anno_database = mmcv.load(self.ann_file)
        for video_name in anno_database:
            video_info = anno_database[video_name]
            video_info['video_name'] = video_name
            video_infos.append(video_info)
        return video_infos

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['data_prefix'] = self.data_prefix
        return self.pipeline(results)

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['data_prefix'] = self.data_prefix
        return self.pipeline(results)

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.video_infos)

    def _import_ground_truth(self):
        """Read ground truth data from video_infos."""
        ground_truth = {}
        for video_info in self.video_infos:
            video_id = video_info['video_name']
            this_video_ground_truths = []
            for ann in video_info['annotations']:
                t_start, t_end = ann['segment']
                this_video_ground_truths.append([t_start, t_end])
            ground_truth[video_id] = np.array(this_video_ground_truths)
        return ground_truth

    @staticmethod
    def proposals2json(results, show_progress=False):
        """Convert all proposals to a final dict(json) format.

        Args:
            results (list[dict]): All proposals.
            show_progress (bool): Whether to show the progress bar.
                Defaults: False.

        Returns:
            dict: The final result dict. E.g.

            .. code-block:: Python

                dict(video-1=[dict(segment=[1.1,2.0]. score=0.9),
                              dict(segment=[50.1, 129.3], score=0.6)])
        """
        result_dict = {}
        print('Convert proposals to json format')
        if show_progress:
            prog_bar = mmcv.ProgressBar(len(results))
        for result in results:
            video_name = result['video_name']
            result_dict[video_name] = result['proposal_list']
            if show_progress:
                prog_bar.update()
        return result_dict

    @staticmethod
    def _import_proposals(results):
        """Read predictions from results."""
        proposals = {}
        num_proposals = 0
        for result in results:
            video_id = result['video_name']
            this_video_proposals = []
            for proposal in result['proposal_list']:
                t_start, t_end = proposal['segment']
                score = proposal['score']
                this_video_proposals.append([t_start, t_end, score])
                num_proposals += 1
            proposals[video_id] = np.array(this_video_proposals)
        return proposals, num_proposals

    def dump_results(self, results, out, output_format, version='VERSION 1.3'):
        """Dump data to json/csv files."""
        if output_format == 'json':
            result_dict = self.proposals2json(results)
            mmcv.dump(result_dict, out)
        elif output_format == 'csv':
            # TODO: add csv handler to mmcv and use mmcv.dump
            os.makedirs(out, exist_ok=True)
            header = 'action,start,end,tmin,tmax'
            for result in results:
                video_name, outputs = result
                output_path = osp.join(out, video_name + '.csv')
                np.savetxt(
                    output_path,
                    outputs,
                    header=header,
                    delimiter=',',
                    comments='')
        else:
            raise ValueError(
                f'The output format {output_format} is not supported.')

    def wrapper_compute_average_precision(self, results, tiou_thresholds):
        """Computes average precision for each class."""
        # load ground truth
        ground_truth = []
        for video_info in self.video_infos():
            video_id = video_info['video_name']
            for anno in video_info['annotations']:
                ground_truth_item = dict()
                ground_truth_item['video-id'] = video_id
                ground_truth_item['t-start'] = float(anno['segment'][0])
                ground_truth_item['t-end'] = float(anno['segment'][1])
                ground_truth_item['label'] = 0
                ground_truth.append(ground_truth_item)

        # load proposals
        prediction = []
        for result in results:
            video_id = result['video_name']
            for proposal in result['proposal_list']:
                prediction_item = dict()
                prediction_item['video-id'] = video_id
                prediction_item['label'] = 0
                prediction_item['t-start'] = float(proposal['segment'][0])
                prediction_item['t-end'] = float(proposal['segment'][1])
                prediction_item['score'] = proposal['score']
                prediction.append(prediction_item)

        activity_index = {0: 0}
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))
        # Adaptation to query faster
        ground_truth_by_label = []
        prediction_by_label = []
        for i in range(len(activity_index)):
            ground_truth_by_label.append([])
            prediction_by_label.append([])
        for gt in ground_truth:
            ground_truth_by_label[gt['label']].append(gt)
        for pred in prediction:
            prediction_by_label[pred['label']].append(pred)

        for i in range(len(activity_index)):
            ap_result = compute_average_precision_detection(
                ground_truth_by_label[i],
                prediction_by_label[i],
                tiou_thresholds=tiou_thresholds)
            ap[:, i] = ap_result

        return ap

    def evaluate(
            self,
            results,
            metrics='AR@AN',
            metric_options={
                'AR@AN':
                dict(
                    max_avg_proposals=100,
                    temporal_iou_thresholds=np.linspace(0.5, 0.95, 10)),
                'mAP':
                dict(
                    tiou_thresholds=np.linspace(0.5, 0.95, 10)
                )
            },
            logger=None,
            **deprecated_kwargs):
        """Evaluation in feature dataset.

        Args:
            results (list[dict]): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'AR@AN'.
            metric_options (dict): Dict for metric options. Options are
                ``max_avg_proposals``, ``temporal_iou_thresholds`` for
                ``AR@AN``.
                default: ``{'AR@AN': dict(max_avg_proposals=100,
                temporal_iou_thresholds=np.linspace(0.5, 0.95, 10))}``.
            logger (logging.Logger | None): Training logger. Defaults: None.
            deprecated_kwargs (dict): Used for containing deprecated arguments.
                See 'https://github.com/open-mmlab/mmaction2/pull/286'.

        Returns:
            dict: Evaluation results for evaluation metrics.
        """
        # Protect ``metric_options`` since it uses mutable value as default
        metric_options = copy.deepcopy(metric_options)

        if deprecated_kwargs != {}:
            warnings.warn(
                'Option arguments for metrics has been changed to '
                "`metric_options`, See 'https://github.com/open-mmlab/mmaction2/pull/286' "  # noqa: E501
                'for more details')
            metric_options['AR@AN'] = dict(metric_options['AR@AN'],
                                           **deprecated_kwargs)

        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = ['AR@AN', 'mAP']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        eval_results = {}
        ground_truth = self._import_ground_truth()
        proposal, num_proposals = self._import_proposals(results)

        for metric in metrics:
            if metric == 'AR@AN':
                temporal_iou_thresholds = metric_options.setdefault(
                    'AR@AN', {}).setdefault('temporal_iou_thresholds',
                                            np.linspace(0.5, 0.95, 10))
                max_avg_proposals = metric_options.setdefault(
                    'AR@AN', {}).setdefault('max_avg_proposals', 100)
                if isinstance(temporal_iou_thresholds, list):
                    temporal_iou_thresholds = np.array(temporal_iou_thresholds)

                recall, _, _, auc = (
                    average_recall_at_avg_proposals(
                        ground_truth,
                        proposal,
                        num_proposals,
                        max_avg_proposals=max_avg_proposals,
                        temporal_iou_thresholds=temporal_iou_thresholds))
                eval_results['auc'] = auc
                eval_results['AR@1'] = np.mean(recall[:, 0])
                eval_results['AR@5'] = np.mean(recall[:, 4])
                eval_results['AR@10'] = np.mean(recall[:, 9])
                eval_results['AR@100'] = np.mean(recall[:, 99])

            if metric == 'mAP':
                tiou_thresholds = metric_options.setdefault(
                    'mAP', {}).setdefault('tiou_thresholds',
                                          np.linspace(0.5, 0.95, 10))
                ap = self.wrapper_compute_average_precision(results, tiou_thresholds)
                mAP = ap.mean(axis=1)
                average_mAP = mAP.mean()
                print(f'mAP: {mAP}')
                print(f'average_mAP: {average_mAP}')

        return eval_results


def compute_average_precision_detection(ground_truth,
                                        prediction,
                                        # proposal_num=100,
                                        tiou_thresholds=np.linspace(
                                            0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as true
    positive. This code is greatly inspired by Pascal VOC devkit.

    Args:
        ground_truth (list[dict]): List containing the ground truth instances
            (dictionaries). Required keys are 'video-id', 't-start' and
            't-end'.
        prediction (list[dict]): List containing the prediction instances
            (dictionaries). Required keys are: 'video-id', 't-start', 't-end'
            and 'score'.
        proposal_num (int): Number of average number to be reported in AR@AN.
        tiou_thresholds (np.ndarray): A 1darray indicates the temporal
            intersection over union threshold, which is optional.
            Default: ``np.linspace(0.5, 0.95, 10)``.

    Returns:
        Float: ap, Average precision score.
    """
    num_thresholds = len(tiou_thresholds)
    num_gts = len(ground_truth)
    num_preds = len(prediction)
    ap = np.zeros(num_thresholds)
    if len(prediction) == 0:
        return ap

    num_positive = float(num_gts)
    lock_gt = np.ones((num_thresholds, num_gts)) * -1
    # Sort predictions by decreasing score order.
    prediction.sort(key=lambda x: -x['score'])
    # Initialize true positive and false positive vectors.
    tp = np.zeros((num_thresholds, num_preds))
    fp = np.zeros((num_thresholds, num_preds))

    # Adaptation to query faster
    ground_truth_by_videoid = {}
    for i, item in enumerate(ground_truth):
        item['index'] = i
        ground_truth_by_videoid.setdefault(item['video-id'], []).append(item)

    # Assigning true positive to truly grount truth instances.
    for idx, pred in enumerate(prediction):
        if pred['video-id'] in ground_truth_by_videoid:
            gts = ground_truth_by_videoid[pred['video-id']]
        else:
            fp[:, idx] = 1
            continue

        tiou_arr = pairwise_temporal_iou(
            np.array([pred['t-start'], pred['t-end']]),
            np.array([np.array([gt['t-start'], gt['t-end']]) for gt in gts]))
        tiou_arr = tiou_arr.reshape(-1)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for t_idx, tiou_threshold in enumerate(tiou_thresholds):
            for j_idx in tiou_sorted_idx:
                if tiou_arr[j_idx] < tiou_threshold:
                    fp[t_idx, idx] = 1
                    break
                if lock_gt[t_idx, gts[j_idx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[t_idx, idx] = 1
                lock_gt[t_idx, gts[j_idx]['index']] = idx
                break

            if fp[t_idx, idx] == 0 and tp[t_idx, idx] == 0:
                fp[t_idx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    recall_cumsum = tp_cumsum / num_positive

    # print(f'AR@AN={proposal_num}: {np.mean(recall_cumsum, axis=0)[-1]}')

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for t_idx in range(len(tiou_thresholds)):
        ap[t_idx] = interpolated_precision_recall(precision_cumsum[t_idx, :],
                                                  recall_cumsum[t_idx, :])

    return ap


def pairwise_temporal_iou(candidate_segments,
                          target_segments,
                          calculate_overlap_self=False):
    """Compute intersection over union between segments.

    Args:
        candidate_segments (np.ndarray): 1-dim/2-dim array in format
            ``[init, end]/[m x 2:=[init, end]]``.
        target_segments (np.ndarray): 2-dim array in format
            ``[n x 2:=[init, end]]``.
        calculate_overlap_self (bool): Whether to calculate overlap_self
            (union / candidate_length) or not. Default: False.

    Returns:
        t_iou (np.ndarray): 1-dim array [n] /
            2-dim array [n x m] with IoU ratio.
        t_overlap_self (np.ndarray, optional): 1-dim array [n] /
            2-dim array [n x m] with overlap_self, returns when
            calculate_overlap_self is True.
    """
    candidate_segments_ndim = candidate_segments.ndim
    if target_segments.ndim != 2 or candidate_segments_ndim not in [1, 2]:
        raise ValueError('Dimension of arguments is incorrect')

    if candidate_segments_ndim == 1:
        candidate_segments = candidate_segments[np.newaxis, :]

    n, m = target_segments.shape[0], candidate_segments.shape[0]
    t_iou = np.empty((n, m), dtype=np.float32)
    if calculate_overlap_self:
        t_overlap_self = np.empty((n, m), dtype=np.float32)

    for i in range(m):
        candidate_segment = candidate_segments[i, :]
        tt1 = np.maximum(candidate_segment[0], target_segments[:, 0])
        tt2 = np.minimum(candidate_segment[1], target_segments[:, 1])
        # Intersection including Non-negative overlap score.
        segments_intersection = (tt2 - tt1).clip(0)
        # Segment union.
        segments_union = ((target_segments[:, 1] - target_segments[:, 0]) +
                          (candidate_segment[1] - candidate_segment[0]) -
                          segments_intersection)
        # Compute overlap as the ratio of the intersection
        # over union of two segments.
        t_iou[:, i] = (segments_intersection.astype(float) / segments_union)
        if calculate_overlap_self:
            candidate_length = candidate_segment[1] - candidate_segment[0]
            t_overlap_self[:, i] = (
                segments_intersection.astype(float) / candidate_length)

    if candidate_segments_ndim == 1:
        t_iou = np.squeeze(t_iou, axis=1)
    if calculate_overlap_self:
        if candidate_segments_ndim == 1:
            t_overlap_self = np.squeeze(t_overlap_self, axis=1)
        return t_iou, t_overlap_self

    return t_iou


def interpolated_precision_recall(precision, recall):
    """Interpolated AP - VOCdevkit from VOC 2011.

    Args:
        precision (np.ndarray): The precision of different thresholds.
        recall (np.ndarray): The recall of different thresholds.

    Returns：
        float: Average precision score.
    """
    mprecision = np.hstack([[0], precision, [0]])
    mrecall = np.hstack([[0], recall, [1]])
    for i in range(len(mprecision) - 1)[::-1]:
        mprecision[i] = max(mprecision[i], mprecision[i + 1])
    idx = np.where(mrecall[1::] != mrecall[0:-1])[0] + 1
    ap = np.sum((mrecall[idx] - mrecall[idx - 1]) * mprecision[idx])
    return ap
