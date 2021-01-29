import argparse

import mmcv

from mmaction.localization import nms_and_dump_results


def parse_args():
    parser = argparse.ArgumentParser(
        description='generate tag proposal results.')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--mode',
        choices=['train', 'test'],
        default='test',
        help='train or test')
    parser.add_argument(
        '--iou_nms',
        action="store_true",
        type=bool,
        help='True or False')
    args = parser.parse_args()
    return args


def main():
    """generate tag proposal results."""
    print('Begin generate post process proposals.')
    args = parse_args()
    mode = args.mode
    cfg = mmcv.Config.fromfile(args.config)
    pgm_proposals_dir = cfg.pgm_proposals_dir
    tem_results_dir = cfg.tem_results_dir
    iou_nms = args.iou_nms
    if iou_nms:
        nms_proposals_dir = cfg.tag_iou_nms_config.pop('proposals_dir')
        nms_features_dir = cfg.tag_iou_nms_config.pop('features_dir')
        out = cfg.tag_iou_nms_config.pop('output_config')['out']
        proposal_kwargs = cfg.tag_iou_nms_config
    else:
        nms_proposals_dir = cfg.tag_score_nms_config.pop('proposals_dir')
        nms_features_dir = cfg.tag_score_nms_config.pop('features_dir')
        out = cfg.tag_score_nms_config.pop('output_config')['out']
        proposal_kwargs = cfg.tag_score_nms_config
    import pdb
    pdb.set_trace()
    feature_kwargs = cfg.feature_kwargs
    ann_file = cfg.ann_file_train if mode == 'train' else cfg.ann_file_val
    nms_and_dump_results(pgm_proposals_dir, tem_results_dir, nms_proposals_dir,
                         nms_features_dir, ann_file, out, iou_nms,
                         proposal_kwargs, feature_kwargs)
    print('Finish generate post process proposals.')


if __name__ == '__main__':
    main()
