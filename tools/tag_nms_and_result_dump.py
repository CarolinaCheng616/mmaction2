import argparse
import os.path as osp

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
        '--nms', choices=['iou', 'score'], help='iou nms or score nms')
    parser.add_argument(
        '--proposal',
        choices=['no_offset', 'offset'],
        help='generate proposals with of without offset')
    args = parser.parse_args()
    if args.mode == 'test' and args.proposal == 'offset':
        raise ValueError('cannot generate test proposals with offset')
    if args.mode == 'test' and args.nms == 'iou':
        raise ValueError('cannot use iou when testing')
    return args


def main():
    """generate tag proposal results."""
    print('Begin generate post process proposals.')
    args = parse_args()
    mode = args.mode
    cfg = mmcv.Config.fromfile(args.config)
    pgm_proposals_dir = cfg.pgm_proposals_dir
    origin = cfg.origin
    nms = args.nms
    iou_nms = False
    header = 'tmin,tmax,action_score,match_iou,match_ioa'
    if args.proposal == 'offset' and cfg.offset is False or args.proposal == 'no_offset' and cfg.offset is True:
        raise ValueError('non matching config.offset and args proposal')
    if args.proposal == 'offset':
        header += ',tmin_offset,tmax_offset'
    if nms == 'iou':
        iou_nms = True
        nms_proposals_dir = cfg.tag_iou_nms_config.pop('proposals_dir')
        nms_features_dir = cfg.tag_iou_nms_config.pop('features_dir')
        out = cfg.tag_iou_nms_config.pop('output_config')['out']
        proposal_kwargs = cfg.tag_iou_nms_config
    elif nms == 'score':
        nms_proposals_dir = cfg.tag_score_nms_config.pop('proposals_dir')
        nms_features_dir = cfg.tag_score_nms_config.pop('features_dir')
        out = cfg.tag_score_nms_config.pop('output_config')['out']
        proposal_kwargs = cfg.tag_score_nms_config
    else:
        print('nms should be iou or score.')
        exit(0)
    dir_name, base_name = osp.dirname(out), osp.basename(out)
    base_name = f'{mode}_{base_name}'
    out = osp.join(dir_name, base_name)
    feature_kwargs = cfg.feature_kwargs
    ann_file = cfg.ann_file_train if mode == 'train' else cfg.ann_file_val
    features_dir = cfg.train_features_dir if mode == 'train' else cfg.test_features_dir
    nms_and_dump_results(pgm_proposals_dir, features_dir, nms_proposals_dir,
                         nms_features_dir, ann_file, out, iou_nms, proposal_kwargs,
                         feature_kwargs, header, origin)
    print('Finish generate post process proposals.')


if __name__ == '__main__':
    main()
