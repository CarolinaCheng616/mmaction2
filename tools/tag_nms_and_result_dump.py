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
    args = parser.parse_args()
    return args


def main():
    """generate tag proposal results."""
    print('Begin generate post process proposals.')
    args = parse_args()
    mode = args.mode
    cfg = mmcv.Config.fromfile(args.config)
    pgm_proposals_dir = cfg.pgm_proposals_dir
    iou_nms = cfg.iou_nms
    if iou_nms:
        tag_pgm_result_dir = cfg.tag_iou_nms_config.pop('tag_pgm_result_dir')
        out = cfg.tag_iou_nms_config.pop('output_config')['out']
        kwargs = cfg.tag_iou_nms_config
    else:
        tag_pgm_result_dir = cfg.tag_score_nms_config.pop('tag_pgm_result_dir')
        out = cfg.tag_score_nms_config.pop('output_config')['out']
        kwargs = cfg.tag_score_nms_config
    ann_file = cfg.ann_file_train if mode == 'train' else cfg.ann_file_val
    nms_and_dump_results(
        pgm_proposals_dir,
        tag_pgm_result_dir,
        ann_file,
        out,
        iou_nms=iou_nms,
        **kwargs)
    print('Finish generate post process proposals.')


if __name__ == '__main__':
    main()
