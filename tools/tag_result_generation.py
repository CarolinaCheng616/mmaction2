import argparse

import mmcv

from mmaction.localization import dump_highest_iou_results, dump_results


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
    iou_sort = cfg.highest_iou
    if iou_sort:
        tag_pgm_result_dir = cfg.tag_highest_iou_config.pop(
            'tag_pgm_result_dir')
        out = cfg.tag_highest_iou_config.pop('output_config')['out']
    else:
        tag_pgm_result_dir = cfg.tag_results_config.pop('tag_pgm_result_dir')
        out = cfg.tag_results_config.pop('output_config')['out']
    if mode == 'train':
        if not iou_sort:
            dump_results(pgm_proposals_dir, tag_pgm_result_dir,
                         cfg.ann_file_train, out, **cfg.tag_results_config)
        else:
            dump_highest_iou_results(pgm_proposals_dir, tag_pgm_result_dir,
                                     cfg.ann_file_train, out,
                                     **cfg.tag_highest_iou_config)
    elif mode == 'test':
        if not iou_sort:
            dump_results(pgm_proposals_dir, tag_pgm_result_dir,
                         cfg.ann_file_val, out, **cfg.tag_results_config)
        else:
            dump_highest_iou_results(pgm_proposals_dir, tag_pgm_result_dir,
                                     cfg.ann_file_val, out,
                                     **cfg.tag_highest_iou_config)
    print('Finish generate post process proposals.')


if __name__ == '__main__':
    main()
