import argparse

import mmcv

from mmaction.localization import dump_results


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
    cfg = mmcv.Config.fromfile(args.config)
    pgm_proposals_dir = cfg.pgm_proposals_dir
    tag_pgm_result_dir = cfg.tag_pgm_result_dir
    out = cfg.output_config.out
    mode = args.mode
    if mode == 'train':
        dump_results(pgm_proposals_dir, tag_pgm_result_dir, cfg.ann_file_train,
                     out, **cfg.tag_results_config)
    elif mode == 'test':
        dump_results(pgm_proposals_dir, tag_pgm_result_dir, cfg.ann_file_val,
                     out, **cfg.tag_results_config)
    print('Finish generate post process proposals.')


if __name__ == '__main__':
    main()
