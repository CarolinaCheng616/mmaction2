import copy
import sys
import torch


def foo_save_module():
    ckpt_path = sys.argv[1]
    original_ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    target_ckpt = copy.deepcopy(original_ckpt)

    for key in original_ckpt['state_dict']:
        new_key = key.replace('module.', '')
        if 'fc' in new_key:
            new_key = 'cls_head.' + new_key
        else:
            new_key = 'backbone.' + new_key
        target_ckpt['state_dict'].pop(key)
        target_ckpt['state_dict'][new_key] = original_ckpt['state_dict'][key]

    torch.save(target_ckpt, sys.argv[2])


def tar():
    import os
    file = 'data/kinetics400/val_list_1.txt'
    src = '/mnt/lustre/DATAshare2/kinetics_400_val_320_frames/'
    dst = 'data/kinetics400/val_1/'
    with open(file, 'r', encoding='utf-8') as f:
        for row in f:
            filename = row.strip().split()[0]
            os.system('ln -s ' + src + filename + ' ' + dst + filename)
    os.system('tar -czvf data/kinetics400/val.tar.gz data/kinetics400/val_1')


if __name__ == '__main__':
    tar()
