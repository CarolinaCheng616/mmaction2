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


if __name__ == '__main__':
    foo_save_module()
