# from mmcv.runner import _load_checkpoint
import torch
import sys


if __name__ == '__main__':
    pretrained = sys.argv[1]
    weights = torch.load(pretrained, map_location='cpu')
    import pdb
    pdb.set_trace()
