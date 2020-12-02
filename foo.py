from mmcv.runner import _load_checkpoint


if __name__ == '__main__':
    pretrained = 'torchvision://resnet152'
    state_dict_r2d = _load_checkpoint(pretrained)
