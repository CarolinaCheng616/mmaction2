model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNetP3D',
        depth=199,
        pretrained='torchvision://resnet152',
        pretrained2d=True),
    cls_head=dict(
        type='P3DHead', num_classes=101, in_channels=2048, dropout_ratio=0.9))
train_cfg = None
test_cfg = dict(average_clips='prob')
dataset_type = 'RawframeDataset'
dataset_root = 'data/ucf101/'
data_root = dataset_root + 'rawframes'
data_root_val = dataset_root + 'rawframes'
ann_file_train = dataset_root + 'ucf101_train_list.txt'
ann_file_val = dataset_root + 'ucf101_val_list.txt'
ann_file_test = ann_file_val
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=16, frame_interval=1, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(182, 242)),
    # dict(type='RandomResizedCrop'),
    dict(type='RandomCrop', size=160),
    # dict(type='Resize', scale=(160, 160), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=1,
        num_clips=20,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(182, 242)),
    dict(type='CenterCrop', crop_size=160),
    # dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=1,
        num_clips=20,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(182, 242)),
    dict(type='CenterCrop', crop_size=160),
    # dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
optimizer = dict(type='SGD', lr=0.001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='step', step=[29, 58], gamma=0.1)
total_epochs = 76
checkpoint_config = dict(interval=10)
workflow = [('train', 1)]
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=20, hooks=[
        dict(type='TextLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/p3d_r152_ucf101_rgb'
load_from = None
resume_from = None
find_unused_parameters = False
