dataset_type = 'SnippetSRDataset'
load_type = 'LoadSnippetRectifiedFeature'

if load_type == 'LoadSnippetRectifiedFeature':  # feature.shape: 4096, 3+temporal+3
    data_root = 'data/TruNet/sup_train_feature/'
    data_root_val = 'data/TruNet/sup_val_feature/'
elif load_type == 'LoadSnippetFeature':  # feature.shape: temporal, 4096
    data_root = 'data/TruNet/train_feature/'
    data_root_val = 'data/TruNet/val_feature/'

model = dict(
    type='SnippetTEM',
    tem_feat_dim=4096,
    tem_hidden_dim=512,
    tem_match_threshold=0.5,
    loss_weight=2)
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips='score')
# dataset settings
ann_file_train = 'data/TruNet/train_meta.json'
ann_file_val = 'data/TruNet/val_meta.json'
ann_file_test = 'data/TruNet/val_meta.json'
# ann_file_val = ann_file_train
# ann_file_test = ann_file_train

work_dir = 'work_dirs/tem_snippet_test_mc/'
tem_results_dir = f'{work_dir}/tem_results/'

test_pipeline = [
    dict(type=load_type,
         use_mc=True,
         array_length=0),
    dict(
        type='Collect',
        keys=['raw_feature'],
        meta_name='video_meta',
        meta_keys=['video_name', 'duration_second', 'snippet_length']),
    dict(type='ToTensor', keys=['raw_feature'])
]
train_pipeline = [
    dict(
        type=load_type,
        use_mc=True,
        array_length=0),
    dict(
        type='Collect',
        keys=['raw_feature', 'label_action', 'label_start', 'label_end'],
        meta_name='video_meta',
        meta_keys=['video_name', 'duration_second', 'snippet_length']),
    dict(
        type='ToTensor',
        keys=['raw_feature', 'label_action', 'label_start', 'label_end']),
    dict(
        type='ToDataContainer',
        fields=[
            dict(key='label_action', stack=False),
            dict(key='label_start', stack=False),
            dict(key='label_end', stack=False)
        ])
]
val_pipeline = [
    dict(type=load_type,
         use_mc=True,
         array_length=0),
    dict(
        type='Collect',
        keys=['raw_feature', 'label_action', 'label_start', 'label_end'],
        meta_name='video_meta',
        meta_keys=['video_name', 'duration_second', 'snippet_length']),
    dict(
        type='ToTensor',
        keys=['raw_feature', 'label_action', 'label_start', 'label_end']),
    dict(
        type='ToDataContainer',
        fields=[
            dict(key='label_action', stack=False),
            dict(key='label_start', stack=False),
            dict(key='label_end', stack=False)
        ])
]

data = dict(
    videos_per_gpu=4096,
    workers_per_gpu=0,
    train_dataloader=dict(drop_last=False, shuffle=False),
    val_dataloader=dict(
        videos_per_gpu=4096 * 8,
        workers_per_gpu=0,
        drop_last=False,
        shuffle=False),
    test_dataloader=dict(
        videos_per_gpu=4096 * 8,
        workers_per_gpu=0,
        drop_last=False,
        shuffle=False),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        data_prefix=data_root_val,
        test_mode=True),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        data_prefix=data_root_val,
        test_mode=True),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        data_prefix=data_root))

# reload
reload = True

# optimizer
optimizer = dict(
    type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)  # batch_size

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=6)

total_epochs = 20
checkpoint_config = dict(interval=1, filename_tmpl='tem_epoch_{}.pth')

log_config = dict(
    interval=2,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
output_config = dict(out=tem_results_dir, output_format='csv')
