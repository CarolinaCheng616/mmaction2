# model settings
# model = dict(
#     type='_TEM_',
#     temporal_dim=2000,
#     boundary_ratio=0.1,
#     tem_feat_dim=4096,
#     tem_hidden_dim=512,
#     tem_match_threshold=0.5)
model = dict(
    type='SnippetTEM',
    tem_feat_dim=4096,
    tem_hidden_dim=512,
    tem_match_threshold=0.5)
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips='score')
# dataset settings
dataset_type = 'SnippetDataset'
data_root = 'data/TruNet/tmp_train_feature/'
# data_root = 'data/TruNet/val_feature/'
# data_root_val = 'data/TruNet/val_feature/'
data_root_val = data_root
ann_file_train = 'data/TruNet/tmp_train_meta.json'
# ann_file_train = 'data/TruNet/val_meta.json'
# ann_file_val = 'data/TruNet/val_meta.json'
# ann_file_test = 'data/TruNet/val_meta.json'
ann_file_val = ann_file_train
ann_file_test = ann_file_train

work_dir = 'work_dirs/tem_snippet/'
tem_results_dir = f'{work_dir}/tem_results/'

test_pipeline = [
    dict(type='LoadSnippetLocalizationFeature'),
    dict(
        type='Collect',
        keys=['raw_feature'],
        meta_name='video_meta',
        meta_keys=['video_name', 'duration_second']),
    dict(type='ToTensor', keys=['raw_feature'])
]
train_pipeline = [
    dict(type='LoadSnippetLocalizationFeature'),
    # dict(type='GenerateSnippetLocalizationLabels'),
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
    dict(type='LoadSnippetLocalizationFeature'),
    # dict(type='GenerateTruNetLocalizationLabels'),
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
    videos_per_gpu=128,
    workers_per_gpu=0,
    train_dataloader=dict(drop_last=False),
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
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

# optimizer
optimizer = dict(
    type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005)  # batch_size

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=25)

total_epochs = 1
checkpoint_config = dict(interval=10, filename_tmpl='tem_epoch_{}.pth')

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
