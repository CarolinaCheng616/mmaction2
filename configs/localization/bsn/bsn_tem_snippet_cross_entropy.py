dataset_type = 'SnippetSRDataset'
load_type = 'LoadSnippetRectifiedFeature'
use_mc = False
array_length = 10
machines = 1
gpus_per_machine = 8
batch_size = 4096

if load_type == 'LoadSnippetRectifiedFeature':  # feature.shape: 4096, 3+temporal+3
    data_root = 'data/TruNet/sup_train_feature/'
    data_root_val = 'data/TruNet/sup_val_feature/'
elif load_type == 'LoadSnippetFeature':  # feature.shape: temporal, 4096
    data_root = 'data/TruNet/train_feature/'
    data_root_val = 'data/TruNet/val_feature/'
else:
    raise ValueError(f'wrong load_feature name {load_type} in bsn_tem_snippet')

model = dict(
    type='SnippetTEMSR',
    tem_feat_dim=4096,
    tem_hidden_dim=512,
    tem_match_threshold=0.5,
    loss_weight=1)
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips='score')
# dataset settings
ann_file_train = 'data/TruNet/train_meta.json'
ann_file_val = 'data/TruNet/val_meta.json'
ann_file_test = 'data/TruNet/val_meta.json'
# ann_file_val = ann_file_train
# ann_file_test = ann_file_train

work_dir = 'work_dirs/tem_snippet_sample_ratio/'
tem_results_dir = f'{work_dir}/tem_results/'

test_pipeline = [
    dict(type=load_type,
         use_mc=use_mc,
         array_length=array_length),
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
        use_mc=use_mc,
        array_length=array_length),
    dict(
        type='Collect',
        keys=['raw_feature', 'cate'],
        meta_name='video_meta',
        meta_keys=['video_name', 'duration_second', 'snippet_length']),
    dict(
        type='ToTensor',
        keys=['raw_feature', 'cate']),
    dict(
        type='ToDataContainer',
        fields=[
            dict(key='cate', stack=False)
        ])
]
val_pipeline = [
    dict(type=load_type,
         use_mc=use_mc,
         array_length=array_length),
    dict(
        type='Collect',
        keys=['raw_feature', 'cate'],
        meta_name='video_meta',
        meta_keys=['video_name', 'duration_second', 'snippet_length']),
    dict(
        type='ToTensor',
        keys=['raw_feature', 'cate']),
    dict(
        type='ToDataContainer',
        fields=[
            dict(key='cate', stack=False)
        ])
]

data = dict(
    videos_per_gpu=batch_size,
    workers_per_gpu=0,
    train_dataloader=dict(drop_last=False, shuffle=False),
    val_dataloader=dict(
        videos_per_gpu=batch_size,
        workers_per_gpu=0,
        drop_last=False,
        shuffle=False),
    test_dataloader=dict(
        videos_per_gpu=batch_size,
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
    type='SGD', lr=0.04 * gpus_per_machine * machines, momentum=0.9, weight_decay=0.0005)  # batch_size

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5,
    warmup_by_epoch=True)

total_epochs = 70
checkpoint_config = dict(interval=5, filename_tmpl='tem_epoch_{}_lr0.04.pth')

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
