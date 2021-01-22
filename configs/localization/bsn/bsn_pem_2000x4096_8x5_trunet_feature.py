# model settings
model = dict(
    type='PEM',
    pem_feat_dim=32,
    pem_hidden_dim=256,
    pem_u_ratio_m=1,
    pem_u_ratio_l=2,
    pem_high_temporal_iou_threshold=0.6,
    pem_low_temporal_iou_threshold=0.2,
    soft_nms_alpha=0.75,
    soft_nms_low_threshold=0.65,
    soft_nms_high_threshold=0.9,
    post_process_top_k=100,
    fc1_ratio=1,
    fc2_ratio=1)
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips='score')
# dataset settings
dataset_type = 'TruNetDataset'
data_root = 'data/TruNet/train_mean_2000/'
data_root_val = 'data/TruNet/val_mean_2000/'
# data_root_val = data_root
ann_file_train = 'data/TruNet/train_meta.json'
ann_file_val = 'data/TruNet/val_meta.json'
ann_file_test = 'data/TruNet/val_meta.json'
# ann_file_val = ann_file_train
# ann_file_test = ann_file_train

work_dir = 'work_dirs/bsn_2000x4096_8x5_trunet_feature/'
pgm_proposals_dir = f'{work_dir}/pgm_proposals/'
pgm_features_dir = f'{work_dir}/pgm_features/'

test_pipeline = [
    dict(
        type='LoadProposals',
        top_k=500,
        pgm_proposals_dir=pgm_proposals_dir,
        pgm_features_dir=pgm_features_dir),
    dict(
        type='Collect',
        keys=['bsp_feature', 'tmin', 'tmax', 'tmin_score', 'tmax_score'],
        meta_name='video_meta',
        meta_keys=[
            'video_name', 'duration_second', 'annotations'
        ]),
    dict(type='ToTensor', keys=['bsp_feature'])
]
train_pipeline = [
    dict(
        type='LoadProposals',
        top_k=500,
        pgm_proposals_dir=pgm_proposals_dir,
        pgm_features_dir=pgm_features_dir),
    dict(
        type='Collect',
        keys=['bsp_feature', 'reference_temporal_iou'],
        meta_name='video_meta',
        meta_keys=[]),
    dict(type='ToTensor', keys=['bsp_feature', 'reference_temporal_iou']),
    dict(
        type='ToDataContainer',
        fields=(dict(key='bsp_feature', stack=False),
                dict(key='reference_temporal_iou', stack=False)))
]
val_pipeline = [
    dict(
        type='LoadProposals',
        top_k=500,
        pgm_proposals_dir=pgm_proposals_dir,
        pgm_features_dir=pgm_features_dir),
    dict(
        type='Collect',
        keys=['bsp_feature', 'tmin', 'tmax', 'tmin_score', 'tmax_score'],
        meta_name='video_meta',
        meta_keys=[
            'video_name', 'duration_second', 'annotations'
        ]),
    dict(type='ToTensor', keys=['bsp_feature'])
]
data = dict(
    videos_per_gpu=256,
    workers_per_gpu=8,
    train_dataloader=dict(drop_last=True),
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        data_prefix=data_root_val),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        data_prefix=data_root_val),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        data_prefix=data_root))

# optimizer
# optimizer = dict(
#     type='Adam', lr=0.01, weight_decay=0.00001)  # this lr is used for 1 gpus
optimizer = dict(
    type='SGD', lr=0.001 * 256 * 1 / 256, momentum=0.9,
    weight_decay=0.0005
)

optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(policy='step', step=())

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5,
    warmup_by_epoch=True)

total_epochs = 70
checkpoint_config = dict(interval=10, filename_tmpl='pem_epoch_{}.pth')

# evaluation = dict(interval=1, metrics=['AR@AN'])

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
output_config = dict(out=f'{work_dir}/results.json', output_format='json')