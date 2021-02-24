# tag proposal generation
# proposals used to train are pre soft nms by action_score or iou
# proposals used to test are totally loaded
# binary logistic regression loss at a given threshold
# tem trained on snippets
# pgm features are original features
# model settings
model = dict(
    type='OriFeatBNPEMReg',
    pem_feat_dim=4096,
    pem_hidden_dim1=256,
    pem_hidden_dim2=256,
    pem_u_ratio_m=1,
    pem_u_ratio_l=2,
    pem_high_temporal_iou_threshold=0.8,
    pem_low_temporal_iou_threshold=0.3,
    soft_nms_alpha=0.75,
    soft_nms_low_threshold=0.65,
    soft_nms_high_threshold=0.6,
    post_process_top_k=100,
    fc_ratio=1,
    classify_ratio=1,
    regression_ratio=1,
    loss_cls=dict(
        type='BinaryThresholdClassificationLoss',
        low_threshold=0.3,
        high_threshold=0.7),
    classify_loss_ratio=1,
    regression_loss_ratio=5,
    offset_scale=1000)
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips='score')
# dataset settings
dataset_type = 'TruNetDataset'
data_root = 'data/TruNet/train_feature/'
data_root_val = 'data/TruNet/val_feature/'
ann_file_train = 'data/TruNet/train_meta.json'
ann_file_val = 'data/TruNet/val_meta.json'
ann_file_test = 'data/TruNet/val_meta.json'

lr = 0.1

# for train
# pgm_work_dir = f'work_dirs/tag_pgm_snippet_offset_clipped_iou_nms/'
# work_dir = f'work_dirs/tag_pem_bn_iou_nms_btc_snippet_offset_orifeat_clipped_lr0.1/'
# pgm_proposals_dir = f'{pgm_work_dir}/pgm_proposals/'
# pgm_features_dir = f'{pgm_work_dir}/pgm_features/'

# for test
pgm_work_dir = f'work_dirs/tag_pgm_snippet_clipped_de_duplicate/'
work_dir = f'work_dirs/tag_pem_bn_iou_nms_btc_snippet_offset_orifeat_clipped_lr0.1/'
pgm_proposals_dir = f'{pgm_work_dir}/pgm_proposals/'
pgm_features_dir = f'{pgm_work_dir}/pgm_features/'

output_config = dict(out=f'{work_dir}/nms_top100_results.json', output_format='json')

test_pipeline = [
    dict(
        type='LoadTAGProposalsOffset',
        top_k=-1,
        pgm_proposals_dir=pgm_proposals_dir,
        pgm_features_dir=pgm_features_dir,
        use_mc=False),
    dict(
        type='Collect',
        keys=['bsp_feature', 'tmin', 'tmax'],
        meta_name='video_meta',
        meta_keys=['video_name', 'duration_second']),
    dict(type='ToTensor', keys=['bsp_feature'])
]
train_pipeline = [
    dict(
        type='LoadTAGProposalsOffset',
        top_k=-1,
        pgm_proposals_dir=pgm_proposals_dir,
        pgm_features_dir=pgm_features_dir,
        use_mc=False),
    dict(
        type='Collect',
        keys=['bsp_feature', 'reference_temporal_iou', 'offset'],
        meta_name='video_meta',
        meta_keys=[]),
    dict(type='ToTensor', keys=['bsp_feature', 'reference_temporal_iou', 'offset']),
    dict(
        type='ToDataContainer',
        fields=(dict(key='bsp_feature', stack=False),
                dict(key='reference_temporal_iou', stack=False),
                dict(key='offset', stack=False)))
]
val_pipeline = [
    dict(
        type='LoadTAGProposalsOffset',
        top_k=-1,
        pgm_proposals_dir=pgm_proposals_dir,
        pgm_features_dir=pgm_features_dir,
        use_mc=False),
    dict(
        type='Collect',
        keys=['bsp_feature', 'tmin', 'tmax'],
        meta_name='video_meta',
        meta_keys=['video_name', 'duration_second']),
    dict(type='ToTensor', keys=['bsp_feature'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=8,
    train_dataloader=dict(drop_last=False),
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
gpu_per_node = 2
machines = 3
optimizer = dict(
    type='SGD',
    lr=lr * data['videos_per_gpu'] * gpu_per_node * machines / 256,
    momentum=0.9,
    weight_decay=0.0005)

optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(policy='step', step=210)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5,
    warmup_by_epoch=True)

total_epochs = 200
checkpoint_config = dict(interval=2, filename_tmpl='pem_epoch_{}.pth')

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
