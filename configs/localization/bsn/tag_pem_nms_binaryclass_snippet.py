# tag proposal generation
# proposals used to train are pre soft nms by action_score or iou
# proposals used to test are totally loaded
# binary logistic regression loss at a given threshold
# tem trained on snippets
# model settings
model = dict(
    type='ClassifyPEM',
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
    fc2_ratio=1,
    loss_cls=dict(type='BinaryLogisticRegressionLoss'))
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

proposal_topk = 500
lr = 0.1
if model['loss_cls']['type'] == 'BinaryLogisticRegressionLoss':
    loss_cls = 'blr'
elif model['loss_cls']['type'] == 'BinaryThresholdClassificationLoss':
    loss_cls = 'btc'
else:
    loss_cls = ''

# for score nms
# nms_type = 'score'
# for iou nms
nms_type = 'iou'

# for train
pgm_work_dir = f'work_dirs/tag_pgm_{nms_type}_nms_snippet_{proposal_topk}/'
work_dir = f'work_dirs/tag_pem_bn_{nms_type}_nms_' \
           f'{proposal_topk}_{loss_cls}_snippet_lr{lr}/'
pgm_proposals_dir = f'{pgm_work_dir}/pgm_proposals/'
pgm_features_dir = f'{pgm_work_dir}/pgm_features/'

# for test
# pgm_work_dir = 'work_dirs/tag_pgm_snippet/'
# work_dir = f'work_dirs/tag_pem_bn_{nms_type}_nms_' \
#            f'{proposal_topk}_{loss_cls}_snippet_lr{lr}/'
# pgm_proposals_dir = f'{pgm_work_dir}/pgm_proposals/'
# pgm_features_dir = f'{pgm_work_dir}/pgm_features/'

output_config = dict(out=f'{work_dir}/results.json', output_format='json')

test_pipeline = [
    dict(
        type='LoadTAGProposals',
        # top_k=500,
        pgm_proposals_dir=pgm_proposals_dir,
        pgm_features_dir=pgm_features_dir),
    dict(
        type='Collect',
        keys=['bsp_feature', 'tmin', 'tmax'],
        meta_name='video_meta',
        meta_keys=['video_name', 'duration_second']),
    dict(type='ToTensor', keys=['bsp_feature'])
]
train_pipeline = [
    dict(
        type='LoadTAGProposals',
        # top_k=500,
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
        type='LoadTAGProposals',
        # top_k=500,
        pgm_proposals_dir=pgm_proposals_dir,
        pgm_features_dir=pgm_features_dir),
    dict(
        type='Collect',
        keys=['bsp_feature', 'tmin', 'tmax'],
        meta_name='video_meta',
        meta_keys=['video_name', 'duration_second']),
    dict(type='ToTensor', keys=['bsp_feature'])
]
data = dict(
    videos_per_gpu=64,
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
gpu_per_node = 4
machines = 1
optimizer = dict(
    type='SGD',
    lr=lr * data['videos_per_gpu'] * gpu_per_node * machines / 256,
    momentum=0.9,
    weight_decay=0.0005)

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=110)

total_epochs = 200
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
