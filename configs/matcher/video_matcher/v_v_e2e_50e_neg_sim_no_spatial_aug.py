bert_path = "work_dirs/bert_model"
syncBN = True

model = dict(
    type="VideoMatcherNSim",
    backbone=dict(
        type="ResNet",
        pretrained="torchvision://resnet50",
        depth=50,
        norm_cfg=dict(type="SyncBN", requires_grad=True, eps=1e-5)
        if syncBN
        else dict(type="BN2d", requires_grad=True),
        norm_eval=False,
    ),
    head=dict(type="NegSimVideoHead"),
    fp16_enabled=False,
    img_feat_dim=2048,
    feature_dim=256,
    init_std=0.01,
    gather_flag=False,
    syncBN=syncBN,
    base_momentum=0.996,
)
train_cfg = None
test_cfg = None
dataset_type = "VideoClipDataset"
data_root = "/mnt/lustre/share_data/MM21-PRETRAIN/video"
data_root_val = data_root
ann_file_train = "/mnt/lustre/share_data/MM21-PRETRAIN/video/full_self_sup_anno"
ann_file_val = ann_file_train
ann_file_test = ann_file_train
mc_cfg = dict(
    server_list_cfg="/mnt/lustre/share/memcached_client/server_list.conf",
    client_cfg="/mnt/lustre/share/memcached_client/client.conf",
    sys_path="/mnt/lustre/share/pymc/py3",
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False
)
train_pipeline = [
    dict(type="DecordInit", io_backend="memcached", **mc_cfg),
    dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=8),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256), lazy=True),
    dict(
        type="MultiScaleCrop",
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        lazy=True,
    ),
    dict(type="Resize", scale=(224, 224), keep_ratio=False, lazy=True),
    dict(type="Flip", flip_ratio=0.5, lazy=True),
    dict(type="Fuse"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="Collect", keys=["imgs"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs"]),
]
val_pipeline = [
    dict(type="DecordInit", io_backend="memcached", **mc_cfg),
    dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=8),
    dict(type="DecordDecode"),
    dict(type="RandomResizedCrop"),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="RandomColorJitter", color_space_aug=True, p=0.8),
    dict(type="RandomGaussianBlur", sigma_min=0.1, sigma_max=2.0, p=0.5),
    dict(type="RandomSolarization", p=0.2),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="Collect", keys=["imgs"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs"]),
]
test_pipeline = [
    dict(type="DecordInit", io_backend="memcached", **mc_cfg),
    dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=8),
    dict(type="DecordDecode"),
    dict(type="RandomResizedCrop"),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="RandomColorJitter", color_space_aug=True, p=0.8),
    dict(type="RandomGaussianBlur", sigma_min=0.1, sigma_max=2.0, p=0.5),
    dict(type="RandomSolarization", p=0.2),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="Collect", keys=["imgs"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs"]),
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        test_mode=True,
    ),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

momentum_update_config = dict(type="BYOLHook", end_momentum=1.0, update_interval=1)
optimizer = dict(
    type="SGD", lr=0.03, momentum=0.9, weight_decay=0.0001
)  # lr for 2*8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(
    policy="CosineAnnealing",
    min_lr=0,
    warmup="linear",
    warmup_by_epoch=True,
    warmup_iters=1,
)
total_epochs = 50
checkpoint_config = dict(interval=10)
evaluation = dict(
    interval=1,
    key_indicator="vt_mean_rk_full",
    metrics=["vt_retrieval_metrics_full", "tv_retrieval_metrics_full"],
)
log_config = dict(
    interval=1, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)
dist_params = dict(backend="nccl")
log_level = "INFO"
work_dir = "./work_dirs/MM21/pt/3m_v_v_e2e_50e_neg_sim_no_s_aug"
load_from = None
resume_from = None
workflow = [("train", 1)]
find_unused_parameters = True