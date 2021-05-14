syncBN = True

model = dict(
    type="Video3DMatcherNSim",
    backbone=dict(
        type="ResNet3dSlowFast",
        pretrained=None,
        resample_rate=8,  # tau
        speed_ratio=8,  # alpha
        channel_ratio=8,  # beta_inv
        slow_pathway=dict(
            type="resnet3d",
            depth=50,
            pretrained=None,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            norm_eval=False,
        ),
        fast_pathway=dict(
            type="resnet3d",
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            norm_eval=False,
        ),
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
dataset_type = "VideoDataset"
data_root = "/mnt/lustre/share_data/MM21-CLASSIFICATION"
data_root_val = "/mnt/lustre/share_data/MM21-CLASSIFICATION"
ann_file_train = "/mnt/lustre/share_data/MM21-CLASSIFICATION/full_self_sup"
ann_file_val = "/mnt/lustre/share_data/MM21-CLASSIFICATION/full_self_sup"
ann_file_test = "/mnt/lustre/share_data/MM21-CLASSIFICATION/full_self_sup"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False
)
mc_cfg = dict(
    server_list_cfg="/mnt/lustre/share/memcached_client/server_list.conf",
    client_cfg="/mnt/lustre/share/memcached_client/client.conf",
    sys_path="/mnt/lustre/share/pymc/py3",
)
train_pipeline = [
    dict(type="DecordInit", io_backend="memcached", **mc_cfg),
    dict(type="SampleFrames", clip_len=32, frame_interval=2, num_clips=2),
    dict(type="DecordDecode"),
    dict(type="RandomResizedCrop"),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="RandomColorJitter", color_space_aug=True, p=0.8),
    dict(type="RandomGaussianBlur", sigma_min=0.1, sigma_max=2.0, p=0.5),
    dict(type="RandomSolarization", p=0.2),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="Collect", keys=["imgs"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs"]),
]
val_pipeline = [
    dict(type="DecordInit", io_backend="memcached", **mc_cfg),
    dict(type="SampleFrames", clip_len=32, frame_interval=2, num_clips=2),
    dict(type="DecordDecode"),
    dict(type="RandomResizedCrop"),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="RandomColorJitter", color_space_aug=True, p=0.8),
    dict(type="RandomGaussianBlur", sigma_min=0.1, sigma_max=2.0, p=0.5),
    dict(type="RandomSolarization", p=0.2),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="Collect", keys=["imgs"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs"]),
]
test_pipeline = [
    dict(type="DecordInit", io_backend="memcached", **mc_cfg),
    dict(type="SampleFrames", clip_len=32, frame_interval=2, num_clips=2),
    dict(type="DecordDecode"),
    dict(type="RandomResizedCrop"),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="RandomColorJitter", color_space_aug=True, p=0.8),
    dict(type="RandomGaussianBlur", sigma_min=0.1, sigma_max=2.0, p=0.5),
    dict(type="RandomSolarization", p=0.2),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="Collect", keys=["imgs"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs"]),
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=10,
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
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

momentum_update_config = dict(type="BYOLHook", end_momentum=1.0, update_interval=1)
# optimizer
optimizer = dict(
    type="SGD", lr=0.025, momentum=0.9, weight_decay=0.0001
)  # this lr is used for 4 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
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
work_dir = "work_dirs/MM21/pt/10w_v_v_e2e_50e_neg_sim"
load_from = "ckpt/slowfast_r50_4x16x1_256e_kinetics400_rgb_20200704-bcde7ed7.pth"
resume_from = None
workflow = [("train", 1)]
