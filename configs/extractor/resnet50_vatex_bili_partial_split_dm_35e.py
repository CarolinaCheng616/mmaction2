model = dict(
    type="VideoExtractor",
    backbone=dict(type="ResNet", pretrained=None, depth=50, norm_eval=False),
)
train_cfg = None
test_cfg = None
dataset_type = "MyDataset"
data_root = "/mnt/lustre/share_data/vatex"
data_root_val = data_root
feature_prefix = "/mnt/lustre/share_data/vatex/vatex_features_bili_partial_split_dm_35e"
feature_prefix_val = feature_prefix
ann_file_train = "/mnt/lustre/chenghaoyue/partial_vatex_training_v1.0.json /mnt/lustre/chenghaoyue/partial_vatex_validation_v1.0.json"
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
    dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=32),
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
    dict(type="Resize", scale=(112, 112), keep_ratio=False, lazy=True),
    dict(type="Flip", flip_ratio=0.5, lazy=True),
    dict(type="Fuse"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="Collect", keys=["imgs"], meta_keys=["featurepath"]),
    dict(type="ToTensor", keys=["imgs"]),
]
val_pipeline = [
    dict(type="DecordInit", io_backend="memcached", **mc_cfg),
    dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=32),
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
    dict(type="Resize", scale=(112, 112), keep_ratio=False, lazy=True),
    dict(type="Flip", flip_ratio=0.5, lazy=True),
    dict(type="Fuse"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="Collect", keys=["imgs"], meta_keys=["featurepath"]),
    dict(type="ToTensor", keys=["imgs"]),
]
test_pipeline = [
    dict(type="DecordInit", io_backend="memcached", **mc_cfg),
    dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=32),
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
    dict(type="Resize", scale=(112, 112), keep_ratio=False, lazy=True),
    dict(type="Flip", flip_ratio=0.5, lazy=True),
    dict(type="Fuse"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="Collect", keys=["imgs"], meta_keys=["featurepath"]),
    dict(type="ToTensor", keys=["imgs"]),
]

data = dict(
    videos_per_gpu=32,
    workers_per_gpu=5,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        feature_prefix=feature_prefix,
        pipeline=train_pipeline,
        data_prefix=data_root,
    ),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        feature_prefix=feature_prefix_val,
        pipeline=val_pipeline,
        data_prefix=data_root_val,
    ),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        feature_prefix=feature_prefix_val,
        pipeline=test_pipeline,
        data_prefix=data_root_val,
    ),
)

optimizer = dict(type="SGD", lr=0.03, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(
    policy="CosineAnnealing",
    min_lr=0,
    warmup="linear",
    warmup_by_epoch=True,
    warmup_iters=1,
)
total_epochs = 1
checkpoint_config = dict(interval=5)
eval_config = dict(metrics=None, metric_options=None, logger=None)
log_config = dict(interval=100000000, hooks=[dict(type="TextLoggerHook")])
dist_params = dict(backend="nccl")
log_level = "INFO"
work_dir = "./work_dirs/vatex_features"
load_from = None
resume_from = None
workflow = [("train", 1)]
find_unused_parameters = True
