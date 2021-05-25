# model settings
model = dict(
    type="Recognizer2D",
    backbone=dict(
        type="CLIP", pretrained="RN50", freeze=False, fp16_enabled=True
    ),  # output: [batch * segs, 512]
    cls_head=dict(
        type="CLIPHead",
        num_classes=240,
        in_channels=1024,
        consensus=dict(type="AvgConsensus", dim=1),
        dropout_ratio=0.8,
        init_std=0.001,
        fp16_enabled=True,
    ),
)
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips=None)
# dataset settings
dataset_type = "VideoDataset"
data_root = "/mnt/lustre/share_data/MM21-CLASSIFICATION"
data_root_val = "/mnt/lustre/share_data/MM21-CLASSIFICATION"
ann_file_train = "/mnt/lustre/share_data/MM21-CLASSIFICATION/train_anno"
ann_file_val = "/mnt/lustre/share_data/MM21-CLASSIFICATION/val_anno"
ann_file_test = "/mnt/lustre/share_data/MM21-CLASSIFICATION/test_anno"
mc_cfg = dict(
    server_list_cfg="/mnt/lustre/share/memcached_client/server_list.conf",
    client_cfg="/mnt/lustre/share/memcached_client/client.conf",
    sys_path="/mnt/lustre/share/pymc/py3",
)
img_norm_cfg = dict(mean=[104, 117, 128], std=[1, 1, 1], to_bgr=False)
train_pipeline = [
    dict(type="DecordInit", io_backend="memcached", **mc_cfg),
    dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=8),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(
        type="MultiScaleCrop",
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
    ),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="Collect", keys=["imgs", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs", "label"]),
]
val_pipeline = [
    dict(type="DecordInit", io_backend="memcached", **mc_cfg),
    dict(
        type="SampleFrames", clip_len=1, frame_interval=1, num_clips=8, test_mode=True
    ),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="CenterCrop", crop_size=224),
    dict(type="Flip", flip_ratio=0),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="Collect", keys=["imgs", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs"]),
]
test_pipeline = [
    dict(type="DecordInit", io_backend="memcached", **mc_cfg),
    dict(
        type="SampleFrames", clip_len=1, frame_interval=1, num_clips=25, test_mode=True
    ),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="TenCrop", crop_size=224),
    dict(type="Flip", flip_ratio=0),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="Collect", keys=["imgs", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs"]),
]
data = dict(
    videos_per_gpu=128,
    workers_per_gpu=10,
    test_dataloader=dict(videos_per_gpu=2),
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
    ),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
    ),
)
# optimizer
optimizer = dict(
    type="SGD", lr=0.00156, momentum=0.9, weight_decay=0.0005
)  # this lr is used for 4 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy="CosineAnnealing", min_lr=0)
total_epochs = 50
checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=1, metrics=["top_k_accuracy", "mean_class_accuracy"], topk=(1, 5)
)
log_config = dict(
    interval=20, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)
# runtime settings
dist_params = dict(backend="nccl", port=25698)
log_level = "INFO"
work_dir = "./work_dirs/MM21/ds/tsn_clip_r50_1x1x8_50e_0.00625_d0.8"
load_from = None
resume_from = None
workflow = [("train", 1)]
find_unused_parameters = True