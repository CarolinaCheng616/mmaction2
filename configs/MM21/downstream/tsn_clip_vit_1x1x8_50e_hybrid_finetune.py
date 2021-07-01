img_size = 224

# model settings
model = dict(
    type="Recognizer2D",
    backbone=dict(
        type="CLIPViT", pretrained="ViT-B/32", freeze=False, fp16_enabled=True
    ),  # output: [batch * segs, 768]
    cls_head=dict(
        type="CLIPHead",
        num_classes=240,
        in_channels=768,
        consensus=dict(type="AvgConsensus", dim=1),
        dropout_ratio=0.8,
        init_std=0.02,
        fp16_enabled=True,
    ),
)
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips=None)
# dataset settings
dataset_type = "VideoDataset"
data_root = "data/mm21"
data_root_val = "data/mm21"
ann_file_train = "data/mm21/train_anno"
ann_file_val = "data/mm21/val_anno"
ann_file_test = "data/mm21/val_anno"
mc_cfg = dict(
    server_list_cfg="/mnt/lustre/share/memcached_client/server_list.conf",
    client_cfg="/mnt/lustre/share/memcached_client/client.conf",
    sys_path="/mnt/lustre/share/pymc/py3",
)
# img_norm_cfg = dict(mean=[104, 117, 128], std=[1, 1, 1], to_bgr=False)
img_norm_cfg = dict(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)
train_pipeline = [
    dict(type="DecordInit", io_backend="memcached", **mc_cfg),
    dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=8),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(
        type="MultiScaleCrop",
        input_size=img_size,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
    ),
    dict(type="Resize", scale=(img_size, img_size), keep_ratio=False),
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
    dict(type="CenterCrop", crop_size=img_size),
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
    dict(type="TenCrop", crop_size=img_size),
    dict(type="Flip", flip_ratio=0),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="Collect", keys=["imgs", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs"]),
]
data = dict(
    videos_per_gpu=64,
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
# # optimizer
# optimizer = dict(
#     type="SGD", lr=0.00625, momentum=0.9, weight_decay=0.0005
# )  # this lr is used for 4 gpus
# optimizer
optimizer = dict(
    type="SGD",
    lr=0.00125,  # for 128
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True,
    paramwise_cfg=dict(
        custom_keys={
            ".backbone.cls_token": dict(decay_mult=0.0),
            ".backbone.pos_embed": dict(decay_mult=0.0),
        }
    ),
)
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
eval_config = dict(metrics=["top_k_accuracy", "mean_class_accuracy"])
output_config = dict(
    out="/mnt/lustre/share_data/MM21-CLASSIFICATION/clip_vit_self_training_finetune.pkl"
)
# runtime settings
dist_params = dict(backend="nccl", port=25698)
log_level = "INFO"
work_dir = "./work_dirs/MM21/ds/tsn_clipvit_1x1x8_50e_hybrid_finetune"
load_from = "work_dirs/MM21/ds/tsn_clipvit_1x1x8_50e_hybrid_sample/epoch_50.pth"
resume_from = None
workflow = [("train", 1)]
find_unused_parameters = True