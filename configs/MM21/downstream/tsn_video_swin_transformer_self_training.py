GPUS = 8
img_size = 224
# model settings
model = dict(
    type="RecognizerSemiSupervised3D",
    teacher_backbone=dict(
        type="SwinTransformer3D",
        pretrained="ckpt/video_swin_base_patch244_window877_kinetics600_22k.pth",
    ),  # output: ([batch, 768], [batch, 768])
    student_backbone=dict(
        type="SwinTransformer3D",
        pretrained="ckpt/video_swin_base_patch244_window877_kinetics600_22k.pth",
    ),  # output: ([batch, 768], [batch, 768])
    teacher_cls_head=dict(
        type="I3DHead",
        num_classes=240,
        in_channels=1024,
        dropout_ratio=0.8,
        init_std=0.02,
    ),
    student_cls_head=dict(
        type="I3DHead",
        num_classes=240,
        in_channels=1024,
        dropout_ratio=0.8,
        init_std=0.02,
    ),
    distill_head=dict(type="DistillHead", alpha=0.5, temperature=1, labeled=False),
    fp16_enabled=False,
)
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips="score")
# dataset settings
dataset_type = "TwoVideoDataset"
data_root = "data/mm21"
data_root_val = "data/mm21"
ann_file_train = "data/mm21/train_anno data/mm21/imagenet_distill_hard_0.95_resample"
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
    dict(type="SampleFrames", clip_len=8, frame_interval=4, num_clips=1),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, img_size)),
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
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="Collect", keys=["imgs", "label"], meta_keys=["labeled"]),
    dict(type="ToTensor", keys=["imgs", "label"]),
]
val_pipeline = [
    dict(type="DecordInit", io_backend="memcached", **mc_cfg),
    dict(
        type="SampleFrames", clip_len=8, frame_interval=4, num_clips=1, test_mode=True
    ),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, img_size)),
    dict(type="CenterCrop", crop_size=img_size),
    dict(type="Flip", flip_ratio=0),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="Collect", keys=["imgs", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs"]),
]
test_pipeline = [
    dict(type="DecordInit", io_backend="memcached", **mc_cfg),
    dict(
        type="SampleFrames", clip_len=8, frame_interval=4, num_clips=4, test_mode=True
    ),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, img_size)),
    dict(type="TenCrop", crop_size=img_size),
    dict(type="Flip", flip_ratio=0),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="Collect", keys=["imgs", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs"]),
]
data = dict(
    videos_per_gpu=9,
    workers_per_gpu=7,
    train_dataloader=dict(shuffle=False),
    test_dataloader=dict(videos_per_gpu=2),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
        test_mode=False,
        gpus=GPUS,
    ),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        test_mode=True,
        gpus=1,
    ),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        test_mode=True,
        gpus=1,
    ),
)
# # optimizer
# optimizer = dict(
#     type="SGD", lr=0.00625, momentum=0.9, weight_decay=0.0005
# )  # this lr is used for 4 gpus
# optimizer
optimizer = dict(
    type="SGD",
    lr=0.0028125,  # for 72
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
    out="/mnt/lustre/share_data/MM21-CLASSIFICATION/test_result/video_swin_semi_sup_result.pkl"
)

# runtime settings
dist_params = dict(backend="nccl", port=25698)
log_level = "INFO"
work_dir = "work_dirs/MM21/st/tsn_video_swin_base_1x1x8_50e_self_training"
load_from = "ckpt/video_swin_teacher.pth"
resume_from = None
workflow = [("train", 1)]
find_unused_parameters = True
