bert_path = "work_dirs/bert_model"
dataset_type = "VideoTextDataset"
data_root = "data/usv/videos"
data_root_val = "data/usv/videos"
ann_file_train = "data/usv/usv_train_list_frame_text_title"
ann_file_val = "data/usv/usv_val_list_frame_text_title"
ann_file_test = "data/usv/usv_val_list_frame_text_title"
train_dataset_size = 20000
val_dataset_size = 24472
train = True

model = dict(
    type="VideoTextMatcherBankE2E",
    backbone1=dict(type="ResNet", pretrained=None, depth=50, norm_eval=False),
    backbone2=dict(type="BERT", pretrained=bert_path, freeze=True),
    head=dict(type="MBNCEHead", temperature=0.05),
    fp16_enabled=False,
    dataset_size=train_dataset_size if train else val_dataset_size,
    bank_size=2048,
    img_feat_dim=2048,
    text_feat_dim=768,
    feature_dim=256,
    init_std=0.01,
)
train_cfg = None
test_cfg = None
mc_cfg = dict(
    server_list_cfg="/mnt/lustre/share/memcached_client/server_list.conf",
    client_cfg="/mnt/lustre/share/memcached_client/client.conf",
    sys_path="/mnt/lustre/share/pymc/py3",
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False
)
train_pipeline = [
    dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=8),
    dict(type="RawFrameDecode", io_backend="memcached", **mc_cfg),
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
    dict(type="LoadTexts", sample_mode="number", sample_number=1),
    dict(type="TextTokenize", tokenizer_dir=bert_path),
    dict(type="Collect", keys=["imgs", "texts_item", "idxes"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs"]),
]
val_pipeline = [
    dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=8),
    dict(type="RawFrameDecode", io_backend="memcached", **mc_cfg),
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
    dict(type="LoadTexts", sample_mode="number", sample_number=1),
    dict(type="TextTokenize", tokenizer_dir=bert_path),
    dict(type="Collect", keys=["imgs", "texts_item"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs"]),
]
test_pipeline = [
    dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=8),
    dict(type="RawFrameDecode", io_backend="memcached", **mc_cfg),
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
    dict(type="LoadTexts", sample_mode="number", sample_number=1),
    dict(type="TextTokenize", tokenizer_dir=bert_path),
    dict(type="Collect", keys=["imgs", "texts_item"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs"]),
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=5,
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
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
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
total_epochs = 100
checkpoint_config = dict(interval=5)
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
work_dir = "./work_dirs/v_t_memory_bank_e2e_100e"
load_from = None
resume_from = None
workflow = [("train", 1)]
find_unused_parameters = True
