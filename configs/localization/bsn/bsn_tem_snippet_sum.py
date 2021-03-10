dataset_type = "SumDataset"
load_type = "LoadSnippetRectifiedFeature"
use_mc = False
array_length = 10
machines = 1
gpus_per_machine = 1
batch_size = 2
split_idx = 0

model = dict(
    type="SumTEM",
    tem_feat_dim=1024,
    tem_hidden_dim=512,
    tem_match_threshold=0.5,
    loss_weight=1,
)
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips="score")
# dataset settings
data_root = "data/summarization"
data_root_val = data_root
ann_file_train = "data/summarization/summe.yml"
ann_file_val = ann_file_train
ann_file_test = ann_file_train

work_dir = f"work_dirs/tem_snippet_sum_dataset/split_{split_idx}"
tem_results_dir = f"{work_dir}/tem_results/"

test_pipeline = [
    dict(type=load_type, use_mc=use_mc, array_length=array_length),
    dict(
        type="Collect",
        keys=["features"],
        meta_name="video_meta",
        meta_keys=["video_name", "segments"],
    ),
    dict(type="ToTensor", keys=["features"]),
]

train_pipeline = [
    dict(type=load_type, use_mc=use_mc, array_length=array_length),
    dict(
        type="Collect",
        keys=["features", "label_action"],
        meta_name="video_meta",
        meta_keys=["video_name"],
    ),
    dict(type="ToTensor", keys=["features", "label_action"]),
    dict(type="ToDataContainer", fields=[dict(key="label_action", stack=False)]),
]

val_pipeline = [
    dict(type=load_type, use_mc=use_mc, array_length=array_length),
    dict(
        type="Collect",
        keys=["features"],
        meta_name="video_meta",
        meta_keys=["video_name", "segments"],
    ),
    dict(type="ToTensor", keys=["features"]),
]

data = dict(
    videos_per_gpu=batch_size,
    workers_per_gpu=0,
    train_dataloader=dict(drop_last=False, shuffle=True),
    val_dataloader=dict(
        videos_per_gpu=1, workers_per_gpu=0, drop_last=False, shuffle=False
    ),
    test_dataloader=dict(
        videos_per_gpu=1, workers_per_gpu=0, drop_last=False, shuffle=False
    ),
    test=dict(
        type=dataset_type,
        split_idx=split_idx,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        data_prefix=data_root_val,
        test_mode=True,
    ),
    val=dict(
        type=dataset_type,
        split_idx=split_idx,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        data_prefix=data_root_val,
        test_mode=True,
    ),
    train=dict(
        type=dataset_type,
        split_idx=split_idx,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        data_prefix=data_root,
        test_mode=False,
    ),
)

# optimizer
optimizer = dict(
    type="SGD",
    lr=0.004 * gpus_per_machine * machines,
    momentum=0.9,
    weight_decay=0.0005,
)  # batch_size
# 0.001 is for batch 256*4=1024
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=5,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5,
    warmup_by_epoch=True,
)

total_epochs = 70
checkpoint_config = dict(interval=5, filename_tmpl="tem_epoch_{}.pth")

log_config = dict(
    interval=2, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)
# runtime settings
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
output_config = dict(out=tem_results_dir, output_format="csv")