# dataset settings
dataset_type = "DmDataset"
data_root = "/mnt/lustre/share_data/bilibili_parse_xml"
data_root_val = data_root
ann_file_train = None
ann_file_val = ann_file_train
ann_file_test = ann_file_val

# model settings
model = dict(
    type="BertExtractor",
    bert_path="/mnt/lustre/share_data/liwei_to_haoyue/bert_model",
    bert_backbone=dict(
        type="BERT", pretrained="/mnt/lustre/share_data/liwei_to_haoyue/bert_model"
    ),
    new_path="/mnt/lustre/share_data/bilibili_text_feature/",
)
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips="score")


test_pipeline = [
    dict(type="LoadDmText"),
    dict(
        type="Collect",
        keys=[],
        meta_name="video_meta",
        meta_keys=["times", "dms", "path"],
    ),
]
train_pipeline = test_pipeline
val_pipeline = test_pipeline
data = dict(
    videos_per_gpu=1,
    workers_per_gpu=0,
    train_dataloader=dict(drop_last=False),
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        data_prefix=data_root_val,
    ),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        data_prefix=data_root_val,
    ),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        data_prefix=data_root,
    ),
)
# optimizer
optimizer = dict(
    type="Adam", lr=0.001, weight_decay=0.0001
)  # this lr is used for 2 gpus
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy="step", step=7)
total_epochs = 9

# runtime settings
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook")])
work_dir = "work_dirs/bert_bilibili/"
output_config = dict(out=None, output_format="json")
dist_params = dict(backend="nccl")
