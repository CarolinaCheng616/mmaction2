# dataset settings
dataset_type = 'SnippetDataset'
data_root = 'data/TruNet/train_feature/'
data_root_val = 'data/TruNet/val_feature/'
ann_file_train = 'data/TruNet/train_meta.json'
ann_file_val = 'data/TruNet/val_meta.json'
ann_file_test = 'data/TruNet/val_meta.json'

pgm_work_dir = 'work_dirs/tag_pgm_snippet'
pgm_proposals_dir = f'{pgm_work_dir}/pgm_proposals/'
tem_results_dir = 'work_dirs/tem_snippet/tem_results/'
tag_pgm_score_nms_dir = 'work_dirs/tag_pgm_score_nms_snippet'
tag_pgm_iou_nms_dir = 'work_dirs/tag_pgm_iou_nms_snippet'

topk = 100
threads = 8

tag_score_nms_config = dict(
    thread_num=threads,
    soft_nms_alpha=0.75,
    soft_nms_low_threshold=0.65,
    soft_nms_high_threshold=0.9,
    top_k=topk,
    proposals_dir=f'{tag_pgm_score_nms_dir}/pgm_proposals/',
    features_dir=f'{tag_pgm_score_nms_dir}/pgm_features/',
    output_config=dict(
        out=f'{tag_pgm_score_nms_dir}/results.json', output_format='json'))
tag_iou_nms_config = dict(
    thread_num=threads,
    soft_nms_alpha=0.75,
    soft_nms_low_threshold=0.65,
    soft_nms_high_threshold=0.9,
    top_k=topk,
    proposals_dir=f'{tag_pgm_iou_nms_dir}/pgm_proposals/',
    features_dir=f'{tag_pgm_iou_nms_dir}/pgm_features/',
    output_config=dict(
        out=f'{tag_pgm_iou_nms_dir}/results.json', output_format='json'))
feature_kwargs = dict(
    top_k=topk,
    bsp_boundary_ratio=0.2,
    num_sample_start=8,
    num_sample_end=8,
    num_sample_action=16,
    num_sample_interp=3)