topk = 1000
threads = 8
origin = True
offset = False

# dataset settings
dataset_type = 'SnippetDataset'
data_root = 'data/TruNet/train_feature/'
data_root_val = 'data/TruNet/val_feature/'
ann_file_train = 'data/TruNet/train_meta.json'
ann_file_val = 'data/TruNet/val_meta.json'
ann_file_test = 'data/TruNet/val_meta.json'

pgm_work_dir = 'work_dirs/tag_pgm_snippet'

if offset:
    pgm_work_dir += '_offset'

pgm_proposals_dir = f'{pgm_work_dir}/pgm_proposals/'

if origin:
    train_features_dir = 'data/TruNet/train_feature/'
    test_features_dir = 'data/TruNet/val_feature/'
    tag_pgm_score_nms_dir = f'work_dirs/' \
                            f'tag_pgm_score_nms_snippet_orifeat_{topk}'
    tag_pgm_iou_nms_dir = f'work_dirs/tag_pgm_iou_nms_snippet_orifeat_{topk}'
else:
    train_features_dir = 'work_dirs/tem_snippet/tem_results/'
    test_features_dir = train_features_dir
    tag_pgm_score_nms_dir = f'work_dirs/tag_pgm_score_nms_snippet_{topk}'
    tag_pgm_iou_nms_dir = f'work_dirs/tag_pgm_iou_nms_snippet_{topk}'
if offset:
    tag_pgm_iou_nms_dir += '_offset'
    tag_pgm_score_nms_dir += '_offset'

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
