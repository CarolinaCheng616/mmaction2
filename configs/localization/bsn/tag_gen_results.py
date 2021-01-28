# dataset settings
dataset_type = 'SnippetDataset'
data_root = 'data/TruNet/train_feature/'
data_root_val = 'data/TruNet/val_feature/'
ann_file_train = 'data/TruNet/train_meta.json'
ann_file_val = 'data/TruNet/val_meta.json'
ann_file_test = 'data/TruNet/val_meta.json'

pgm_work_dir = 'work_dirs/tag_pgm_snippet'
pgm_proposals_dir = f'{pgm_work_dir}/pgm_proposals/'
tag_proposal_result_dir = 'work_dirs/tag_result_snippet'
# tag_pgm_result_dir = f'{tag_proposal_result_dir}/pgm_proposals/'
# output_config = dict(
#     out=f'{tag_proposal_result_dir}/results.json', output_format='json')

tag_results_config = dict(
    threads=8,
    soft_nms_alpha=0.75,
    soft_nms_low_threshold=0.65,
    soft_nms_high_threshold=0.9,
    post_process_top_k=100,
    tag_pgm_result_dir=f'{tag_proposal_result_dir}/nms_pgm_proposals/',
    output_config=dict(
        out=f'{tag_proposal_result_dir}/nms_results.json',
        output_format='json'))

tag_highest_iou_config = dict(
    threads=8,
    post_process_top_k=100,
    tag_pgm_result_dir=f'{tag_proposal_result_dir}/iou_pgm_proposals/',
    output_config=dict(
        out=f'{tag_proposal_result_dir}/iou_results.json',
        output_format='json'))

highest_iou = True
