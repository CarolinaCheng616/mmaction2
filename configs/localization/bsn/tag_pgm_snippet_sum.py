offset = True
split_idx = 0

# dataset settings
dataset_type = "SumDataset"
data_root = "data/summarization"
data_root_val = "data/summarization"
ann_file = "data/summarization/summe.yml"

work_dir = f"work_dirs/tem_snippet_sum_dataset/split_{split_idx}"

tem_work_dir = work_dir
tem_results_dir = f"{tem_work_dir}/tem_results/"

# pgm_work_dir = f"work_dirs/tem_snippet_sum_dataset/split_{split_idx}"
pgm_work_dir = f"{work_dir}/pgm_results"
if offset:
    pgm_work_dir += "_offset"

pgm_proposals_dir = f"{pgm_work_dir}/pgm_proposals/"
pgm_features_dir = f"{pgm_work_dir}/pgm_features/"

pgm_proposals_cfg = dict(pgm_proposals_thread=2)
pgm_features_cfg = dict(
    pgm_features_thread=4,
    bsp_boundary_ratio=0.2,
    num_sample_start=8,
    num_sample_end=8,
    num_sample_action=16,
    num_sample_interp=3,
)
