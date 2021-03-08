import json
import os
import os.path as osp

import h5py
# import numpy as np
import scipy.io as scio

# import tempfile

# summe video keys:
# change_points, features, gtscore, gtsummary, n_frame_per_seg,
# n_frames, n_steps, picks, user_summary, video_name

# tvsum video keys:
# change_points, features, gtscore, gtsummary, n_frame_per_seg,
# n_frames, n_steps, picks, user_summary

# youtube video keys:
# change_points, features, gtscore, gtsummary, n_frame_per_seg, n_frames, picks

# ovp video keys:
# change_points, features, gtscore, gtsummary, n_frame_per_seg, n_frames, picks

# summe mat file keys:
# FPS, video_duration, nFrames, user_score(shape:[nFrames, users]),
# gt_score(average by users), segments(segments by every user)

# def read_h5_file(file_path):
#     file = h5py.File(file_path, 'r')
#     import pdb
#     pdb.set_trace()

# def read_mat_file(file_path):
#     file = scio.loadmat(file_path)
#     import pdb
#     pdb.set_trace()


def translate_summe_mat_to_json(files_dir, json_file):
    mat_files = sorted(
        [osp.join(files_dir, path) for path in os.listdir(files_dir)])
    file_names = [osp.splitext(osp.basename(path))[0] for path in mat_files]
    dic = dict()
    for i, mat_file in enumerate(mat_files):
        file = scio.loadmat(mat_file)
        name = file_names[i]
        dic[name] = dict()
        dic[name]['duration_second'] = float(file['video_duration'][0][0])
        dic[name]['fps'] = float(file['FPS'][0][0])
        dic[name]['annotations'] = list()
        annos = file['segments'][0]
        for anno in annos:
            tmp_dict = dict(segments=anno.tolist())
            dic[name]['annotations'].append(tmp_dict)
    with open(json_file, 'w') as f:
        json.dump(dic, f, ensure_ascii=False, indent=2)


def translate_summe_tvsum_to_json(file_path, json_file):
    file = h5py.File(file_path, 'r')
    video_names = list(file.keys())
    dic = dict()
    for name in video_names:
        dic[name] = dict()
        annos = file[name]
        dic[name]['n_frames'] = int(annos['n_frames'][()])


def translate_youtube_ovp_to_json(file_path):
    pass


if __name__ == '__main__':
    # path = 'data/summarization/keyshot/eccv16_dataset_summe_google_pool5.h5'
    # read_h5_file(path)
    # path = 'data/SumMe/Air_Force_One.mat'
    # read_mat_file(path)
    # files_dir = 'data/SumMe'
    # json_file = 'data/summe.json'
    # translate_summe_mat_to_json(files_dir, json_file)
    pass
