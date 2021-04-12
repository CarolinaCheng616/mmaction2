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
    mat_files = sorted([osp.join(files_dir, path) for path in os.listdir(files_dir)])
    file_names = [osp.splitext(osp.basename(path))[0] for path in mat_files]
    dic = dict()
    for i, mat_file in enumerate(mat_files):
        file = scio.loadmat(mat_file)
        name = file_names[i]
        dic[name] = dict()
        dic[name]["duration_second"] = float(file["video_duration"][0][0])
        dic[name]["fps"] = float(file["FPS"][0][0])
        dic[name]["annotations"] = list()
        annos = file["segments"][0]
        for anno in annos:
            tmp_dict = dict(segments=anno.tolist())
            dic[name]["annotations"].append(tmp_dict)
    with open(json_file, "w") as f:
        json.dump(dic, f, ensure_ascii=False, indent=2)


def translate_summe_tvsum_to_json(file_path, json_file):
    file = h5py.File(file_path, "r")
    video_names = list(file.keys())
    dic = dict()
    for name in video_names:
        dic[name] = dict()
        annos = file[name]
        dic[name]["n_frames"] = int(annos["n_frames"][()])


def translate_youtube_ovp_to_json(file_path):
    pass


def get_paths(ROOT, depth=4):
    path_list = set()
    path_list.add(ROOT)
    for i in range(depth):
        tmp_list = set()
        for path in path_list:
            for subdir in os.listdir(path):
                tmp_list.add(osp.join(path, subdir))
        path_list = tmp_list
    return path_list


def read_tree_dir_files_to_file(path, wfile, depth=4):
    """
    read root dir path in depth and write files path into wfile

    :param path:    root directory
    :param wfile:   write the files in the root directory's depth's into wfile
    :param depth:
    :return:
    """
    path_list = sorted(get_paths(path, depth))
    with open(wfile, "w", encoding="utf-8") as f:
        f.write("\n".join(path_list))


if __name__ == "__main__":
    root2 = "data/bilibili/bilibili_dm"
    wfile2 = "data/bilibili/dm_files.txt"
    read_tree_dir_files_to_file(root2, wfile2)
