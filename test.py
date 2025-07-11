# 用来统计一些数据
import os
import numpy as np


if __name__ == '__main__':
    file_path = "/home/xjc/workspace/fusiontrack/data/3d_detections/pointgnn/training/det_scores/0000/000015.npy"
    # file_path = "/home/xjc/workspace/fusiontrack/data/3d_detections/pointgnn/training/det_bboxes_3d/0000/000015.npy"

    data = np.load(file_path)

    print(data.shape)
    print(len(data))
    print(data)