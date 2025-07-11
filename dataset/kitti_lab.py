import os
import numpy as np


def read_detection_2d_label(path):
    """
    读取检测结果，返回一个二维列表，外层列表表示每一帧的检测结果，内层列表表示目标的检测结果
        eg: [
            [x1,y1,x2,y2,score,class_id],
            [x1,y1,x2,y2,score,class_id]
        ]
    """
    if not os.path.exists(path):
        assert False, "{} does not exist".format(path)

    with open(path, "r") as f:
        lines = f.readlines()

    # 解析每一行数据，假设每行格式为 "x1 y1 x2 y2 score class_id"
    results = []
    for line in lines:
        values = line.strip().split()  # 按空格分割
        if len(values) == 6:  # 确保每行有6个值
            results.append([float(x) for x in values[:5]] + [int(float(values[5]))])  # 前5个为float，最后一个为int
        else:
            print(f"Warning: Invalid format in line '{line}'")

    return results

def read_gt_2d(path,frame):
    """
    读取真实标签，返回一个二维列表，外层列表表示gt的指定帧内所有追踪结果，内层列表表示的单个目标的结果
        eg: [
            [track_id,x1,y1,x2,y2],
            [track_id,x1,y1,x2,y2]
        ]
    """
    if not os.path.exists(path):
        assert False, "{} does not exist".format(path)

    with open(path, "r") as f:
        lines = f.readlines()

    mask = "Car"
    results = []
    for line in lines:
        values = line.strip().split()  # 按空格分割
        if values[2] != mask:
            continue
        if int(float(values[0])) == frame:
            results.append([int(float(values[1]))] + [float(x) for x in values[6:10]])  # track_id为int，其余为float

    return results

def read_tracking_2d_label(path,mask):
    """
    读取追踪结果中的2d标签，返回一个字典，键为帧数，值为一个二维列表，外层列表表示每一帧的检测结果，内层列表表示目标的检测结果
        eg: 
        {
            frame_id: [
                [track_id,x1,y1,x2,y2],
                [track_id,x1,y1,x2,y2]
            ]
        }
    """
    if not os.path.exists(path):
        assert False, "{} does not exist".format(path)

    with open(path, "r") as f:
        lines = f.readlines()

    results = {}
    for line in lines:
        values = line.strip().split()  # 按空格分割
        if values[2] != mask:
            continue
        else:
            frame_id = int(float(values[0]))  # 帧ID
            x1, y1, x2, y2 = float(values[6]), float(values[7]), float(values[8]), float(values[9])  # 边界框坐标
            track_id = int(float(values[1]))  # 跟踪ID

            if frame_id not in results:
                results[frame_id] = []  # 初始化该帧的结果列表
            results[frame_id].append([track_id, x1, y1, x2, y2])  # 添加检测结果

    return results



def read_detection_3d_label(box_path, score_path):
    """
    输入是两个npy文件的路径，一个是检测结果，一个是得分
    读取3D检测结果，返回一个二维列表，外层列表表示每一帧的检测结果，内层列表表示目标的检测结果
        eg: [
            [x,y,z,l,w,h,yaw,score],
            [x,y,z,l,w,h,yaw,score]
        ]
    """
    if not os.path.exists(box_path) or not os.path.exists(score_path):
        assert False, f"One of the files does not exist: {box_path}, {score_path}"

    # 加载 .npy 文件
    boxes = np.load(box_path)
    scores = np.load(score_path)

    # 确保两个文件的数据长度一致
    assert len(boxes) == len(scores), "The number of boxes and scores must match."

    # 合并检测结果和得分
    results = []
    for box, score in zip(boxes, scores):
        x, y, z, w, l, h, ry = box  # 解包边界框参数
        results.append([x, y, z, w, l, h, ry, float(score)])  # 添加到结果列表

    return results
