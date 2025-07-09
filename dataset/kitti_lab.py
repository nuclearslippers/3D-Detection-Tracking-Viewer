import os


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