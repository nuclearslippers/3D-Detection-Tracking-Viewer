# 用来统计一些数据
import os



if __name__ == '__main__':
    txt_path = "/home/xjc/workspace/fusiontrack/evaluation/data/tracking/gt/kitti/kitti_2d_box_train/label_02/0020.txt"
    count = 0


    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if line[2] == "Car":
                count += 1

    print(count)