import numpy as np
import re
from .kitti_data_base import *
import os

class KittiDetectionDataset:
    def __init__(self,root_path,label_path = None):
        self.root_path = root_path
        self.velo_path = os.path.join(self.root_path,"velodyne")
        self.image_path = os.path.join(self.root_path,"image_2")
        self.calib_path = os.path.join(self.root_path,"calib")
        if label_path is None:
            self.label_path = os.path.join(self.root_path, "label_2")
        else:
            self.label_path = label_path

        self.all_ids = os.listdir(self.velo_path)

    def __len__(self):
        return len(self.all_ids)
    def __getitem__(self, item):

        name = str(item).zfill(6)

        velo_path = os.path.join(self.velo_path,name+'.bin')
        image_path = os.path.join(self.image_path, name+'.png')
        calib_path = os.path.join(self.calib_path, name+'.txt')
        label_path = os.path.join(self.label_path, name+".txt")

        P2,V2C = read_calib(calib_path)
        points = read_velodyne(velo_path,P2,V2C)
        image = read_image(image_path)
        labels,label_names = read_detection_label(label_path)
        labels[:,3:6] = cam_to_velo(labels[:,3:6],V2C)[:,:3]

        return P2,V2C,points,image,labels,label_names



# 补充自己实现的库函数
from .kitti_lab import read_detection_2d_label

class KittiTrackingDataset:
    def __init__(self,root_path,seq_id,det2d_path=None,label_path=None):
        self.seq_name = str(seq_id).zfill(4)
        self.root_path = root_path
        self.velo_path = os.path.join(self.root_path,"velodyne",self.seq_name)
        self.image_path = os.path.join(self.root_path,"image_02",self.seq_name)
        self.calib_path = os.path.join(self.root_path,"calib",self.seq_name)

        self.all_ids = os.listdir(self.velo_path)
        calib_path = self.calib_path + '.txt'

        if label_path is None:

            label_path = os.path.join(self.root_path, "label_02", self.seq_name+'.txt')


        self.P2, self.V2C = read_calib(calib_path)
        self.labels, self.label_names = read_tracking_label(label_path)

        # detection2d
        if det2d_path is not None:    
            self.lab = True
            self.det2d_path = os.path.join(det2d_path, self.seq_name)

    def __len__(self):
        return len(self.all_ids)-1
    def __getitem__(self, item):

        name = str(item).zfill(6)

        velo_path = os.path.join(self.velo_path,name+'.bin')
        image_path = os.path.join(self.image_path, name+'.png')

        points = read_velodyne(velo_path,self.P2,self.V2C)
        image = read_image(image_path)

        if item in self.labels.keys():
            labels = self.labels[item]
            labels = np.array(labels)
            labels[:,3:6] = cam_to_velo(labels[:,3:6],self.V2C)[:,:3]
            label_names = self.label_names[item]
            label_names = np.array(label_names)
        else:
            labels = None
            label_names = None

        if self.det2d_path is not None:
            det_2d_path = os.path.join(self.det2d_path,name+'.txt')
            det_2d = read_detection_2d_label(det_2d_path)


        if self.lab:
            return self.P2,self.V2C,points,image,labels,label_names,det_2d
        else:
            return self.P2,self.V2C,points,image,labels,label_names
        

from .kitti_lab import read_detection_2d_label,read_gt_2d,read_tracking_2d_label,read_detection_3d_label

class KittiLabDataset:
    def __init__(self,root_path,seq_id,det2d_path=None,det3d_path=None,label_path=None,gt=None):
        self.seq_name = str(seq_id).zfill(4)
        self.root_path = root_path
        self.velo_path = os.path.join(self.root_path,"velodyne",self.seq_name)
        self.image_path = os.path.join(self.root_path,"image_02",self.seq_name)
        self.calib_path = os.path.join(self.root_path,"calib",self.seq_name)

        self.all_ids = os.listdir(self.velo_path)
        calib_path = self.calib_path + '.txt'

        if label_path is None:

            label_path = os.path.join(self.root_path, "label_02", self.seq_name+'.txt')


        self.P2, self.V2C = read_calib(calib_path)
        self.labels, self.label_names = read_tracking_label(label_path)
        self.labels_2d = read_tracking_2d_label(label_path,mask='Car')

        # detection2d
        if det2d_path is not None:    
            self.det2d_path = os.path.join(det2d_path, self.seq_name)

        # detection3d
        if det3d_path is not None:    
            self.det3d_box_path = os.path.join(det3d_path, "det_bboxes_3d", self.seq_name)
            self.det3d_score_path = os.path.join(det3d_path, "det_scores", self.seq_name)

        # gt
        if gt is not None:
            self.gt_path = os.path.join(gt, self.seq_name+'.txt')
        else:
            self.gt_path = None
            

    def __len__(self):
        return len(self.all_ids)-1
    def __getitem__(self, item):

        name = str(item).zfill(6)

        velo_path = os.path.join(self.velo_path,name+'.bin')
        image_path = os.path.join(self.image_path, name+'.png')

        points = read_velodyne(velo_path,self.P2,self.V2C)
        image = read_image(image_path)

        if item in self.labels.keys():
            labels = self.labels[item]
            labels = np.array(labels)
            labels[:,3:6] = cam_to_velo(labels[:,3:6],self.V2C)[:,:3]
            label_names = self.label_names[item]
            label_names = np.array(label_names)
        else:
            labels = None
            label_names = None

        # 单检测2d
        if self.det2d_path is not None:
            det_2d_path = os.path.join(self.det2d_path,name+'.txt')
            det_2d = read_detection_2d_label(det_2d_path)
        else:
            print("no det2d")
            det_2d = None

        # 检测3d
        if self.det3d_box_path is not None:
            det3d_box_path = os.path.join(self.det3d_box_path,name+'.npy')
            det3d_score_path = os.path.join(self.det3d_score_path,name+'.npy')
            det3d = read_detection_3d_label(det3d_box_path,det3d_score_path)
        else:
            print("no det3d")
            det3d = None
            


        # 追踪/gt/检测都要有
        if self.gt_path is not None:
            gt = read_gt_2d(self.gt_path,int(name))
        else:
            print("no gt")
            gt = None

        # 追踪结果的2D
        if item in self.labels_2d.keys():
            labels_2d = self.labels_2d[item]
        else:
            labels_2d = None


        return self.P2,self.V2C,points,image,labels,label_names,det_2d,gt,labels_2d,det3d