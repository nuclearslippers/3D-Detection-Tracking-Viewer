from viewer.viewer import Viewer
import numpy as np
from dataset.kitti_dataset import KittiLabDataset

def kitti_viewer():
    lab_mode = True
    root="/home/xjc/dataset/kitti_tracking/training"
    label_path = "/home/xjc/workspace/fusiontrack/evaluation/results/sha_key/data/0001.txt"
    det2d_path = "/home/xjc/workspace/fusiontrack/data/2d_detections/rrc/training"
    gt_path = "/home/xjc/dataset/kitti_tracking/training/label_02"
    sequence_id = 1
    dataset = KittiLabDataset(root,seq_id=sequence_id,
                                   det2d_path=det2d_path,
                                   label_path=label_path,
                                   gt=gt_path)
    vi = Viewer(box_type="Kitti")



    for i in range(len(dataset)):
        P2, V2C, points, image, labels, label_names, det_2d, gt_2d, labels_2d = dataset[i]
        frame = i


        if labels is not None:
            mask = (label_names=="Car")
            labels = labels[mask]
            label_names = label_names[mask]
            vi.add_3D_boxes(labels, ids=labels[:, -1].astype(int), box_info=label_names,caption_size=(0.09,0.09))
            vi.add_3D_cars(labels, ids=labels[:, -1].astype(int), mesh_alpha=1)
        vi.add_points(points[:,:3])

        vi.add_image(image)
        vi.set_extrinsic_mat(V2C)
        vi.set_intrinsic_mat(P2)

        vi.show_2D()

        vi.show_det_2d(det_2d)
        vi.show_lab(gt_2d,labels_2d,frame)

        vi.show_3D()

        


if __name__ == '__main__':
    kitti_viewer()
