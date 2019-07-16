import numpy as np

import cv2
import torch

def get_iou(out ,gt):
    assert out.shape[1] == 4 and gt.shape[1]==4, "dimension 1 must be 4"

    cat_box = torch.cat((out[:,:,None], gt[:,:,None]), dim=2)

    x_min, _ = torch.max(cat_box[:,0,:], dim=1)
    x_max, _ = torch.min(cat_box[:,2,:], dim=1)
    y_min, _ = torch.max(cat_box[:,1,:], dim=1)
    y_max, _ = torch.min(cat_box[:,3,:], dim=1)

    width = x_max - x_min
    heigth = y_max - y_min
    width[width < 0] = 0
    heigth[heigth < 0] = 0

    inter = width * heigth

    out_area = (out[:,2] - out[:,0]) * (out[:, 3] - out[:, 1])
    gt_area = (gt[:,2] - gt[:,0]) * (gt[:, 3] - gt[:, 1])

    union = out_area + gt_area - inter

    return inter / union

