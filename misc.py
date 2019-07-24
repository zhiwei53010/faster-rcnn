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


def show_box(img, box):
    image = np.array(img)

    for i in range(6):
        top_left, bottom_right = box[[4*i, 4*i+1]], box[[4*i+2, 4*i+3]]
        image = cv2.rectangle(img=image, pt1=tuple(top_left), pt2=tuple(bottom_right), color=255, thickness=2)
    # top_left, bottom_right = box[[1,0]].tolist(), box[[3,2]].tolist()
    # image = cv2.rectangle(
    #     image, tuple(top_left), tuple(bottom_right), 1, 1
    # )

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('name', image)
    cv2.waitKey(0)
    print()
    # plt.imshow(image)
    # plt.show()
