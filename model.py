# -*- coding: utf-8 -*
import os, argparse, sys
import numpy as np

from PIL import Image

import torch
from torchvision import transforms as T

from rcnn.fdataset import Dataset as FlyDataset
from rcnn.dta.transforms import build_transforms
from rcnn.structures.image_list import to_image_list
from rcnn.structures.bounding_box import BoxList
from rcnn.config import cfg
from flyai.model.base import Base

from path import MODEL_PATH, DATA_PATH



class Model(Base):
    def __init__(self, data=None):
        if data is None:
            data = FlyDataset()
        self.data = data

        this_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(this_dir, 'e2e_faster_rcnn_R_50_FPN_1x.yaml')
        cfg.merge_from_file(cfg_path)
        cfg.freeze()
        self.cfg = cfg
        self.transform = build_transforms(cfg, is_train=False)


    def save_model(self, net, path=MODEL_PATH, name='last.pkl', overwrite=False):
        self.check(path, overwrite)
        torch.save(net, os.path.join(path, name))


    def predict_all(self, datas):
        '''

        :param datas: list[{'image_path':'hand/0000001.jpg'},{'image_path':'hand/0000002.jpg'} ... ]
        :return: outputs: list[np(1,24), np(1,24)...]
                img_size: list[224, 224]
        '''
        img_size = [512, 512]
        device = torch.device('cuda')
        net = torch.load(os.path.join(MODEL_PATH, 'last.pkl'))

        net = net.to(device)
        net = net.eval()

        outputs = []
        for data in datas:
            image_path = data['image_path']
            image = Image.open(os.path.join(DATA_PATH, image_path)).convert('RGB')

            target_fault = BoxList(torch.zeros(1,4), img_size)

            input, target_fault = self.transform(image, target_fault)
            input = input.to(device)
            input = to_image_list(input, 32)
            box_result = net(input)
            box_result = box_result[0].resize(img_size)
            labels = box_result.get_field('labels')
            scores = box_result.get_field('scores')

            i = 1
            last_result = [torch.tensor((0,0,512,512), device=device, dtype=torch.float) for i in range(6)]
            scores_temp = 0
            class_flag = [0, 0, 0, 0, 0, 0]

            for l, s, box in list(zip(labels.tolist(), scores.tolist(), box_result.bbox)):
                if i > 6:
                    break
                if l == i:
                    i = l + 1
                    last_result[l - 1] = box
                    class_flag[l - 1] = 1
                    scores_temp = s
                elif l > i:
                    last_result[l - 1] = box
                    class_flag[l - 1] = 1
                    i = l + 1
                elif l < i:
                    if s > scores_temp:
                        last_result[l - 1] = box
                    scores_temp = s

            if sum(class_flag) < 6:
                top_left = last_result[0][0:2]
                bottom_right = last_result[1][2:]
                for idx in range(6):
                    if class_flag[idx] < 1:
                        sys.stderr.writr('{}\t'.format(idx))
                        # if idx == 4:
                        #     r_4 = (last_result[3] + last_result[5]) / 2
                        #     last_result[4] = r_4
                        # if idx == 5:
                        #     y1 = (top_left[1] + bottom_right[1]) / 2 - 20
                        #     y2 = y1 + 40
                        #     x1, x2 = top_left[0], top_left[0] + 75
                        #     last_result[5] = torch.tensor((x1, x2, y1, y2), device=device)
                sys.stderr.writr('\n')


            last_result = torch.cat(last_result).detach().cpu().numpy()
            outputs.append(last_result[[2*i for i in range(12)] + [2*i+1 for i in range(12)]].reshape(1,24))

            # torch.cuda.empty_cache()

        return [outputs, img_size]


if __name__ == '__main__':
    Model.predict_all()
