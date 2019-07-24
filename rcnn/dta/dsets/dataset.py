# -*- coding: utf-8 -*
import os,sys

import numpy as np
from PIL import Image

import torch
import torchvision.transforms
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data._utils.collate import default_collate
from rcnn.structures.bounding_box import BoxList
from rcnn.fdataset import Dataset as FlyDataset

# from .collate_batch import BatchCollator
# from .bounding_box import BoxList
# from .transforms import build_transforms
from rcnn.dta.collate_batch import BatchCollator


DATA_PATH = os.path.join(sys.path[0], 'data', 'input')

class MyDataset(Dataset):

    def __init__(self, root='.', train=True, transforms=None, **kwargs):
        self.root = os.path.join(root, 'data')

        # 线上代码用
        self.img_paths, self.labels = self.read_data2()
        # 本地调试用
        # self.img_paths, self.labels = self.read_data()

        self.train = train

        assert len(self.img_paths) == len(self.labels)

        length = int(len(self.img_paths) * 0.9)
        if train:
            # self.img_paths = self.img_paths[:length]
            # self.labels = self.labels[:length]
            pass
        else:
            self.img_paths = self.img_paths[length:]
            self.labels = self.labels[length:]
            pass

        self.length = len(self.img_paths)
        self.transform = transforms



    def __getitem__(self, item):
        # img = self.dataset[0][item]
        # label = self.dataset[1][item]
        # img = np.transpose(img, (2,0,1))
        # return img.astype(np.float32), label.astype(np.float32)

        path = self.img_paths[item]
        info = self.labels[item]

        img_height, img_weight = info[0:2].astype(np.int32).tolist()
        label = info[2:].astype(np.float32)

        image = Image.open(path).convert('RGB')


        # x, y 坐标位置调换
        # [ i-((i+1)%2<<1) for i in range(1,25)] = [1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14,17,16,19,18,21,20,23,22]
        label = label[[i - ((i + 1) % 2 << 1) for i in range(1, 25)]]

        target = label #
        target = target.reshape(-1, 4)
        # BoxList 数据结构
        target = BoxList(target, image.size, mode="xyxy")
        # 每个box的类别
        classes = torch.tensor([1, 2, 3, 4, 5, 6])
        target.add_field('labels', classes)
        target = target.clip_to_image(remove_empty=True)

        if self.transform:
            img, target = self.transform(image, target)
        else:
            raise RuntimeError('transform not implement')

        return img, target, item, image

    def __len__(self):
        return self.length

    def get_img_info(self, index):
        label = self.labels[index]
        img_data = {'height': label[0], 'width': label[1]}
        return img_data

    # 本地调试专用
    def read_data(self):
        info = torch.load(os.path.join(self.root, 'label.pkl'))
        img_paths = []
        temp = []
        for i in info:
            if np.sum(i[3:7].astype(np.int32)) == 0:
                continue
            if np.sum(i[15:19].astype(np.int32)) == 0:
                continue
            temp.append(i)

        labels = []
        for i in temp:
            img_paths.append(os.path.join(self.root, i[0]))
            labels.append(i[1:])
        return img_paths, labels



    # 线上代码专用
    def read_data2(self):
        data = FlyDataset()
        dataset = data.get_all_data()
        image_paths = []
        labels = []

        def fun(image_path, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18,
                    p19, p20, p21, p22, p23, p24, p25, p26):
            return image_path, (p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18,
                    p19, p20, p21, p22, p23, p24, p25, p26)

        for i in dataset[1]:
            image_path, label = fun(**i)
            image_paths.append(os.path.join(DATA_PATH, image_path))
            labels.append(label)

        return image_paths, np.array(labels)


    @staticmethod
    def batch_collator(batch):
        transposed_batch = list(zip(*batch))
        transposed_batch[0] = default_collate(transposed_batch[0])
        transposed_batch[1] = default_collate(transposed_batch[1])

        if len(transposed_batch) > 4:
            transposed_batch[2] = default_collate(transposed_batch[2])
            transposed_batch[3] = default_collate(transposed_batch[3])

        return transposed_batch

def get_dataloader(batch, is_train=True, transform=None):
    is_shuffle = is_train == True
    trainset = MyDataset(train=is_train, transforms=transform)
    trainloader = DataLoader(trainset, batch_size=batch, shuffle=is_shuffle, collate_fn=BatchCollator(size_divisible=32))

    return trainloader


