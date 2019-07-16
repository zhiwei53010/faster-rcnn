import os
import numpy as np
import torch
# import mycsrc
# print(mycsrc)
import cv2
#
# from PIL import Image
# from modeling.misc import *
# from model import Model
# from flyai.dataset import Dataset
#
# data = Dataset()
# model = Model()
#
# model.predict_all(data.get_all_data()[0])

# print([ i- ((i+1)%2<<1) for i in range(1,25)])
# root = '/home/administer/Desktop/flyai/My_HandDetectionFlyAI_FlyAI/data/hand'
# path_list = os.listdir(root)
# img_list = [np.array(cv2.imread(os.path.join(root, path))) for path in path_list]
# temp = [0,0,0]
# for img in img_list:
#     area = img.shape[0] * img.shape[1]
#     b = np.sum(img[:,:,0])/ area
#     g = np.sum(img[:,:,1])/ area
#     r = np.sum(img[:,:,2])/ area
#     temp[0] += b
#     temp[1] += g
#     temp[2] += r
#
# temp = np.array(temp)
# print(temp/len(img_list))


