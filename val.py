# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from rcnn.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os, sys

import torch
from rcnn.config import cfg
from rcnn.dta.build import make_data_loader
from rcnn.engine.inference import inference


from misc import *

def val():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="./e2e_faster_rcnn_R_50_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # model = build_detection_model(cfg)
    MODEL_PATH = os.path.join(sys.path[0], 'data', 'output', 'model')
    model = torch.load(os.path.join(MODEL_PATH, 'last.pkl'))
    model.to(cfg.MODEL.DEVICE)

    # output_dir = cfg.OUTPUT_DIR
    # checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    # _ = checkpointer.load(cfg.MODEL.WEIGHT)

    dataset_names = cfg.DATASETS.TEST
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=False)
    result = inference(
        model,
        data_loaders_val[0],
        device=cfg.MODEL.DEVICE,
    )
    sum_iou = 0
    for k in result:
        box_result, p_img, target = result[k]
        p_size = (256, 256)
        box_result = box_result.resize(p_size)
        target = target.resize(p_size)
        labels = box_result.get_field('labels')
        scores = box_result.get_field('scores')

        i = 1
        last_result = [torch.zeros(4) for i in range(6)]
        scores_temp = 0
        for l, s, box in list(zip(labels.tolist(), scores.tolist(), box_result.bbox)):
            if i > 6:
                break
            if l == i:
                i = l + 1
                last_result[l - 1] = box
                scores_temp = s
            elif l > i:
                last_result[l - 2] = last_result[i - 1]
                last_result[l - 1] = box
            elif l < i:
                if s > scores_temp:
                    last_result[l - 1] = box
                scores_temp = s
        last_result = torch.cat(last_result)
        target_box = target.bbox.reshape(1, -1)

        for i in range(6):
            iou = get_iou(target_box[:, 4 * i:4 * i + 4], last_result[None, 4 * i:4 * i + 4])
            sum_iou += float(torch.sum(iou))
    print('avg iou = {:.3f}'.format(sum_iou / 6 / len(result)))
    return sum_iou


if __name__ == "__main__":
    val()
