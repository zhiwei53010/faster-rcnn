# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
# from tqdm import tqdm

def compute_on_dataset(model, data_loader, device):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(data_loader):
        images, _, image_ids, p_imgs = batch
        images = images.to(device)
        with torch.no_grad():
            output = model(images)
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: [result, p_img] for img_id, result, p_img in zip(image_ids, output, p_imgs)}
        )
    return results_dict


def inference(
        model,
        data_loader,
        device="cuda",
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    # predictions = compute_on_dataset(model, data_loader, device)
    
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(data_loader):
        images, targets, image_ids, p_imgs = batch
        images = images.to(device)
        with torch.no_grad():
            output = model(images)
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: [result, p_img, target] for img_id, result, p_img, target in zip(image_ids, output, p_imgs, targets)}
        )

    return results_dict
