# -*- coding: utf-8 -*

import argparse
import os, sys
sys.path.append('./rcnn')
import torch
from rcnn.config import cfg
from rcnn.dta.build import make_data_loader
from rcnn.solver.build import make_lr_scheduler
from rcnn.solver.build import make_optimizer
from rcnn.engine.trainer import do_train
from rcnn.utils.checkpoint import DetectronCheckpointer
from rcnn.utils.comm import get_rank
from rcnn.utils.logger import setup_logger
from rcnn.modeling.detector import build_detection_model

def train(cfg, local_rank, distributed):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)


    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    model = do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        # checkpointer,
        None,
        device,
        checkpoint_period,
        arguments,
    )

    return model

def main_train(*argss):
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="e2e_faster_rcnn_R_50_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = setup_logger("rcnn", None, get_rank())

    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    net = train(cfg, args.local_rank, args.distributed)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--EPOCHS", default=100, type=int, help="train epochs")
    parser.add_argument("-b", "--BATCH", default=2, type=int, help="batch size")
    args = parser.parse_args()

    main_train([args.BATCH, args.EPOCHS])



