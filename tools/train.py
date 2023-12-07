#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import random
import warnings
from loguru import logger
import json

import torch
import torch.backends.cudnn as cudnn

from yolox.core import Trainer, launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, configure_omp, get_num_devices

_SUPPORTED_DATASETS = ["coco", "lm", "lmo", "ycbv", "tless", "coco_kpts", 'oms']
_NUM_CLASSES = {"coco":80, "lm":15, "lmo": 8, "ycbv": 21, "tless": 30, "coco_kpts":1, 'oms':5}
_VAL_ANN = {
    "coco":"instances_val2017.json", 
    "lm":"instances_test.json",
    "lmo":"instances_test_bop.json",
    "ycbv": "instances_test_bop.json",
    "tless": "instances_test_bop.json",
    "coco_kpts": "person_keypoints_val2017.json",
    "oms": "instances_test_oms.json"
}
_TRAIN_ANN = {
    "coco":"instances_train2017.json", 
    "lm":"instances_train.json",
    "lmo":"instances_train_pbr.json",   #This can be the default setting for the LMO datase
    "ycbv": "instances_train.json",
    "tless": "instances_train.json", #"instances_train.json"
    "coco_kpts": "person_keypoints_train2017.json",
    "oms": "instances_train_oms.json"
}
_SUPPORTED_TASKS = {
    "coco":["2dod"],
    "lm":["2dod", "object_pose"],
    "lmo": ["2dod", "object_pose"],
    "ycbv": ["2dod", "object_pose"],
    "tless": ["2dod", "object_pose"],
    "coco_kpts": ["2dod", "human_pose"],
    "oms":["2dod"]
}

def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--device_type", default='cuda', type=str, help="device type for training. cpu for cpu based training, otherwise gpu based training"
    )
    parser.add_argument(
        "-w", "--workers", default=None, type=int, help="number of workers per gpu"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="plz input your experiment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("--dataset", default=None, type=str, help="dataset for training")
    parser.add_argument("--task", default="2dod", type=str, help="type of task for model eval")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-odw", 
        "--od-weights", 
        default=None, 
        type=str, 
        help="weights for trained 2DOD network"
    )
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--cache",
        dest="cache",
        default=False,
        action="store_true",
        help="Caching imgs to RAM for fast training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "--visualize",
        dest="visualize",
        default=False,
        action="store_true",
        help="Enable drawing of bounding cuboid"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--train_ann",
        help="train annotation file name",
        default=None,
    )
    parser.add_argument(
        "--val_ann",
        help="val annotation file name",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--max_epoch",
        default=300,
        type=int,
        help="Max epochs for training",
    )
    return parser


@logger.catch
def main(exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True
    
    exp.max_epoch = max(exp.warmup_epochs, args.max_epoch)

    if args.dataset is not None:
        assert (
            args.dataset in _SUPPORTED_DATASETS
        ), "The given dataset is not supported for training!"
        exp.data_set = args.dataset
        exp.num_classes = _NUM_CLASSES[args.dataset]
        exp.val_ann = args.val_ann or _VAL_ANN[args.dataset]
        exp.train_ann = args.train_ann or _TRAIN_ANN[args.dataset]

        if args.task is not None:
            assert (
                args.task in _SUPPORTED_TASKS[args.dataset]
            ), "The specified task cannot be performed with the given dataset!"
            if args.dataset == "ycbv" or args.dataset == "lmo" or args.dataset == "lm":
                #exp.pose = True if args.task == "object_pose" else exp.pose = False
                if args.task == "object_pose":
                    exp.object_pose = True
            elif args.dataset=="coco_kpts":
                if args.task == "human_pose":
                    exp.human_pose=True

        if args.visualize:
            exp.visualize = args.visualize

    trainer = Trainer(exp, args)
    trainer.train()


def run(**kwargs):
    '''
    This run function is not called from inside this repository
    This is for use from external programs such as the edgeai-modemaker
    '''
    args = make_parser().parse_args()
    for k, v in kwargs.items():
        setattr(args, k, v)
    exp = get_exp(args.exp_file, args.name)
    exp.max_epoch = args.max_epoch
    exp.warmup_epochs = max(min(exp.warmup_epochs, exp.max_epoch//4), 1)
    exp.no_aug_epochs = max(min(exp.no_aug_epochs, exp.max_epoch//4), 1)
    exp.output_dir = args.output_dir
    exp.visualize = args.visualize
    exp.data_dir = args.data_dir
    exp.train_ann = args.train_ann
    exp.val_ann = args.val_ann
    exp.img_folder_names = args.img_folder_names
    exp.flip_prob = 0.0
    exp.default_sigmas = False
    exp.device_type = args.device_type

    if args.ckpt is not None:
        exp.od_weights = args.ckpt

    with open(args.train_ann) as train_ann_fp:
        train_anno = json.load(train_ann_fp)
        categories = train_anno['categories']
        exp.num_kpts = len(categories[0]['keypoints'])

    if not args.experiment_name:
        args.experiment_name = ''

    if args.workers is not None:
        exp.data_num_workers = args.workers

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    dist_url = "auto" if args.dist_url is None else args.dist_url

    if num_gpu > 1:
        launch(
            main,
            num_gpu,
            args.num_machines,
            args.machine_rank,
            backend=args.dist_backend,
            dist_url=dist_url,
            args=(exp, args),
        )
    else:
        main(exp, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    args.name = 'yolox-s-ti-lite'
    args.d = 1
    args.b = 64
    args.fp16=True
    args.dataset = 'oms'
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    # additional merges
    for key in ['device_type', 'train_ann', 'val_ann']:
        val = getattr(args, key)
        if val is not None:
            setattr(exp, key, val)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    if args.workers is not None:
        exp.data_num_workers = args.workers

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )
