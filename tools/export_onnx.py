#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from loguru import logger

import torch
from torch import nn

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module, PostprocessExport
from yolox.data.data_augment import preproc as preprocess
from yolox.utils.proto import mmdet_meta_arch_pb2
from google.protobuf import text_format
from yolox.utils.proto.pytorch2proto import prepare_model_for_layer_outputs, retrieve_onnx_names

import cv2

_SUPPORTED_DATASETS = ["coco", "linemod", "coco_kpts"]
_NUM_CLASSES = {"coco":80, "linemod":15, "coco_kpts":57}
_VAL_ANN = {
    "coco":"instances_val2017.json", 
    "linemod":"instances_test.json"
}
_TRAIN_ANN = {
    "coco":"instances_train2017.json", 
    "linemod":"instances_train.json"
}
_SUPPORTED_TASKS = {
    "coco":["2dod"],
    "linemod":["2dod", "6dpose"]
}

def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
    parser.add_argument(
        "--output-name", type=str, default="yolox.onnx", help="output name of models"
    )
    parser.add_argument(
        "--input", default="images", type=str, help="input node name of onnx model"
    )
    parser.add_argument(
        "--output", default="output", type=str, help="output node name of onnx model"
    )
    parser.add_argument(
        "-o", "--opset", default=11, type=int, help="onnx opset version"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--dynamic", action="store_true", help="whether the input shape should be dynamic or not"
    )
    parser.add_argument("--no-onnxsim", action="store_true", help="use onnxsim or not")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="expriment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("--dataset", default=None, type=str, help="dataset for training")
    parser.add_argument("--task", default=None, type=str, help="type of task for model eval")
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument("--export-det",  action='store_true', help='export the nms part in ONNX model')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser

def export_prototxt(model, img, onnx_model_name):
    # Prototxt export for a given ONNX model

    anchor_grid = model.head.strides
    num_heads = len(model.head.strides)
    num_keypoint = model.head.num_kpts if hasattr(model.head, "num_kpts") else None
    keypoint_confidence = True if (num_keypoint is not None and num_keypoint>0) else None
    keep_top_k = 20 if (num_keypoint is not None and num_keypoint>0) else 200
    matched_names = retrieve_onnx_names(img, model, onnx_model_name)
    prototxt_name = onnx_model_name.replace('onnx', 'prototxt')

    background_label_id = -1
    num_classes = model.head.num_classes
    assert len(matched_names) == num_heads; "There must be a matched name for each head"
    proto_names = [f'{matched_names[i]}' for i in range(num_heads)]
    yolo_params = []
    for head_id in range(num_heads):
        yolo_param = mmdet_meta_arch_pb2.TIDLYoloParams(input=proto_names[head_id],
                                                        anchor_width=[anchor_grid[head_id]],
                                                        anchor_height=[anchor_grid[head_id]])
        yolo_params.append(yolo_param)

    nms_param = mmdet_meta_arch_pb2.TIDLNmsParam(nms_threshold=0.65, top_k=500)
    detection_output_param = mmdet_meta_arch_pb2.TIDLOdPostProc(num_classes=num_classes, share_location=True,
                                            background_label_id=background_label_id, nms_param=nms_param,
                                            code_type=mmdet_meta_arch_pb2.CODE_TYPE_YOLO_X, keep_top_k=keep_top_k,
                                            confidence_threshold=0.01, num_keypoint=num_keypoint, keypoint_confidence=keypoint_confidence)

    yolov3 = mmdet_meta_arch_pb2.TidlYoloOd(name='yolo_v3', output=["detections"],
                                            in_width=img.shape[3], in_height=img.shape[2],
                                            yolo_param=yolo_params,
                                            detection_output_param=detection_output_param,
                                            )
    arch = mmdet_meta_arch_pb2.TIDLMetaArch(name='yolo_v3', tidl_yolo=[yolov3])

    with open(prototxt_name, 'wt') as pfile:
        txt_message = text_format.MessageToString(arch)
        pfile.write(txt_message)



@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    if args.dataset is not None:
        assert (
            args.dataset in _SUPPORTED_DATASETS
        ), "The given dataset is not supported for training!"
        exp.data_set = args.dataset
        exp.num_classes = _NUM_CLASSES[args.dataset]
        exp.val_ann = _VAL_ANN[args.dataset]
        exp.train_ann = _TRAIN_ANN[args.dataset]

        if args.task is not None:
            assert (
                args.task in _SUPPORTED_TASKS[args.dataset]
            ), "The specified task cannot be performed with the given dataset!"
            if args.dataset == "linemod":
                if args.task == "6dpose":
                    exp.object_pose = True
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    
    
    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    elif args.ckpt == "random":
        pass
    else:
        ckpt_file = args.ckpt

    if args.ckpt == "random":
        #Proceed with initialized values
        ckpt = None
    else:
        # load the model state dict
        ckpt = torch.load(ckpt_file, map_location="cpu")

    model.eval()
    if ckpt is not None:
        if "model" in ckpt:
            ckpt = ckpt["model"]
        model.load_state_dict(ckpt)
    model = replace_module(model, nn.SiLU, SiLU)
    if not args.export_det:
        model.head.decode_in_inference = False
    if args.export_det:
        post_process = PostprocessExport(conf_thre=0.25, nms_thre=0.45, num_classes=80)
        model_det = nn.Sequential(model, post_process)
        model_det.eval()
        args.output = 'detections'

    logger.info("loading checkpoint done.")
    img = cv2.imread("./assets/dog.jpg")
    img, ratio = preprocess(img, exp.test_size)
    img = img[None, ...]
    img = img.astype('float32')
    img = torch.from_numpy(img)
    dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1])
    if args.export_det:
        output = model_det(img)

    if args.export_det:
        torch.onnx._export(
            model_det,
            img,
            args.output_name,
            input_names=[args.input],
            output_names=[args.output],
            dynamic_axes={args.input: {0: 'batch'},
                          args.output: {0: 'batch'}} if args.dynamic else None,
            opset_version=args.opset,
        )
        logger.info("generated onnx model named {}".format(args.output_name))
    else:
        torch.onnx._export(
            model,
            img,
            args.output_name,
            input_names=[args.input],
            output_names=[args.output],
            dynamic_axes={args.input: {0: 'batch'},
                          args.output: {0: 'batch'}} if args.dynamic else None,
            opset_version=args.opset,
        )
        logger.info("generated onnx model named {}".format(args.output_name))

    if not args.no_onnxsim:
        import onnx

        from onnxsim import simplify

        input_shapes = {args.input: list(dummy_input.shape)} if args.dynamic else None

        # use onnxsimplify to reduce reduent model.
        onnx_model = onnx.load(args.output_name)
        model_simp, check = simplify(onnx_model,
                                     dynamic_input_shape=args.dynamic,
                                     input_shapes=input_shapes)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name)
        logger.info("generated simplified onnx model named {}".format(args.output_name))

    # export_prototxt(model, img, args.output_name)
    # logger.info("generated prototxt {}".format(args.output_name.replace('onnx', 'prototxt')))

if __name__ == "__main__":
    main()
