import argparse
import os

import cv2
import numpy as np
import tensorflow.lite as tflite

from utils import COCO_CLASSES, multiclass_nms_class_aware, preprocess, postprocess, vis


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help='path to .tflite model')
    parser.add_argument('-i', '--img', required=True, help='path to image file')
    parser.add_argument('-o', '--out-dir', default='tmp/tflite', help='path to output directory')
    parser.add_argument('-s', '--score-thr', type=float, default=0.3, help='threshould to filter by scores')
    return parser.parse_args()


def main():
    # reference:
    # https://github.com/PINTO0309/PINTO_model_zoo/blob/main/132_YOLOX/demo/tflite/yolox_tflite_demo.py

    args = parse_args()

    # prepare model
    interpreter = tflite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # preprocess
    input_shape = input_details[0]['shape']
    b, h, w, c = input_shape
    img_size = (h, w)

    origin_img = cv2.imread(args.img)
    img, ratio = preprocess(origin_img, img_size)
    img = img[np.newaxis].astype(np.float32)  # add batch dim

    # run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    outputs = interpreter.get_tensor(output_details[0]['index'])
    outputs = outputs[0]  # remove batch dim

    # postprocess
    preds = postprocess(outputs, (h, w))
    boxes = preds[:, :4]
    scores = preds[:, 4:5] * preds[:, 5:]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    boxes_xyxy /= ratio
    dets = multiclass_nms_class_aware(boxes_xyxy, scores, nms_thr=0.45, score_thr=args.score_thr)

    # visualize and save
    if dets is None:
        print("no object detected.")
    else:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                         conf=args.score_thr, class_names=COCO_CLASSES)

    os.makedirs(args.out_dir, exist_ok=True)
    output_path = os.path.join(args.out_dir, args.img.split('/')[-1])
    cv2.imwrite(output_path, origin_img)


if __name__ == '__main__':
    main()
