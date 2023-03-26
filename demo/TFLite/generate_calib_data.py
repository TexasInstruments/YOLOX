import argparse
import glob
import os
import random

import cv2
import numpy as np

from utils import preprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', default='/workdir/datasets/COCO/train2017/')
    parser.add_argument('--img-size', type=int, nargs=2, default=[416, 416])
    parser.add_argument('--n-img', type=int, default=200)
    parser.add_argument('-o', '--out', default='calib_data.npy')
    return parser.parse_args()


def main():
    # reference:
    # https://github.com/PINTO0309/onnx2tf#7-calibration-data-creation-for-int8-quantization

    args = parse_args()

    img_paths = glob.glob(os.path.join(args.img_dir, '*'))
    img_paths.sort()

    assert len(img_paths) >= args.n_img

    random.seed(0)
    random.shuffle(img_paths)

    calib_data = []
    for i, img_path in enumerate(img_paths):
        if i >= args.n_img:
            break
        img = cv2.imread(img_path)
        img, _ = preprocess(img, (416, 416))
        img = img[np.newaxis]
        calib_data.append(img)
    calib_data = np.vstack(calib_data)  # [n_img, img_size[0], img_size[1], 3]

    print(f'calib_datas.shape: {calib_data.shape}')

    np.save(file=args.out, arr=calib_data)


if __name__ == '__main__':
    main()
