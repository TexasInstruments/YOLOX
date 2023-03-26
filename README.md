# YOLOX-ti-lite Object Detection Models

This repository is based on [TexasInstruments/edgeai-yolox](https://github.com/TexasInstruments/edgeai-yolox).

Based on the TexasInstruments' repository, I added new features and some modifications:
- ONNX -> TFLite export with [PINTO0309/onnx2tf](https://github.com/PINTO0309/onnx2tf)
- TFLite int8 quantization with [PINTO0309/onnx2tf](https://github.com/PINTO0309/onnx2tf)
- Minor fix of YOLOX head (see [here](https://github.com/motokimura/edgeai-yolox/blob/bb45abd0a34b2e09df08755f24dff61877860d49/yolox/models/yolo_head.py#L189-L200))

<img src="assets/demo_tflite.png" width="800">

## Requirements

- Docker Compose

## Setup

Download COCO dataset:

```bash
./scripts/get_coco.sh
```

Download YOLOX-ti-lite (nano) pre-trained weight:

```bash
wget http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/detection/coco/edgeai-yolox/yolox_nano_ti_lite_26p1_41p8_checkpoint.pth
```

## Export PyTorch to ONNX

Build and run `torch2onnx` Docker container.

```bash
docker compose build torch2onnx
docker compose run --rm torch2onnx bash
```

Note that `(torch2onnx)` in the code blocks below means you have to run the command in the `torch2onnx` container.

Evaluate PyTorch model (optional):

```bash
(torch2onnx) python tools/eval.py -n yolox_nano_ti_lite -c yolox_nano_ti_lite_26p1_41p8_checkpoint.pth --conf 0.001
```

Expected result:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.262
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.421
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.273
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.095
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.280
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.405
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.245
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.391
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.426
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.174
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.478
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.620
```

Export PyTorch to ONNX:

```bash
(torch2onnx) python tools/export_onnx.py -n yolox_nano_ti_lite -c yolox_nano_ti_lite_26p1_41p8_checkpoint.pth --output-name yolox_nano_ti_lite.onnx
```

Run inference on a sample image with ONNX (optional):

```bash
(torch2onnx) python demo/ONNXRuntime/onnx_inference.py -m yolox_nano_ti_lite.onnx -i assets/dog.jpg -o tmp/onnx/ -s 0.6 --input_shape 416,416 
```

## Export ONNX to TFLite

Build and run `onnx2tf` Docker container.

```bash
docker compose build onnx2tf
docker compose run --rm onnx2tf bash
```

Note that `(onnx2tf)` in the code blocks below means you have to run the command in the `onnx2tf` container.

Generate calibration data from COCO train-set (can be skipped if you don't need quantized TFLite models):

```bash
(onnx2tf) python demo/TFLite/generate_calib_data.py --img-size 416 416 --n-img 200 -o calib_data_416x416_n200.npy
```

Export ONNX to TFLite:

```bash
# If you need quantized TFlite models:
(onnx2tf) onnx2tf -i yolox_nano_ti_lite.onnx -oiqt -qcind images calib_data_416x416_n200.npy "[[[[0,0,0]]]]" "[[[[1,1,1]]]]"

# if you don't need quantized TFLite models:
(onnx2tf) onnx2tf -i yolox_nano_ti_lite.onnx
```

Run inference on a sample image with TFLite:

```bash
# fp32:
(onnx2tf) TFLITE_PATH=saved_model/yolox_nano_ti_lite_float32.tflite
# fp16:
(onnx2tf) TFLITE_PATH=saved_model/yolox_nano_ti_lite_float16.tflite
# int8 static quantized:
(onnx2tf) TFLITE_PATH=saved_model/yolox_nano_ti_lite_integer_quant.tflite
# check under `saved_model/` for some other models.

(onnx2tf) python demo/TFLite/tflite_inference.py -m $TFLITE_PATH -i assets/dog.jpg -o tmp/tflite/ -s 0.6
```

## TODO

- [ ] evaluate mAP of ONNX models
- [ ] evaluate mAP of TFLite models
- [ ] compare inference speed of ONNX, TFLite fp32, and int8 models
- [ ] ONNX/TFlite export of the model decoder and NMS
- [ ] add webcam demo

## Acknowledgements

- [TexasInstruments/edgeai-yolox](https://github.com/TexasInstruments/edgeai-yolox)
- [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [PINTO0309/onnx2tf](https://github.com/PINTO0309/onnx2tf)
- [PINTO0309/PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)
