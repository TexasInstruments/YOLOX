# YOLOX-ti-lite Object Detection Models

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

Note that `(torch2onnx)` in the code blocks below means you have to run the commands in the `torch2onnx` container.

Evaluate PyTorch model (optional):

```bash
(torch2onnx) python tools/eval.py -n yolox_nano_ti_lite -c yolox_nano_ti_lite_26p1_41p8_checkpoint.pth --conf 0.001
```

Expected result:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.263
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
(torch2onnx) python demo/ONNXRuntime/onnx_inference.py -m yolox_nano_ti_lite.onnx -i assets/dog.jpg -s 0.4 --input_shape 416,416 -o tmp/
```

Evaluate ONNX model:

TBW.

## Export ONNX to TFLite

Build and run `onnx2tf` Docker container.

```bash
docker compose build onnx2tf
docker compose run --rm onnx2tf bash
```

Note that `(onnx2tf)` in the code blocks below means you have to run the commands in the `onnx2tf` container.

Export PyTorch to ONNX:

```bash
(onnx2tf) onnx2tf -i yolox_nano_ti_lite.onnx
```

Run inference on a sample image with TFLite (optional):

TBW.

Evaluate TFLite model:

TBW.
