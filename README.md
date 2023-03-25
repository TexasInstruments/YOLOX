# YOLOX-ti-lite Object Detection Models

## Install

```bash
docker compose build dev
docker compose run dev bash
```

## Usage

### Evaluate PyTorch model

Download COCO dataset:

```bash
./scripts/get_coco.sh
```

Download YOLOX-ti-lite (nano) pre-trained weight:

```bash
wget http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/detection/coco/edgeai-yolox/yolox_nano_ti_lite_26p1_41p8_checkpoint.pth
```

Evaluate PyTorch model:

```bash
python tools/eval.py -n yolox_nano_ti_lite -c yolox_nano_ti_lite_26p1_41p8_checkpoint.pth --conf 0.001
```

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

### Evaluate ONNX model

Export PyTorch to ONNX:

```bash
python tools/export_onnx.py -n yolox_nano_ti_lite -c yolox_nano_ti_lite_26p1_41p8_checkpoint.pth --output-name yolox_nano_ti_lite.onnx
```

Run inference on a sample image (optional):

```bash
python demo/ONNXRuntime/onnx_inference.py -m yolox_nano_ti_lite.onnx -i assets/dog.jpg -s 0.4 --input_shape 416,416 -o tmp/
```

Evaluate ONNX model:

TBW.

### Evaluate TFLite model

TBW.
