# YOLOX-ti-lite Object Detection Models

## Install

```bash
docker compose build dev
docker compose run dev bash
```

## Usage

### ONNX Export

```bash
wget http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/detection/coco/edgeai-yolox/yolox_nano_ti_lite_26p1_41p8_checkpoint.pth

python tools/export_onnx.py -n yolox_nano_ti_lite -c yolox_nano_ti_lite_26p1_41p8_checkpoint.pth --output-name yolox_nano_ti_lite.onnx

python demo/ONNXRuntime/onnx_inference.py -m yolox_nano_ti_lite.onnx -i assets/dog.jpg -s 0.4 --input_shape 416,416 -o tmp/
```
