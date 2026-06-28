# Smart Retail Cart - 2026

Minimal YOLO11n pipeline for detecting supermarket products inside a shopping cart. The default model is `yolo11n.pt` for lightweight training/export and Raspberry Pi 4 CPU inference.

## Add Roboflow Dataset

Export the dataset from Roboflow in Ultralytics YOLO format, then unzip it to:

```text
dataset/
  train/images
  train/labels
  valid/images
  valid/labels
  test/images
  test/labels
```

Edit `data/smart_retail_cart.yaml` and update `nc` and `names`.

## Train On PC/Laptop

```bash
pip install -r requirements-train.txt
python src/train.py --data data/smart_retail_cart.yaml --imgsz 416 --batch 8 --workers 2
```

Best weights will be saved at `runs/detect/train/weights/best.pt`.

## Export Model

```bash
python src/export_model.py --model runs/detect/train/weights/best.pt
```

This exports ONNX and tries NCNN when supported.

## Run On Raspberry Pi 4

```bash
pip install -r requirements-rpi.txt
python src/predict_camera.py --model runs/detect/train/weights/best.onnx --camera 0 --conf 0.35 --imgsz 416
```

Use ONNX or NCNN exports when available. Inference defaults to CPU.

## Push To GitHub

```bash
git remote add origin <YOUR_GITHUB_REPO_URL>
git branch -M main
git push -u origin main
```
