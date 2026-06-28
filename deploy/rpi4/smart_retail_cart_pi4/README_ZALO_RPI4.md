# Smart Retail Cart - Raspberry Pi 4

Goi nay da co model YOLO11n da train de detect:

- apple
- hao_hao_noodles
- orange
- sprite

Mac dinh chay ban `NCNN 320px` de uu tien FPS tren Raspberry Pi 4 RAM 8GB.

## 1. Giai nen

```bash
unzip smart_retail_cart_pi4_zalo.zip
cd smart_retail_cart_pi4
```

## 2. Cai dat tren Raspberry Pi OS 64-bit

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip libgl1 libglib2.0-0
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements-rpi.txt
```

Neu `pip install ncnn` loi, bo qua NCNN va chay ONNX bang script fallback ben duoi.

## 3. Chay camera USB / OpenCV camera 0

Uu tien nhanh nhat:

```bash
bash run_camera_ncnn_320.sh
```

Neu NCNN khong chay:

```bash
bash run_camera_onnx_320.sh
```

Neu muon chinh xac hon nhung cham hon:

```bash
bash run_camera_ncnn_416.sh
```

Neu SSH/headless khong co man hinh:

```bash
bash run_camera_ncnn_320.sh --no-display
```

Thoat cua so camera bang phim `q` hoac `Esc`.

## 4. Chay test anh

Copy anh vao thu muc `test_images/`, sau do:

```bash
bash run_test_image.sh test_images
```

Ket qua nam trong:

```text
runs/predict/
```

## 5. Goi y toi uu Pi4

- Dung Raspberry Pi OS 64-bit.
- Dung camera USB `/dev/video0` truoc de de test.
- Mac dinh `imgsz 320`, `conf 0.35`, camera `640x480`.
- Khong dung CUDA tren Pi4.
- Neu bi cham, chay them `--frame-skip 1`.
- Neu missed object nho, chuyen sang `run_camera_ncnn_416.sh`.
