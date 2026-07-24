# Smart Retail Cart - 2026

Smart Retail Cart là dự án nhận diện sản phẩm trong xe đẩy siêu thị bằng YOLO11n. Mục tiêu của project là tạo một pipeline gọn, dễ train trên laptop/PC và dễ deploy lên Raspberry Pi 4 RAM 8GB để chạy camera thời gian thực.

Dự án hiện tập trung vào phần computer vision: phát hiện sản phẩm trong khung hình, vẽ bounding box và trả về tên lớp kèm độ tin cậy. Project chưa bao gồm UI web, cơ sở dữ liệu, thanh toán hoặc hệ thống tính tiền.

## Mô Hình

Model chính là `YOLO11n`, bản nano nhẹ của Ultralytics YOLO, phù hợp cho bài toán cần tốc độ và tài nguyên thấp.

Các lớp đang được train:

- `apple`
- `hao_hao_noodles`
- `orange`
- `sprite`

Model sau train được lưu tại:

```text
runs/detect/train/weights/best.pt
```

Các bản export cho Raspberry Pi nằm trong:

```text
deploy/rpi4/smart_retail_cart_pi4/models/
```

## Cấu Trúc Chính

```text
data/smart_retail_cart.yaml          # cấu hình dataset
src/train.py                         # train YOLO11n
src/predict_image.py                 # detect trên ảnh
src/predict_camera.py                # detect bằng camera
src/export_model.py                  # export ONNX và NCNN
deploy/rpi4/smart_retail_cart_pi4/   # gói chạy tối ưu cho Raspberry Pi 4
```

## Dataset Roboflow

Dataset dùng định dạng Ultralytics YOLO. Khi export từ Roboflow, giải nén vào:

```text
dataset/
  train/images
  train/labels
  valid/images
  valid/labels
  test/images
  test/labels
```

File cấu hình dataset:

```text
data/smart_retail_cart.yaml
```

Ví dụ hiện tại:

```yaml
train: ../dataset/train/images
val: ../dataset/valid/images
test: ../dataset/test/images
nc: 4
names:
  - apple
  - hao_hao_noodles
  - orange
  - sprite
```

Lưu ý: thứ tự `names` phải khớp đúng thứ tự class ID trong label Roboflow.

## Train Trên PC/Laptop

Cài thư viện:

```bash
pip install -r requirements-train.txt
```

Train YOLO11n:

```bash
python src/train.py --model yolo11n.pt --data data/smart_retail_cart.yaml --imgsz 416 --batch 8 --workers 2 --epochs 80
```

Kết quả tốt nhất sẽ nằm ở:

```text
runs/detect/train/weights/best.pt
```

## Export Model

Export sang ONNX và thử thêm NCNN:

```bash
python src/export_model.py --model runs/detect/train/weights/best.pt --imgsz 416
```

Gợi ý dùng trên Raspberry Pi:

- `NCNN 320`: nhanh nhất, ưu tiên FPS.
- `ONNX 320`: fallback tùy chọn trên Raspberry Pi OS 64-bit.
- `NCNN 416`: chính xác hơn nhưng chậm hơn.

## Chạy Trên Raspberry Pi 4 Bằng Camera CSI

Gói deploy đã chuẩn bị sẵn tại:

```text
deploy/rpi4/smart_retail_cart_pi4/
```

Camera mặc định của gói này là camera CSI gắn bằng dây ribbon trực tiếp lên board Raspberry Pi 4, không phải webcam USB. Trên Raspberry Pi OS Bullseye/Bookworm, camera CSI nên chạy qua `Picamera2/libcamera`.

Kiểm tra camera CSI trước:

```bash
rpicam-hello --timeout 5000
```

Cài môi trường trên Pi (chạy ngay trong thư mục gói deploy):

```bash
cd deploy/rpi4/smart_retail_cart_pi4
bash install_rpi.sh
```

Script cài đặt tạo `.venv` với `Picamera2` từ Raspberry Pi OS, đồng thời cài NCNN và OpenCV từ wheel có sẵn cho Pi. Đường chạy NCNN mặc định không dùng Ultralytics/PyTorch, nên chạy được cả Raspberry Pi OS 32-bit và 64-bit.

Trên Raspberry Pi 4 với camera CSI, ưu tiên chạy:

```bash
bash run_camera_ncnn_320.sh
```

Nếu cần dùng ONNX trên Raspberry Pi OS 64-bit, cài thêm runtime trước:

```bash
cd deploy/rpi4/smart_retail_cart_pi4
.venv/bin/python -m pip install -r requirements-rpi-onnx.txt
bash run_camera_onnx_320.sh
```

`onnxruntime` không có wheel cho Raspberry Pi OS 32-bit; trên hệ này hãy dùng các script NCNN.

Nếu cần chính xác hơn:

```bash
bash run_camera_ncnn_416.sh
```

Nếu muốn dùng lại webcam USB thay vì CSI:

```bash
bash run_camera_usb_ncnn_320.sh
```

Các script có thể gọi từ bất kỳ thư mục nào. Khi chạy qua SSH không có màn hình, chúng tự chuyển sang chế độ không hiển thị và in FPS mỗi giây. Thêm `--display` nếu Pi đang có màn hình desktop.

Mặc định tối ưu cho Pi4:

- CPU inference, không dùng CUDA.
- Camera CSI qua `Picamera2`.
- Camera `640x480`.
- `imgsz 320`.
- `conf 0.35`.
- Model mặc định: `best_320_ncnn_model`.

Test model trước khi mở camera:

```bash
cd deploy/rpi4/smart_retail_cart_pi4
bash run_test_image.sh
```

## Test Ảnh

Chạy detect trên ảnh hoặc thư mục ảnh:

```bash
python src/predict_image.py --model runs/detect/train/weights/best.pt --source dataset/test/images --conf 0.35 --imgsz 416
```

Kết quả được lưu trong:

```text
runs/detect/predict/
```

## GitHub

Repository:

[https://github.com/ekt0810/Smart-Retail-Cart-2026](https://github.com/ekt0810/Smart-Retail-Cart-2026)

Push cập nhật:

```bash
git add .
git commit -m "Update project documentation"
git push
```

## Ghi Chú

`dataset/` và `runs/` được ignore để tránh push dữ liệu train và log nặng lên GitHub. Riêng gói deploy trong `deploy/rpi4/` có chứa model đã export để người nhận có thể chạy ngay trên Raspberry Pi.
