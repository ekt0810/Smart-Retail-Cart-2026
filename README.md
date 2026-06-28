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
- `ONNX 320`: fallback nếu NCNN lỗi.
- `NCNN 416`: chính xác hơn nhưng chậm hơn.

## Chạy Trên Raspberry Pi 4

Gói deploy đã chuẩn bị sẵn tại:

```text
deploy/rpi4/smart_retail_cart_pi4/
```

File zip để gửi qua Zalo:

```text
deploy/rpi4/smart_retail_cart_pi4_zalo.zip
```

Trên Raspberry Pi 4, ưu tiên chạy:

```bash
bash run_camera_ncnn_320.sh
```

Nếu NCNN không chạy:

```bash
bash run_camera_onnx_320.sh
```

Nếu cần chính xác hơn:

```bash
bash run_camera_ncnn_416.sh
```

Mặc định tối ưu cho Pi4:

- CPU inference, không dùng CUDA.
- Camera `640x480`.
- `imgsz 320`.
- `conf 0.35`.
- Model mặc định: `best_320_ncnn_model`.

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
