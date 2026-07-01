# Smart Retail Cart - Hướng Dẫn Chạy Trên Raspberry Pi 4

Gói này dùng để chạy model nhận diện sản phẩm trong xe đẩy siêu thị trên Raspberry Pi 4 RAM 8GB. Model đã được train bằng YOLO11n và đã export sẵn sang các định dạng nhẹ hơn để chạy inference trên CPU.

Các sản phẩm model đang nhận diện:

- `apple`
- `hao_hao_noodles`
- `orange`
- `sprite`

Mặc định gói này dùng camera CSI gắn bằng dây ribbon trực tiếp lên board Raspberry Pi 4. Code đã chuyển sang `Picamera2/libcamera`, không còn mặc định dùng webcam USB qua OpenCV nữa.

Mặc định nên chạy bản `NCNN 320px` vì nhẹ và nhanh nhất trên Raspberry Pi 4. Nếu cần nhận diện chính xác hơn, có thể chuyển sang bản `NCNN 416px`.

## 1. Giải Nén Gói

```bash
unzip smart_retail_cart_pi4_zalo.zip
cd smart_retail_cart_pi4
```

## 2. Cài Đặt Môi Trường

Khuyến nghị dùng Raspberry Pi OS 64-bit.

Kiểm tra camera CSI trước:

```bash
rpicam-hello --timeout 5000
```

Nếu lệnh trên không mở được preview hoặc báo không thấy camera, kiểm tra lại dây ribbon, chiều cắm cáp, nguồn cấp và cấu hình camera trong Raspberry Pi OS.

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-picamera2 python3-opencv libgl1 libglib2.0-0
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements-rpi.txt
```

Phải tạo venv với `--system-site-packages` để Python thấy được thư viện `picamera2` cài bằng `apt`.

Nếu cài `ncnn` bị lỗi, vẫn có thể chạy bằng ONNX ở bước fallback bên dưới.

## 3. Chạy Camera CSI

Chạy bản nhanh nhất, khuyến nghị dùng đầu tiên:

```bash
bash run_camera_ncnn_320.sh
```

Nếu NCNN không chạy được:

```bash
bash run_camera_onnx_320.sh
```

Nếu muốn chính xác hơn, đổi sang ảnh vào lớn hơn:

```bash
bash run_camera_ncnn_416.sh
```

Nếu chạy qua SSH hoặc không có màn hình:

```bash
bash run_camera_ncnn_320.sh --no-display
```

Thoát cửa sổ camera bằng phím `q` hoặc `Esc`.

Nếu cần quay lại webcam USB, dùng script riêng:

```bash
bash run_camera_usb_ncnn_320.sh
```

## 4. Chạy Test Bằng Ảnh

Copy ảnh cần test vào thư mục:

```text
test_images/
```

Sau đó chạy:

```bash
bash run_test_image.sh test_images
```

Kết quả ảnh đã vẽ bounding box nằm trong:

```text
runs/predict/
```

## 5. Các Model Có Sẵn

```text
models/best_320_ncnn_model/   # ưu tiên chạy trên Pi4, nhanh nhất
models/best_320.onnx          # fallback khi NCNN lỗi
models/best_416_ncnn_model/   # chính xác hơn, chậm hơn
models/best_416.onnx          # fallback ONNX ở 416px
models/best.pt                # model PyTorch gốc
```

## 6. Cấu Hình Mặc Định

- Camera: `0`
- Backend camera: `csi`
- Độ phân giải camera: `640x480`
- Kích thước inference: `320`
- Ngưỡng confidence: `0.35`
- Thiết bị chạy: `CPU`
- Không dùng CUDA trên Raspberry Pi 4

## 7. Gợi Ý Tối Ưu

Nếu FPS thấp:

```bash
bash run_camera_ncnn_320.sh --frame-skip 1
```

Nếu bị bỏ sót vật thể nhỏ:

```bash
bash run_camera_ncnn_416.sh
```

Nếu camera không mở được, thử kiểm tra camera:

```bash
rpicam-hello --timeout 5000
```

Nếu có nhiều camera CSI, thử đổi camera index:

```bash
bash run_camera_ncnn_320.sh --camera 1
```

Nếu dùng webcam USB thì mới kiểm tra:

```bash
ls /dev/video*
bash run_camera_usb_ncnn_320.sh --camera 0
```

## 8. Ghi Chú

Đây là model nhận diện sản phẩm, chưa phải hệ thống tính tiền hoàn chỉnh. Phần ứng dụng có thể lấy kết quả detection từ script Python để tích hợp tiếp vào UI, giỏ hàng hoặc luồng xử lý thanh toán.
