from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter, sleep

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_camera(value: str):
    return int(value) if value.isdigit() else value


class OpenCVCamera:
    def __init__(self, camera, width: int, height: int):
        self.cap = cv2.VideoCapture(parse_camera(camera))
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open OpenCV/USB camera: {camera}")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class Picamera2Camera:
    def __init__(self, camera: str, width: int, height: int):
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "Picamera2 is required for Raspberry Pi CSI camera. "
                "Install on Pi with: sudo apt install -y python3-picamera2"
            ) from exc

        if not str(camera).isdigit():
            raise RuntimeError("--camera must be a number when using CSI/Picamera2.")

        self.picam2 = Picamera2(camera_num=int(camera))
        config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        sleep(0.3)

    def read(self):
        frame_rgb = self.picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return True, frame_bgr

    def release(self):
        self.picam2.stop()
        self.picam2.close()


def create_camera(args):
    backend = args.backend.lower()
    if backend == "auto":
        try:
            return Picamera2Camera(args.camera, args.width, args.height)
        except RuntimeError as exc:
            print(f"CSI camera unavailable, falling back to OpenCV/USB: {exc}")
            return OpenCVCamera(args.camera, args.width, args.height)
    if backend in ("csi", "picamera2"):
        return Picamera2Camera(args.camera, args.width, args.height)
    return OpenCVCamera(args.camera, args.width, args.height)


def parse_args():
    parser = ArgumentParser(description="Run YOLO11 product detection from a camera.")
    parser.add_argument("--backend", default="auto", choices=["auto", "csi", "picamera2", "usb", "opencv"])
    parser.add_argument("--camera", default="0", help="CSI camera number or OpenCV camera index/URL.")
    parser.add_argument("--model", default=str(ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frame-skip", type=int, default=0)
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model, task="detect")
    camera = create_camera(args)

    fps = 0.0
    last_time = perf_counter()
    last_print = last_time

    try:
        while True:
            for _ in range(max(args.frame_skip, 0)):
                camera.read()

            ok, frame = camera.read()
            if not ok:
                print("Camera frame not available.")
                break

            result = model.predict(
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]

            now = perf_counter()
            frame_fps = 1.0 / max(now - last_time, 1e-6)
            fps = frame_fps if fps == 0 else (0.9 * fps + 0.1 * frame_fps)
            last_time = now

            if args.no_display:
                if now - last_print >= 1.0:
                    detections = len(result.boxes) if result.boxes is not None else 0
                    print(f"FPS: {fps:.1f} | detections: {detections}")
                    last_print = now
                continue

            annotated = result.plot()
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Smart Retail Cart", annotated)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
