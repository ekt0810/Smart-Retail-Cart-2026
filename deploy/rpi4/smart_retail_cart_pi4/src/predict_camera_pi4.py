import os
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter, sleep

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2

from ncnn_detector import annotate, create_detector


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = [
    ROOT / "models" / "best_320_ncnn_model",
    ROOT / "models" / "best_320.onnx",
    ROOT / "models" / "best_416_ncnn_model",
    ROOT / "models" / "best_416.onnx",
]


def default_model():
    for path in DEFAULT_MODELS:
        if path.exists():
            return str(path)
    return str(DEFAULT_MODELS[0])


def parse_camera(value: str):
    return int(value) if value.isdigit() else value


class OpenCVCamera:
    def __init__(self, camera, width: int, height: int):
        self.cap = cv2.VideoCapture(parse_camera(camera))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if isinstance(parse_camera(camera), int):
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
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
                "Picamera2 is required for Raspberry Pi CSI camera. Run: bash install_rpi.sh"
            ) from exc

        if not str(camera).isdigit():
            raise RuntimeError("--camera must be a number when using CSI/Picamera2.")
        self.picam2 = Picamera2(camera_num=int(camera))
        config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        sleep(0.5)

    def read(self):
        frame_rgb = self.picam2.capture_array()
        return True, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

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
    parser = ArgumentParser(description="Smart Retail Cart camera inference for Raspberry Pi 4.")
    parser.add_argument("--backend", default="csi", choices=["auto", "csi", "picamera2", "usb", "opencv"])
    parser.add_argument("--model", default=default_model())
    parser.add_argument("--camera", default="0")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--cv-threads", type=int, default=2)
    parser.add_argument("--frame-skip", type=int, default=0, help="Skip captured frames before each inference.")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--display", dest="display", action="store_true", help="Force an OpenCV preview window.")
    display.add_argument("--no-display", dest="display", action="store_false", help="Disable the preview window.")
    parser.set_defaults(display=bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")))
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.conf <= 1 or not 0 < args.iou <= 1:
        raise SystemExit("--conf and --iou must be values greater than 0 and at most 1.")
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    cv2.setNumThreads(max(args.cv_threads, 0))
    try:
        detector = create_detector(model_path, args.imgsz, args.conf, args.iou, args.cv_threads)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
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

            detections = detector.predict(frame)
            now = perf_counter()
            frame_fps = 1.0 / max(now - last_time, 1e-6)
            fps = frame_fps if fps == 0 else 0.9 * fps + 0.1 * frame_fps
            last_time = now

            if not args.display:
                if now - last_print >= 1.0:
                    print(f"FPS: {fps:.1f} | detections: {len(detections)}")
                    last_print = now
                continue

            annotated = annotate(frame, detections, detector.names)
            cv2.putText(
                annotated, f"FPS: {fps:.1f}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA
            )
            cv2.imshow("Smart Retail Cart - Pi4", annotated)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
