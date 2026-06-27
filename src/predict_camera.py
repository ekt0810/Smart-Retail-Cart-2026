from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_camera(value: str):
    return int(value) if value.isdigit() else value


def parse_args():
    parser = ArgumentParser(description="Run YOLOv8 product detection from a USB/Pi camera.")
    parser.add_argument("--camera", default="0", help="OpenCV camera index or stream URL.")
    parser.add_argument("--model", default=str(ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    cap = cv2.VideoCapture(parse_camera(args.camera))

    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise SystemExit(f"Could not open camera: {args.camera}")

    fps = 0.0
    last_time = perf_counter()

    try:
        while True:
            ok, frame = cap.read()
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

            if args.no_display:
                print(f"FPS: {fps:.1f}", end="\r")
                continue

            cv2.imshow("Smart Retail Cart", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
