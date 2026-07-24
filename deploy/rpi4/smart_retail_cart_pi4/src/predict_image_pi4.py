from argparse import ArgumentParser
from pathlib import Path

from ncnn_detector import create_detector


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = [
    ROOT / "models" / "best_320_ncnn_model",
    ROOT / "models" / "best_320.onnx",
    ROOT / "models" / "best_416_ncnn_model",
    ROOT / "models" / "best_416.onnx",
    ROOT / "models" / "best.pt",
]


def default_model():
    for path in DEFAULT_MODELS:
        if path.exists():
            return str(path)
    return str(DEFAULT_MODELS[0])


def parse_args():
    parser = ArgumentParser(description="Smart Retail Cart image inference for Raspberry Pi 4.")
    parser.add_argument("--model", default=default_model())
    parser.add_argument("--source", required=True)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=320)
    return parser.parse_args()


def main():
    args = parse_args()
    import cv2

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"Image or directory not found: {source}")
    image_extensions = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
    images = [source] if source.is_file() else sorted(path for path in source.iterdir() if path.suffix.lower() in image_extensions)
    if not images:
        raise SystemExit(f"No supported images found in: {source}")

    try:
        detector = create_detector(args.model, args.imgsz, args.conf, args.iou)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output_dir = ROOT / "runs" / "predict"
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Skipped unreadable image: {image_path.name}")
            continue
        detections = detector.predict(frame)
        from ncnn_detector import annotate

        output_path = output_dir / image_path.name
        if not cv2.imwrite(str(output_path), annotate(frame, detections, detector.names)):
            raise RuntimeError(f"Could not write output image: {output_path}")
        print(image_path.name)
        if not detections:
            print("  no detections")
        for detection in detections:
            name = detector.names[detection.class_id] if detection.class_id < len(detector.names) else str(detection.class_id)
            print(f"  {name}: {detection.confidence:.3f}")

    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
