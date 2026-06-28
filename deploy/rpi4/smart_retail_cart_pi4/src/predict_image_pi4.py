from argparse import ArgumentParser
from pathlib import Path

from ultralytics import YOLO


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
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model, task="detect")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project=str(ROOT / "runs"),
        name="predict",
        exist_ok=True,
    )

    print(f"Processed {len(results)} image(s). Output: {ROOT / 'runs' / 'predict'}")
    for result in results:
        print(Path(result.path).name)
        if result.boxes is None or len(result.boxes) == 0:
            print("  no detections")
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            name = model.names[cls_id]
            print(f"  {name}: {conf:.3f}")


if __name__ == "__main__":
    main()
