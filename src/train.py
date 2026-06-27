from argparse import ArgumentParser
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_batch(value: str):
    if value.lower() == "auto":
        return -1
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError("--batch must be an integer or 'auto'") from exc


def parse_args():
    parser = ArgumentParser(description="Train YOLOv8n for Smart Retail Cart detection.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLOv8 model.")
    parser.add_argument("--data", default=str(ROOT / "data" / "smart_retail_cart.yaml"), help="Dataset YAML path.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=416, help="Training image size.")
    parser.add_argument("--batch", default="8", help="Batch size or 'auto'.")
    parser.add_argument("--workers", type=int, default=2, help="Low worker count for small machines.")
    parser.add_argument("--device", default=None, help="Device for training, for example cpu or 0.")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)

    train_args = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": parse_batch(args.batch),
        "workers": args.workers,
        "project": str(ROOT / "runs" / "detect"),
        "name": "train",
        "exist_ok": True,
    }
    if args.device:
        train_args["device"] = args.device

    model.train(**train_args)
    print(f"Best model: {ROOT / 'runs' / 'detect' / 'train' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
