from argparse import ArgumentParser
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = ArgumentParser(description="Detect products in a shopping cart image.")
    parser.add_argument("--model", default=str(ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"))
    parser.add_argument("--source", required=True, help="Image path or folder.")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project=str(ROOT / "runs" / "detect"),
        name="predict",
        exist_ok=True,
    )
    print(f"Processed {len(results)} image(s). Output: {ROOT / 'runs' / 'detect' / 'predict'}")


if __name__ == "__main__":
    main()
