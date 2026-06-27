from argparse import ArgumentParser
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = ArgumentParser(description="Export YOLOv8 model for Smart Retail Cart inference.")
    parser.add_argument("--model", default=str(ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"))
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-ncnn", action="store_true", help="Skip optional NCNN export.")
    return parser.parse_args()


def export_format(model: YOLO, fmt: str, imgsz: int, device: str):
    try:
        output = model.export(format=fmt, imgsz=imgsz, device=device)
        print(f"Exported {fmt}: {output}")
        return True
    except Exception as exc:
        print(f"Could not export {fmt}: {exc}")
        if fmt == "onnx":
            print("Install export packages: pip install onnx onnxruntime")
        if fmt == "ncnn":
            print("NCNN export is optional. Try: pip install -U ultralytics ncnn")
        return False


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    model = YOLO(str(model_path))
    export_format(model, "onnx", args.imgsz, args.device)

    if not args.skip_ncnn:
        export_format(model, "ncnn", args.imgsz, args.device)


if __name__ == "__main__":
    main()
