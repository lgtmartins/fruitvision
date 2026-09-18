from pathlib import Path
import argparse
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "data.yaml"
DEFAULT_WEIGHTS = ROOT / "runs" / "fruitvision_yolo26n_scratch" / "weights" / "best.pt"

parser = argparse.ArgumentParser(description="Avalia o modelo no conjunto de validação.")
parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Caminho do .pt treinado")
args = parser.parse_args()

if not args.weights.exists():
    raise FileNotFoundError(f"Modelo não encontrado: {args.weights}\nExecute training/train.py primeiro.")

model = YOLO(str(args.weights))
metrics = model.val(data=str(DATA), split="val", plots=True)

print("\n=== MÉTRICAS ===")
print(f"mAP50:     {metrics.box.map50:.4f}")
print(f"mAP50-95:  {metrics.box.map:.4f}")
print(f"Precisão:  {metrics.box.mp:.4f}")
print(f"Recall:    {metrics.box.mr:.4f}")

print("\n=== mAP50 POR CLASSE ===")
for idx, class_idx in enumerate(metrics.box.ap_class_index):
    print(f"  {model.names[int(class_idx)]:<10}: {metrics.box.ap50[idx]:.4f}")
