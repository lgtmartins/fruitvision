from pathlib import Path
import argparse
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "runs" / "fruitvision_yolo26n_scratch" / "weights" / "best.pt"

parser = argparse.ArgumentParser(description="Detecta frutas em uma imagem.")
parser.add_argument("image", help="Caminho da imagem a ser analisada")
parser.add_argument("--conf", type=float, default=0.25, help="Confiança mínima")
parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Caminho do .pt treinado")
args = parser.parse_args()

if not args.weights.exists():
    raise FileNotFoundError("Treine o modelo antes: python training/train.py")

model = YOLO(str(args.weights))
results = model.predict(source=args.image, conf=args.conf, save=True, save_txt=True)

for result in results:
    print("\n=== DETECÇÕES ===")
    if result.boxes is None or len(result.boxes) == 0:
        print("Nenhuma fruta detectada.")
        continue
    for cls, conf in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
        name = result.names[int(cls)]
        print(f"{name}: {conf * 100:.2f}%")
