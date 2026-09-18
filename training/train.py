"""
FruitVision - treinamento do detector YOLO.

Uso:
    python training/train.py                      # treino completo (padrão: 100 épocas, do zero)
    python training/train.py --epochs 3 --name teste_rapido
    python training/train.py --pretrained         # transfer learning (só se o professor permitir)

O dispositivo é detectado automaticamente: usa a GPU CUDA se existir,
senão cai para a CPU (bem mais lento; prefira o Google Colab com GPU).
"""

from pathlib import Path
import argparse

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "data.yaml"
RUNS = ROOT / "runs"

parser = argparse.ArgumentParser(description="Treina o detector de frutas.")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--patience", type=int, default=20)
parser.add_argument("--device", default=None, help="0 para GPU, 'cpu' para CPU (padrão: automático)")
parser.add_argument("--name", default=None, help="nome da pasta em runs/")
parser.add_argument(
    "--pretrained",
    action="store_true",
    help="usa yolo26n.pt (transfer learning) em vez de treinar do zero",
)
args = parser.parse_args()

# IMPORTANT:
# .yaml = architecture only -> random initialization (training from scratch)
# .pt = pretrained checkpoint -> transfer learning
MODEL_CONFIG = "yolo26n.pt" if args.pretrained else "yolo26n.yaml"
name = args.name or ("fruitvision_yolo26n_pretrained" if args.pretrained else "fruitvision_yolo26n_scratch")
device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")

print(f"Modelo: {MODEL_CONFIG} | dispositivo: {device} | épocas: {args.epochs} | saída: runs/{name}")
if device == "cpu":
    print("Aviso: treinando na CPU. Para o treino completo, prefira o Google Colab com GPU.")

model = YOLO(MODEL_CONFIG)

results = model.train(
    data=str(DATA),
    epochs=args.epochs,
    imgsz=args.imgsz,
    batch=args.batch,
    device=device,
    project=str(RUNS),
    name=name,
    exist_ok=True,
    patience=args.patience,
    workers=2,
    plots=True,
    verbose=True,
)

print("Treinamento concluído.")
print(f"Resultados: {RUNS / name}")
