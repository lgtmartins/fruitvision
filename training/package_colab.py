"""
Gera fruitvision_colab.zip (dataset + scripts de treino) para enviar ao
Google Drive e treinar com notebooks/treino_colab.ipynb.

Uso:
    python training/package_colab.py
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "fruitvision_colab.zip"

files = [ROOT / "requirements.txt"]
for folder in ("dataset", "training"):
    files += [p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".cache"]

with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in files:
        # as_posix(): barras "/" para o unzip do Colab (Linux) recriar as pastas corretamente
        zf.write(path, path.relative_to(ROOT).as_posix())

print(f"{len(files)} arquivos -> {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.1f} MB)")
