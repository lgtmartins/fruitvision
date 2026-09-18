from pathlib import Path
from collections import Counter
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset" / "fruit-detection-dataset"
CLASSES = ["apple", "avocado", "banana", "guava", "kiwi", "mango", "orange", "peach", "pineapple"]

for split in ["train", "val"]:
    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split
    images = [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    counts = Counter()
    missing = []
    invalid = []

    for image in images:
        label = label_dir / f"{image.stem}.txt"
        if not label.exists():
            missing.append(image.name)
            continue
        try:
            with Image.open(image) as im:
                im.verify()
        except Exception:
            invalid.append(image.name)
            continue
        for line in label.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                invalid.append(f"{label.name}: esperado 5 valores, veio {len(parts)}")
                continue
            try:
                cls = int(parts[0])
                coords = [float(v) for v in parts[1:]]
            except ValueError:
                invalid.append(f"{label.name}: valor não numérico")
                continue
            if not 0 <= cls < len(CLASSES):
                invalid.append(f"{label.name}: classe {cls} fora do intervalo")
                continue
            if not all(0 <= v <= 1 for v in coords):
                invalid.append(f"{label.name}: coordenada fora de [0, 1]")
                continue
            counts[cls] += 1

    print(f"\n{split.upper()}")
    print(f"Imagens: {len(images)}")
    print(f"Objetos: {sum(counts.values())}")
    for idx, name in enumerate(CLASSES):
        print(f"  {idx:>2} {name:<10}: {counts[idx]}")
    print(f"Labels ausentes: {len(missing)}")
    print(f"Arquivos inválidos: {len(invalid)}")
    for problem in invalid:
        print(f"  - {problem}")
