"""
FruitVision - camada de banco de dados.

Usa SQLite (biblioteca padrão do Python, sem dependências externas) para
guardar o histórico de detecções feitas pelo modelo: cada vez que uma
imagem é analisada, salvamos a imagem anotada e a lista de objetos
detectados (classe, confiança e coordenadas da caixa delimitadora).

Duas tabelas:
- detections: uma linha por imagem enviada para análise.
- detection_items: uma linha por objeto detectado naquela imagem
  (uma imagem pode ter várias frutas detectadas).
"""

import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent / "fruitvision.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Cria as tabelas caso ainda não existam. Chamar uma vez ao iniciar o backend."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_filename TEXT NOT NULL,
            annotated_image_path TEXT NOT NULL,
            conf_threshold REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS detection_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            x1 REAL NOT NULL,
            y1 REAL NOT NULL,
            x2 REAL NOT NULL,
            y2 REAL NOT NULL,
            FOREIGN KEY (detection_id) REFERENCES detections(id) ON DELETE CASCADE
        )
    """)
    conn.commit()
    conn.close()


def save_detection(original_filename, annotated_image_path, conf_threshold, items):
    """
    Salva uma detecção e seus itens.

    items: lista de dicts no formato
        {"class_name": str, "confidence": float, "x1": float, "y1": float, "x2": float, "y2": float}

    Retorna o id da detecção criada.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO detections
           (original_filename, annotated_image_path, conf_threshold, created_at)
           VALUES (?, ?, ?, ?)""",
        (original_filename, annotated_image_path, conf_threshold, datetime.now().isoformat(timespec="seconds")),
    )
    detection_id = cur.lastrowid

    for item in items:
        cur.execute(
            """INSERT INTO detection_items
               (detection_id, class_name, confidence, x1, y1, x2, y2)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                detection_id,
                item["class_name"],
                item["confidence"],
                item["x1"], item["y1"], item["x2"], item["y2"],
            ),
        )

    conn.commit()
    conn.close()
    return detection_id


def get_history(limit=50):
    """Retorna as últimas `limit` detecções, mais recentes primeiro, com seus itens aninhados."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        # id DESC desempata análises feitas no mesmo segundo
        "SELECT * FROM detections ORDER BY datetime(created_at) DESC, id DESC LIMIT ?",
        (limit,),
    )
    detections = [dict(row) for row in cur.fetchall()]

    for d in detections:
        cur.execute(
            "SELECT class_name, confidence, x1, y1, x2, y2 FROM detection_items WHERE detection_id = ?",
            (d["id"],),
        )
        d["items"] = [dict(r) for r in cur.fetchall()]

    conn.close()
    return detections


def get_detection(detection_id):
    """Retorna uma detecção específica com seus itens, ou None se não existir."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM detections WHERE id = ?", (detection_id,))
    row = cur.fetchone()
    if row is None:
        conn.close()
        return None

    detection = dict(row)
    cur.execute(
        "SELECT class_name, confidence, x1, y1, x2, y2 FROM detection_items WHERE detection_id = ?",
        (detection_id,),
    )
    detection["items"] = [dict(r) for r in cur.fetchall()]
    conn.close()
    return detection


def get_stats():
    """Retorna a contagem total de detecções por classe (útil para um painel/relatório)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """SELECT class_name, COUNT(*) as total, AVG(confidence) as avg_confidence
           FROM detection_items
           GROUP BY class_name
           ORDER BY total DESC"""
    )
    stats = [dict(r) for r in cur.fetchall()]
    conn.close()
    return stats


def delete_detection(detection_id):
    """Remove uma detecção e seus itens. Retorna True se algo foi apagado."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM detection_items WHERE detection_id = ?", (detection_id,))
    cur.execute("DELETE FROM detections WHERE id = ?", (detection_id,))
    deleted = cur.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


if __name__ == "__main__":
    init_db()
    print(f"Banco de dados inicializado em: {DB_PATH}")
