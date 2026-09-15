import json
import os
import uuid
from datetime import datetime
from pathlib import Path

WORKSPACE_DATA = Path("/workspace/data")
LOCAL_DATA = Path(__file__).resolve().parents[2] / "data"
DATA_DIR = WORKSPACE_DATA if WORKSPACE_DATA.exists() else LOCAL_DATA
HISTORY_FILE = DATA_DIR / "chat_history.json"


def _ensure_storage():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not HISTORY_FILE.exists():
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)


def get_history(limit=None):
    _ensure_storage()
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
            if not isinstance(history, list):
                return []
            if limit:
                return history[-limit:]
            return history
    except Exception:
        return []


def add_interaction(query, answer, sources=None, file_path=None):
    _ensure_storage()
    history = get_history()

    interaction = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "answer": answer,
        "file_path": file_path,
        "sources": sources or []
    }

    history.append(interaction)

    temp_file = HISTORY_FILE.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    temp_file.replace(HISTORY_FILE)

    return interaction


def clear_history():
    _ensure_storage()
    temp_file = HISTORY_FILE.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2)
    temp_file.replace(HISTORY_FILE)
    return True
