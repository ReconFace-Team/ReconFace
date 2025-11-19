import json
import threading
from pathlib import Path
from main.recognition import config as cfg

_lock = threading.Lock()

def _ensure_file():
    Path(cfg.ACCESS_CONTROL_DIR).mkdir(parents=True, exist_ok=True)
    p = Path(cfg.ACCESS_CONTROL_JSON)
    if not p.exists():
        p.write_text(json.dumps({"whitelist": [], "blacklist": []}, ensure_ascii=False, indent=2), encoding="utf-8")
    return p

def load_access_lists():
    p = _ensure_file()
    with _lock:
        return json.loads(p.read_text(encoding="utf-8"))

def save_access_lists(data):
    p = _ensure_file()
    with _lock:
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def set_person_status(person_name: str, status: str):
    """
    status in {"white", "black", "none"}
    """
    data = load_access_lists()
    wl = set(data.get("whitelist", []))
    bl = set(data.get("blacklist", []))
    wl.discard(person_name)
    bl.discard(person_name)
    if status == "white":
        wl.add(person_name)
    elif status == "black":
        bl.add(person_name)
    data["whitelist"] = sorted(list(wl))
    data["blacklist"] = sorted(list(bl))
    save_access_lists(data)

def get_person_status(person_name: str) -> str:
    data = load_access_lists()
    if person_name in data.get("blacklist", []):
        return "black"
    if person_name in data.get("whitelist", []):
        return "white"
    return "none"
