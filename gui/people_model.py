import os
from pathlib import Path
from main.recognition import config as cfg
from .storage import get_person_status, set_person_status

def list_identities():
    """
    Devuelve las personas según carpetas en EMBEDDING_DIR (un nivel).
    """
    root = Path(cfg.EMBEDDING_DIR)
    if not root.exists():
        return []
    people = []
    for d in sorted(p.name for p in root.iterdir() if p.is_dir()):
        people.append({
            "name": d,
            "status": get_person_status(d),
        })
    return people

def set_status(name: str, status: str):
    set_person_status(name, status)
