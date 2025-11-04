# main/recognition/autolearn.py
import os
import cv2
import json
import time
import math
import queue
import shutil
import random
import logging
import threading
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import config as cfg

# --- importar transformations con tolerancia ---
try:
    # si recognition es paquete
    from .transformations import get_transform_options
except Exception:
    try:
        # import absoluto si está en sys.path
        from transformations import get_transform_options
    except Exception:
        # Fallback identidad si no hay transformations.py
        logging.getLogger(__name__).warning(
            "No se encontró transformations.get_transform_options(). Usando augmentations mínimas."
        )
        def get_transform_options():
            # devuelve una lista de “fábricas” que entregan callables con firma tf(image=RGB)['image']
            return [lambda: (lambda image=None, **kw: {"image": image})]

# --- OpenVINO (usar el nuevo import sugerido por DeprecationWarning) ---
try:
    from openvino import Core, get_version  # 2025+
except Exception:
    # compatibilidad con versiones previas
    from openvino.runtime import Core, get_version


logger = logging.getLogger(__name__)

# ========== utilidades de IO ==========
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _utc_now_ts() -> float:
    return time.time()

def _write_jsonl(path: str, obj: Dict):
    _ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _read_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


@dataclass
class QuarantineItem:
    person: str
    confidence_pct: float
    ts: float
    rgb: np.ndarray
    bbox: Tuple[int, int, int, int]  # x1,y1,x2,y2


class AutoLearner:
    """
    Autoaprendizaje con cuarentena:
    - Encola frames de alta confianza (>= AUTOLEARN_QUARANTINE_MIN_CONF_PCT).
    - Sólo considerará promoción si han pasado >= AUTOLEARN_MIN_DAYS_SINCE_ENROLL
      desde el 'enrolamiento' (primera vez que se creó embeddings de esa persona).
    - Al promover, genera AUTOLEARN_BATCH_EMBS embeddings con augmentations
      y los guarda en AUTOLEARN_ALT_EMBEDDINGS_DIR/<person>/.
    """

    def __init__(self):
        # --- rutas desde config ---
        self.quarantine_dir = cfg.AUTOLEARN_QUARANTINE_DIR
        self.alt_embeddings_dir = cfg.AUTOLEARN_ALT_EMBEDDINGS_DIR
        self.db_path = cfg.AUTOLEARN_DB_PATH

        _ensure_dir(self.quarantine_dir)
        _ensure_dir(self.alt_embeddings_dir)

        # colas/locks
        self._q = queue.deque(maxlen=500)  # cola modesta
        self._lock = threading.Lock()

        # flags
        self.enabled = True

        # thresholds & policy
        self.min_days_since_enroll = float(getattr(cfg, "AUTOLEARN_MIN_DAYS_SINCE_ENROLL", 14))
        self.quarantine_min_conf = float(getattr(cfg, "AUTOLEARN_QUARANTINE_MIN_CONF_PCT", 95.0))
        self.promote_min_conf    = float(getattr(cfg, "AUTOLEARN_PROMOTE_MIN_CONF_PCT", 97.5))
        self.batch_embs          = int(getattr(cfg, "AUTOLEARN_BATCH_EMBS", 100))

        # OpenVINO pipeline (reconocimiento puro)
        self._init_openvino_models()

        # augmentations
        self._aug_factories = get_transform_options()

        # cache enrol timestamps (por persona) para la política de “min days since enroll”
        self._enroll_cache: Dict[str, float] = self._build_enroll_cache()

        logger.info("AutoLearner listo (quarantine_dir=%s, alt_dir=%s)", self.quarantine_dir, self.alt_embeddings_dir)

    # ========= openvino =========
    def _init_openvino_models(self):
        core = Core()
        logger.info("OpenVINO runtime: %s | devices=%s", get_version(), core.available_devices)
        compile_cfg = {}
        if getattr(cfg, "OV_PERFORMANCE_HINT", None):
            compile_cfg["PERFORMANCE_HINT"] = cfg.OV_PERFORMANCE_HINT

        # Detector
        self.det_model = core.compile_model(cfg.OV_DET_MODEL, cfg.OV_DEVICE, compile_cfg)
        self.det_out   = self.det_model.outputs[0]

        # Landmarks
        self.lmk_model = core.compile_model(cfg.OV_LMK_MODEL, cfg.OV_DEVICE, compile_cfg)
        self.lmk_out   = self.lmk_model.outputs[0]

        # Reconocimiento (ArcFace)
        self.rec_model = core.compile_model(cfg.OV_REC_MODEL, cfg.OV_DEVICE, compile_cfg)
        self.rec_out   = self.rec_model.outputs[0]

    # ========= helpers de preproc =========
    @staticmethod
    def _to_bgr(img):
        if cfg.EXPECTED_COLOR_ORDER.upper() == "BGR":
            return img
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _blob_from_bgr(bgr, size_hw):
        h, w = size_hw
        blob = cv2.resize(bgr, (w, h))
        blob = blob.transpose(2, 0, 1)[None].astype(np.float32)  # NCHW
        return blob

    def _detect(self, frame_bgr):
        H, W = frame_bgr.shape[:2]
        blob = self._blob_from_bgr(frame_bgr, (256, 256))
        out = self.det_model([blob])[self.det_out]  # [1,1,200,7]
        dets = out[0, 0, :, :]
        faces = []
        thr = float(getattr(cfg, "DET_MIN_CONFIDENCE", 0.6))
        for det in dets:
            conf = float(det[2])
            if conf < thr:
                continue
            x1 = max(0, int(det[3] * W))
            y1 = max(0, int(det[4] * H))
            x2 = min(W - 1, int(det[5] * W))
            y2 = min(H - 1, int(det[6] * H))
            if x2 > x1 and y2 > y1:
                faces.append((conf, (x1, y1, x2, y2)))
        return faces

    def _landmarks(self, face_bgr):
        fh, fw = face_bgr.shape[:2]
        blob = self._blob_from_bgr(face_bgr, cfg.LMK_INPUT_SIZE)
        out  = self.lmk_model([blob])[self.lmk_out].reshape(-1)
        pts  = out.reshape(5, 2)
        pts[:, 0] *= fw
        pts[:, 1] *= fh
        return pts.astype(np.float32)

    @staticmethod
    def _arcface_template():
        return np.array(
            [[38.2946, 51.6963],
             [73.5318, 51.5014],
             [56.0252, 71.7366],
             [41.5493, 92.3655],
             [70.7299, 92.2041]], dtype=np.float32
        )

    def _align112(self, face_bgr, lmk_xy):
        M, _ = cv2.estimateAffinePartial2D(lmk_xy, self._arcface_template(), method=cv2.LMEDS)
        aligned = cv2.warpAffine(face_bgr, M, (cfg.REC_INPUT_SIZE[1], cfg.REC_INPUT_SIZE[0]), flags=cv2.INTER_LINEAR)
        return aligned

    def _embed(self, aligned_bgr):
        blob = self._blob_from_bgr(aligned_bgr, cfg.REC_INPUT_SIZE)
        feat = self.rec_model([blob])[self.rec_out].reshape(-1).astype(np.float32)
        norm = np.linalg.norm(feat) + 1e-9
        return feat / norm

    def _crop_and_embed(self, rgb, bbox) -> Optional[np.ndarray]:
        bgr = self._to_bgr(rgb)
        x1, y1, x2, y2 = map(int, bbox)
        roi = bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        lmk = self._landmarks(roi)
        aligned = self._align112(roi, lmk)
        emb = self._embed(aligned)
        return emb

    # ========= política de enroll =========
    def _build_enroll_cache(self) -> Dict[str, float]:
        """
        Intenta estimar “cuándo se enroló” cada persona.
        Como heurística: usa el mtime de la primera carpeta de embeddings base.
        """
        base_dir = cfg.EMBEDDING_DIR
        cache = {}
        if not os.path.exists(base_dir):
            return cache
        for person in os.listdir(base_dir):
            pdir = os.path.join(base_dir, person)
            if not os.path.isdir(pdir):
                continue
            try:
                # heurística: fecha de creación/primer archivo
                mtimes = []
                for f in os.listdir(pdir):
                    fp = os.path.join(pdir, f)
                    try:
                        mtimes.append(os.path.getmtime(fp))
                    except Exception:
                        pass
                if mtimes:
                    cache[person] = min(mtimes)
            except Exception:
                pass
        return cache

    def _enrolled_long_enough(self, person: str) -> bool:
        if person not in self._enroll_cache:
            # si no hay registro, asumimos que sí (o podrías negar)
            return True
        days = ( _utc_now_ts() - self._enroll_cache[person] ) / 86400.0
        return days >= self.min_days_since_enroll

    # ========= API pública =========
    def enabled_set(self, flag: bool):
        with self._lock:
            self.enabled = bool(flag)

    def queue_len(self) -> int:
        with self._lock:
            return len(self._q)

    def maybe_queue(self, frame_bgr_or_rgb, identity: str, bbox, confidence_pct: float):
        """
        Encola un frame si cumple confianza y política de días desde enrolamiento.
        Recibe frame en BGR (de cámara) o RGB, lo pasamos a RGB para augmentations consistentes.
        """
        if not self.enabled:
            return
        if confidence_pct < self.quarantine_min_conf:
            return
        if not self._enrolled_long_enough(identity):
            return

        # normalizar a RGB
        if frame_bgr_or_rgb is None:
            return
        bgr = frame_bgr_or_rgb
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        x1, y1, x2, y2 = map(int, bbox)
        h, w = rgb.shape[:2]
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > w or y2 > h:
            return

        item = QuarantineItem(
            person=identity,
            confidence_pct=float(confidence_pct),
            ts=_utc_now_ts(),
            rgb=rgb.copy(),
            bbox=(x1, y1, x2, y2),
        )
        with self._lock:
            self._q.append(item)

        # opcional: persistir mini-metadata en JSONL
        _write_jsonl(self.db_path, {
            "ts": item.ts,
            "event": "queued",
            "person": identity,
            "conf_pct": confidence_pct,
            "bbox": [x1, y1, x2, y2],
        })

    def maybe_process_periodically(self, frame_count: int, every_n: int = 30):
        """Llama de vez en cuando desde el loop principal."""
        if not self.enabled:
            return
        if frame_count % max(1, int(every_n)) == 0:
            self._process_quarantine()

    def force_process_now(self) -> Tuple[int, int]:
        """Procesa todo lo que haya en cola, retorna (promoted, skipped)."""
        return self._process_quarantine(force_all=True)

    # ========= núcleo de promoción =========
    def _process_quarantine(self, force_all: bool = False) -> Tuple[int, int]:
        promoted = 0
        skipped = 0

        pending: List[QuarantineItem] = []
        with self._lock:
            if not self._q:
                return (0, 0)
            # consume todo o una parte
            if force_all:
                while self._q:
                    pending.append(self._q.popleft())
            else:
                # procesa un lote pequeño por invocación
                for _ in range(min(len(self._q), 4)):
                    pending.append(self._q.popleft())

        for item in pending:
            try:
                # Re-verifica confianza mínima para promover (puede ser mayor a la de cuarentena)
                if item.confidence_pct < self.promote_min_conf:
                    skipped += 1
                    continue

                # Embedding del frame (face crop + align)
                emb = self._crop_and_embed(item.rgb, item.bbox)
                if emb is None or not np.isfinite(emb).all():
                    skipped += 1
                    continue

                # Generar lote de embeddings alternos (augmentations)
                n = max(1, int(self.batch_embs))
                self._generate_alt_embeddings(item.person, item.rgb, item.bbox, n)

                _write_jsonl(self.db_path, {
                    "ts": _utc_now_ts(),
                    "event": "promoted",
                    "person": item.person,
                    "batch_embs": n,
                })
                promoted += 1

            except Exception as e:
                logger.error("AutoLearn promote error: %s", e)
                skipped += 1

        return (promoted, skipped)

    def _generate_alt_embeddings(self, person: str, rgb: np.ndarray, bbox, n: int):
        """
        Crea n embeddings alternos para 'person' aplicando augmentations al crop facial.
        Guarda cada vector en AUTOLEARN_ALT_EMBEDDINGS_DIR/<person>/...
        """
        x1, y1, x2, y2 = map(int, bbox)
        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return

        out_dir = os.path.join(self.alt_embeddings_dir, person)
        _ensure_dir(out_dir)

        # guardar snapshot de cuarentena (opcional/útil para auditorías)
        try:
            qshot = os.path.join(cfg.AUTOLEARN_QUARANTINE_DIR, f"{person}_{int(_utc_now_ts())}.jpg")
            _ensure_dir(os.path.dirname(qshot))
            cv2.imwrite(qshot, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        except Exception:
            pass

        factories = self._aug_factories if self._aug_factories else [lambda: (lambda image=None, **kw: {"image": image})]

        saved = 0
        attempts = 0
        while saved < n and attempts < n * 10:
            attempts += 1
            tf = random.choice(factories)()
            try:
                aug = tf(image=crop)['image']  # RGB
            except Exception:
                aug = crop

            bgr = cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)
            # re-detect dentro del crop (por si la aug movió algo); si no detecta, usamos bbox completo
            faces = self._detect(bgr)
            if faces:
                faces.sort(key=lambda t: t[0], reverse=True)
                _, bb = faces[0]
                emb = self._crop_and_embed(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), bb)
            else:
                # embed del crop original (alineando con landmarks)
                emb = self._crop_and_embed(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (0, 0, bgr.shape[1], bgr.shape[0]))

            if emb is None:
                continue

            # save
            fname = os.path.join(out_dir, f"{person}_autolearn_{int(_utc_now_ts())}_{saved:04d}.npy")
            np.save(fname, emb.astype(np.float32))
            saved += 1
