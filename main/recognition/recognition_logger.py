# recognition_logger.py
import json
import logging
import datetime
from pathlib import Path
from typing import Optional, Set, Tuple

# Importa la ruta base del proyecto y el umbral alto definido en config.py
from .config import PROJECT_ROOT, HIGH_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

# Umbral estricto: solo reconocer si confianza >= HIGH_CONFIDENCE_THRESHOLD + 5
# Si quieres hacerlo menos estricto, bájalo a HIGH_CONFIDENCE_THRESHOLD.
MIN_STRICT_CONFIDENCE = HIGH_CONFIDENCE_THRESHOLD + 5


class RecognitionLogger:
    """
    Logger de eventos de reconocimiento (conocidos y desconocidos).

    ✔ Guarda archivos JSONL por día:
        src/metrics/recognition/events_YYYYMMDD.jsonl

    ✔ Solo guarda a lo más UN evento por persona (por cámara y por día).

    ✔ Para desconocidos, solo un evento por cámara.
    """

    def __init__(self) -> None:
        # Crear carpeta si no existe
        metrics_root = PROJECT_ROOT / "src" / "metrics" / "recognition"
        metrics_root.mkdir(parents=True, exist_ok=True)

        # Nombre del archivo basado en la fecha
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        self.log_path: Path = metrics_root / f"events_{date_str}.jsonl"

        # Conjunto para evitar duplicados: (tipo, camara, persona_id/tag)
        self.logged_today: Set[Tuple[str, str, str]] = set()

        logger.info(f"[RecognitionLogger] Registrando eventos en: {self.log_path}")

    def log_event(
        self,
        camara: str,
        identity: str,
        confidence: float,
        det_score: Optional[float] = None,
        face_size: Optional[int] = None,
        motivo: Optional[str] = None,
    ) -> None:
        """
        Registra un evento, aplicando:
        - Regla estricta : solo se reconoce si la confianza >= MIN_STRICT_CONFIDENCE
        - Deduplicación   : un registro por persona/cámara por día
        - Manejo correcto de desconocidos
        """

        # Timestamp y ID único
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        event_id = now.strftime("%Y%m%d%H%M%S%f")

        # Lo que dice el modelo (puede no ser confiable)
        model_says_known = identity != "Desconocido"

        # Regla estricta para decidir si lo guardamos como reconocido
        if model_says_known and confidence >= MIN_STRICT_CONFIDENCE:
            recognized = True
            persona_id = identity
            nombre = identity
            motivo = motivo or "match_por_umbral"
            dedup_key = ("known", camara, persona_id)
        else:
            # Rebotear a desconocido
            recognized = False
            persona_id = None
            nombre = "Desconocido"
            motivo = motivo or "bajo_umbral_similitud"
            dedup_key = ("unknown", camara, "Desconocido")

        # Evitar duplicados (solo una vez por día)
        if dedup_key in self.logged_today:
            return

        self.logged_today.add(dedup_key)

        # Armar evento completo
        record = {
            "id_evento": event_id,
            "recognized": recognized,
            "persona_id": persona_id,
            "nombre": nombre,
            "fecha_hora": timestamp,
            "camara": camara,
            "confianza": float(confidence),
            "det_score": float(det_score) if det_score is not None else None,
            "face_size": int(face_size) if face_size is not None else None,
            "motivo": motivo,
        }

        # Escribir en archivo JSONL (una línea por evento)
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"[RecognitionLogger] Error al escribir evento: {e}")
