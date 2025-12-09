import os
import time
import json
import logging
import datetime
from pathlib import Path

from .config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# psutil (opcional)
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    psutil = None
    logger.warning("[PerfMonitor] psutil no instalado. CPU/RAM irán como null.")

# GPU NVIDIA vía pynvml (opcional)
try:
    import pynvml
    pynvml.nvmlInit()
    _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _HAS_NVIDIA = True
    logger.info("[PerfMonitor] GPU NVIDIA detectada, logging habilitado.")
except Exception:
    _HAS_NVIDIA = False
    _GPU_HANDLE = None
    logger.info("[PerfMonitor] No se detectó GPU NVIDIA o pynvml no disponible.")


class PerformanceMonitor:
    """
    Monitor de rendimiento para reconocimiento y entrenamiento.

    Guarda un archivo JSONL cada "interval_sec" segundos en:

        src/metrics/recognition/perf_YYYYMMDD.jsonl     (fase runtime)
        src/metrics/training/perf_YYYYMMDD.jsonl        (fase training)
    """

    def __init__(self, interval_sec: int = 5, phase: str = "runtime"):
        """
        phase:
            "runtime"  -> reconocimiento en vivo
            "training" -> proceso de entrenamiento
        """
        self.interval_sec = max(1, int(interval_sec))
        self.phase = phase.lower().strip()

        # Seleccionar carpeta según fase
        metrics_root = PROJECT_ROOT / "src" / "metrics"

        if self.phase == "training":
            self.log_dir = metrics_root / "training"
        else:
            self.log_dir = metrics_root / "recognition"

        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Archivo JSON por día
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        self.log_path = self.log_dir / f"perf_{date_str}.jsonl"

        self.start_time = time.time()
        self.last_flush = self.start_time

        # Contadores
        self.frames_interval = 0
        self.known_interval = 0
        self.unknown_interval = 0

        # psutil process
        if _HAS_PSUTIL:
            try:
                self.process = psutil.Process(os.getpid())
                self.process.cpu_percent(interval=None)  # primer tick
            except Exception:
                self.process = None
        else:
            self.process = None

        logger.info(
            f"[PerfMonitor] Iniciado (fase={self.phase}). "
            f"Intervalo={self.interval_sec}s. Archivo={self.log_path}"
        )

    # ----------------------------------------------------------------------
    # tick — se llama cada iteración del pipeline
    # ----------------------------------------------------------------------
    def tick(self, num_known: int, num_unknown: int, frames: int = 1):
        """
        num_known   -> runtime: caras conocidas / training: items procesados
        num_unknown -> runtime: caras desconocidas / training: opcional
        frames      -> runtime: frames utilizados / training: unidades de trabajo
        """
        self.frames_interval += max(0, int(frames))
        self.known_interval += max(0, int(num_known))
        self.unknown_interval += max(0, int(num_unknown))

        now = time.time()
        if now - self.last_flush >= self.interval_sec:
            self._flush(now)

    # ----------------------------------------------------------------------
    # _flush — escribe un JSON
    # ----------------------------------------------------------------------
    def _flush(self, now: float):
        interval = now - self.last_flush
        uptime = now - self.start_time

        fps = self.frames_interval / interval if interval > 0 else 0.0

        # CPU / RAM
        cpu_percent = None
        ram_mb = None
        if self.process:
            try:
                cpu_percent = self.process.cpu_percent(interval=None)
                ram_mb = self.process.memory_info().rss / (1024**2)
            except Exception:
                pass

        # GPU NVIDIA
        gpu_percent = None
        gpu_mem_mb = None
        if _HAS_NVIDIA and _GPU_HANDLE:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(_GPU_HANDLE)
                mem = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
                gpu_percent = util.gpu
                gpu_mem_mb = mem.used / (1024**2)
            except Exception:
                pass

        total_faces = self.known_interval + self.unknown_interval
        timestamp = datetime.datetime.fromtimestamp(now).isoformat(timespec="seconds")

        record = {
            "phase": self.phase,
            "timestamp": timestamp,
            "uptime_sec": round(uptime, 2),
            "cpu_percent": cpu_percent,
            "ram_mb": ram_mb,
            "gpu_percent": gpu_percent,
            "gpu_mem_mb": gpu_mem_mb,
            "fps": round(fps, 2),
            "frames_interval": int(self.frames_interval),
            "known_faces": int(self.known_interval),
            "unknown_faces": int(self.unknown_interval),
            "total_faces": int(total_faces),
        }

        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"[PerfMonitor] Error escribiendo JSONL: {e}")

        # Reset
        self.frames_interval = 0
        self.known_interval = 0
        self.unknown_interval = 0
        self.last_flush = now
