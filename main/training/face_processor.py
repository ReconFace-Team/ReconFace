import os
import cv2
import time
import random
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from openvino.runtime import Core, get_version

from transformations import get_transform_options
from utils import (
    show_preview, resize_image_if_needed, create_output_directory,
    count_existing_embeddings, check_if_image_processed
)
from config import (
    MIN_DET_SCORE, MAX_ATTEMPTS, REGENERATE_TRANSFORM_INTERVAL,
    PREVIEW_INTERVAL, LOG_INTERVAL, N_AUGMENTATIONS,
    CHECK_EXISTING_EMBEDDINGS, SKIP_COMPLETED_IMAGES,
    INPUT_DIR, OUTPUT_DIR,
    OV_DEVICE, OV_PERFORMANCE_HINT, OV_DET_MODEL, OV_LMK_MODEL, OV_REC_MODEL,
    LMK_INPUT_SIZE, REC_INPUT_SIZE, EXPECTED_COLOR_ORDER,
    SHOW_PREVIEW
)

# ====== Parámetros opcionales para reintentos en imagen original (con defaults) ======
try:
    from config import ORIG_DET_RETRIES, ORIG_SCALES, ORIG_USE_ENHANCERS, ORIG_PREVIEW_EACH_HIT
except Exception:
    ORIG_DET_RETRIES = 10
    ORIG_SCALES = [1.0, 1.2, 0.8, 1.4, 0.6, 1.6, 0.7]
    ORIG_USE_ENHANCERS = True
    ORIG_PREVIEW_EACH_HIT = True

# ====== Objetivo por persona ======
try:
    from config import TARGET_EMB_PER_PERSON, AUGS_PER_IMAGE_MIN, AUGS_PER_IMAGE_MAX
except Exception:
    TARGET_EMB_PER_PERSON = 1000
    AUGS_PER_IMAGE_MIN = 0
    AUGS_PER_IMAGE_MAX = 200

# ====== Borrado al terminar ======
try:
    from config import DELETE_SOURCE_IMAGES, DELETE_REQUIRE_FULL_AUGS, MOVE_TO_TRASH
except Exception:
    DELETE_SOURCE_IMAGES = False
    DELETE_REQUIRE_FULL_AUGS = True
    MOVE_TO_TRASH = False  # permanente

# ====== Buckets multiescala (si no existen en config.py, usa defaults) ======
try:
    from config import SCALE_BUCKETS, SMALL_CANVAS_MULT, LARGE_ZOOM
except Exception:
    SCALE_BUCKETS = {"small": 0.34, "medium": 0.33, "large": 0.33}
    SMALL_CANVAS_MULT = 1.7
    LARGE_ZOOM = 1.6

logger = logging.getLogger(__name__)

BUCKETS = ("small", "medium", "large")  # orden consistente

# Plantilla 5 puntos ArcFace 112x112
ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963],
     [73.5318, 51.5014],
     [56.0252, 71.7366],
     [41.5493, 92.3655],
     [70.7299, 92.2041]], dtype=np.float32
)

@dataclass
class DetectedFace:
    bbox: np.ndarray                 # np.array([x1,y1,x2,y2], int32)
    det_score: float
    embedding: Optional[np.ndarray] = None  # (512,) o None


class FaceProcessor:
    """Detección y generación de embeddings con OpenVINO + OMZ (ArcFace 512D) + buckets exactos por persona"""

    def __init__(self):
        print("🔄 Inicializando OpenVINO (training)...")
        # Log de rutas a modelos (ayuda a depurar)
        print("   DET:", OV_DET_MODEL, "exists:", os.path.exists(OV_DET_MODEL))
        print("   LMK:", OV_LMK_MODEL, "exists:", os.path.exists(OV_LMK_MODEL))
        print("   REC:", OV_REC_MODEL, "exists:", os.path.exists(OV_REC_MODEL))

        try:
            core = Core()
            print("   OpenVINO runtime:", get_version())
            print("   Dispositivos disponibles:", core.available_devices)
            for dev in core.available_devices:
                try:
                    name = core.get_property(dev, "FULL_DEVICE_NAME")
                    print(f"   - {dev}: {name}")
                except Exception as e:
                    print(f"   - {dev}: (sin nombre) {e}")

            compile_cfg = {"PERFORMANCE_HINT": OV_PERFORMANCE_HINT} if OV_PERFORMANCE_HINT else {}

            print("   Compilando detector...")
            self.det_model = core.compile_model(OV_DET_MODEL, OV_DEVICE, compile_cfg)
            print("   ✔ detector OK")

            print("   Compilando landmarks...")
            self.lmk_model = core.compile_model(OV_LMK_MODEL, OV_DEVICE, compile_cfg)
            print("   ✔ landmarks OK")

            print("   Compilando reconocimiento...")
            self.rec_model = core.compile_model(OV_REC_MODEL, OV_DEVICE, compile_cfg)
            print("   ✔ reconocimiento OK")

        except Exception as e:
            print("❌ Error compilando modelos OpenVINO:", repr(e))
            raise

        self.det_in = self.det_model.inputs[0]
        self.det_out = self.det_model.outputs[0]
        self.lmk_in = self.lmk_model.inputs[0]
        self.lmk_out = self.lmk_model.outputs[0]
        self.rec_in = self.rec_model.inputs[0]
        self.rec_out = self.rec_model.outputs[0]

        self.transform_options = get_transform_options()
        print("✅ Modelos OpenVINO listos")

    # ----------------- Helpers de preproc/inferencia -----------------
    def _to_bgr(self, img):
        if EXPECTED_COLOR_ORDER.upper() == "BGR":
            return img
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def _blob_from_bgr(self, bgr, size_hw):
        h, w = size_hw  # (H, W)
        blob = cv2.resize(bgr, (w, h))
        blob = blob.transpose(2, 0, 1)[None].astype(np.float32)  # NCHW float32
        return blob

    def _detect(self, frame_bgr) -> List[DetectedFace]:
        """Detector face-detection-0200 con timing y postpro."""
        t0 = time.time()
        H, W = frame_bgr.shape[:2]
        blob = self._blob_from_bgr(frame_bgr, (256, 256))
        try:
            out = self.det_model([blob])[self.det_out]  # [1,1,200,7]
        except Exception as e:
            print(f"❌ Error en inferencia DET: {e}")
            return []
        dt = (time.time() - t0) * 1000
        print(f"⏱️ DET inferencia: {dt:.1f} ms")

        dets = out[0, 0, :, :]
        faces: List[DetectedFace] = []
        thr = float(MIN_DET_SCORE)
        for det in dets:
            conf = float(det[2])
            if conf < thr:
                continue
            x1 = max(0, int(det[3] * W))
            y1 = max(0, int(det[4] * H))
            x2 = min(W - 1, int(det[5] * W))
            y2 = min(H - 1, int(det[6] * H))
            if x2 > x1 and y2 > y1:
                faces.append(DetectedFace(np.array([x1, y1, x2, y2], dtype=np.int32), conf))
        print(f"🧩 DET detecciones válidas (>= {thr:.2f}): {len(faces)}")
        return faces

    def _landmarks(self, face_bgr):
        fh, fw = face_bgr.shape[:2]
        blob = self._blob_from_bgr(face_bgr, LMK_INPUT_SIZE)  # (48,48)
        out = self.lmk_model([blob])[self.lmk_out].reshape(-1)  # 10 vals [0..1]
        pts = out.reshape(5, 2)
        pts[:, 0] *= fw
        pts[:, 1] *= fh
        return pts.astype(np.float32)

    def _align112(self, face_bgr, lmk_xy):
        M, _ = cv2.estimateAffinePartial2D(lmk_xy, ARCFACE_TEMPLATE, method=cv2.LMEDS)
        aligned = cv2.warpAffine(face_bgr, M, (REC_INPUT_SIZE[1], REC_INPUT_SIZE[0]), flags=cv2.INTER_LINEAR)
        return aligned

    def _embed(self, aligned_bgr):
        blob = self._blob_from_bgr(aligned_bgr, REC_INPUT_SIZE)  # (112,112)
        feat = self.rec_model([blob])[self.rec_out].reshape(-1).astype(np.float32)
        norm = np.linalg.norm(feat) + 1e-9
        return feat / norm

    def _compute_embedding_for_detection(self, frame_bgr, det_face: DetectedFace) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = det_face.bbox.tolist()
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        lmk = self._landmarks(roi)
        aligned = self._align112(roi, lmk)
        emb = self._embed(aligned)
        return emb

    # ----------------- Mejora/resize helpers -----------------
    def _enhance_variants(self, img_bgr):
        """Variantes de mejora suaves para ayudar al detector."""
        variants = [img_bgr]
        if not ORIG_USE_ENHANCERS:
            return variants

        # 1) CLAHE en YCrCb
        try:
            ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            y2 = clahe.apply(y)
            ycrcb2 = cv2.merge([y2, cr, cb])
            variants.append(cv2.cvtColor(ycrcb2, cv2.COLOR_YCrCb2BGR))
        except Exception:
            pass

        # 2) Sharpen suave
        try:
            kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]], dtype=np.float32)
            variants.append(cv2.filter2D(img_bgr, -1, kernel))
        except Exception:
            pass

        # 3) Gamma < 1 (levanta sombras)
        try:
            gamma = 0.8
            inv = 1.0 / max(gamma, 1e-6)
            table = (np.linspace(0, 1, 256) ** inv * 255.0).astype("uint8")
            variants.append(cv2.LUT(img_bgr, table))
        except Exception:
            pass

        return variants

    def _resize_keep(self, img_bgr, scale: float):
        if abs(scale - 1.0) < 1e-3:
            return img_bgr
        h, w = img_bgr.shape[:2]
        return cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

    # ---------- Helpers multiescala ----------
    def _clamp_box(self, x1, y1, x2, y2, W, H):
        x1 = max(0, min(x1, W-1)); y1 = max(0, min(y1, H-1))
        x2 = max(0, min(x2, W-1)); y2 = max(0, min(y2, H-1))
        if x2 <= x1: x2 = min(W-1, x1+1)
        if y2 <= y1: y2 = min(H-1, y1+1)
        return x1, y1, x2, y2

    def _render_small(self, img_bgr, canvas_mult):
        """Cara MÁS PEQUEÑA: lienzo grande + centrado, luego reescala a tamaño original."""
        H, W = img_bgr.shape[:2]
        newW, newH = int(W * canvas_mult), int(H * canvas_mult)
        canvas = np.zeros((newH, newW, 3), dtype=img_bgr.dtype)
        sx = (newW - W) // 2
        sy = (newH - H) // 2
        canvas[sy:sy+H, sx:sx+W] = img_bgr
        out = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)
        return out

    def _render_large(self, img_bgr, bbox, zoom):
        """Cara MÁS GRANDE: recorta alrededor de la cara (zoom-in) y reescala a tamaño original."""
        H, W = img_bgr.shape[:2]
        x1, y1, x2, y2 = bbox.tolist()
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        cropW = int(W / max(zoom, 1.001))
        cropH = int(H / max(zoom, 1.001))
        x1c = cx - cropW // 2
        y1c = cy - cropH // 2
        x2c = x1c + cropW
        y2c = y1c + cropH
        x1c, y1c, x2c, y2c = self._clamp_box(x1c, y1c, x2c, y2c, W, H)
        roi = img_bgr[y1c:y2c, x1c:x2c]
        if roi.size == 0:
            return img_bgr
        out = cv2.resize(roi, (W, H), interpolation=cv2.INTER_CUBIC)
        return out

    def _render_medium(self, img_bgr):
        """Cara tamaño MEDIO: tal cual (puedes añadir jitter leve si quieres)."""
        return img_bgr

    # ----------------- Conteo por buckets (por persona) -----------------
    def _count_person_bucket_embeddings(self, person_dir: str) -> Dict[str, int]:
        """
        Cuenta embeddings existentes por bucket en la carpeta de la persona,
        buscando nombres *_aug_small_*, *_aug_medium_* y *_aug_large_*.
        """
        counts = {b: 0 for b in BUCKETS}
        if not os.path.isdir(person_dir):
            return counts
        for fn in os.listdir(person_dir):
            if not fn.endswith(".npy"):
                continue
            lower = fn.lower()
            # respeta prefijo común "_aug_" para compatibilidad con check_if_image_processed()
            if "_aug_small_" in lower:
                counts["small"] += 1
            elif "_aug_medium_" in lower:
                counts["medium"] += 1
            elif "_aug_large_" in lower:
                counts["large"] += 1
        return counts

    def _bucket_targets_for_person(self) -> Dict[str, int]:
        """
        Objetivo fijo por persona: 1/3 cada bucket. Si TARGET no es múltiplo de 3,
        el resto (1) se asigna a 'medium' para llegar al total.
        """
        base = TARGET_EMB_PER_PERSON // 3
        rem = TARGET_EMB_PER_PERSON % 3
        targets = {"small": base, "medium": base, "large": base}
        if rem > 0:
            targets["medium"] += rem
        return targets

    def _images_left_for_person(self, person_name: str) -> int:
        """Cuenta cuántas imágenes originales quedan en INPUT_DIR/<persona>."""
        person_image_dir = os.path.join(INPUT_DIR, person_name)
        try:
            original_files = [f for f in os.listdir(person_image_dir)
                              if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            return max(1, len(original_files))
        except Exception:
            return 1

    def _per_image_bucket_plan(
        self, person_dir: str, person_name: str
    ) -> Dict[str, int]:
        """
        Calcula cuántos augs generar para ESTA imagen por bucket,
        de manera que al final, por persona, se cumpla:
        small=333, medium=333(+1 si total=1000), large=333.
        No forzamos mínimos por imagen para no sobreproducir; sí
        limitamos por arriba con AUGS_PER_IMAGE_MAX.
        """
        # objetivos y conteo actual
        targets = self._bucket_targets_for_person()
        current = self._count_person_bucket_embeddings(person_dir)
        remaining = {b: max(0, targets[b] - current.get(b, 0)) for b in BUCKETS}

        # si ya llegó al objetivo total, nada que hacer
        if sum(remaining.values()) == 0:
            return {b: 0 for b in BUCKETS}

        imgs_left = max(1, self._images_left_for_person(person_name))

        # reparto por bucket: ceil(remaining / imgs_left)
        plan = {b: int(np.ceil(remaining[b] / imgs_left)) for b in BUCKETS}

        # topes por imagen (no forzamos mínimos para evitar pasarnos del objetivo global)
        total_plan = sum(plan.values())
        if AUGS_PER_IMAGE_MAX > 0 and total_plan > AUGS_PER_IMAGE_MAX:
            # reduce proporcionalmente para no pasar el tope por imagen
            factor = AUGS_PER_IMAGE_MAX / max(1, total_plan)
            plan = {b: max(0, int(np.floor(plan[b] * factor))) for b in BUCKETS}
            # si por redondeo quedó 0 pero hay remanente, rellena en orden medium->small->large
            while sum(plan.values()) < AUGS_PER_IMAGE_MAX and any(remaining[b] > 0 for b in BUCKETS):
                for b in ("medium", "small", "large"):
                    if sum(plan.values()) >= AUGS_PER_IMAGE_MAX:
                        break
                    if plan[b] < remaining[b]:
                        plan[b] += 1

        return plan

    # ----------------- Detección original con reintentos -----------------
    def detect_faces_in_original(self, img, filename):
        """
        Intenta detectar caras en la imagen original con reintentos:
        - distintas escalas (ORIG_SCALES)
        - mejoras suaves (CLAHE, sharpen, gamma) si ORIG_USE_ENHANCERS=True
        Muestra preview cada vez que haya detecciones válidas (si SHOW_PREVIEW=True).
        """
        print("🔍 Probando detección en imagen original (con reintentos)...")
        base_bgr = self._to_bgr(img)

        attempts = 0
        best_faces = None
        best_info = None

        variants = self._enhance_variants(base_bgr)

        for v_idx, variant in enumerate(variants):
            for _s_idx, scale in enumerate(ORIG_SCALES):
                attempts += 1
                if attempts > ORIG_DET_RETRIES:
                    break

                test_img = self._resize_keep(variant, scale)
                t0 = time.time()
                faces = self._detect(test_img)
                elapsed = (time.time() - t0) * 1000

                print(f"   ▶️ intento {attempts}/{ORIG_DET_RETRIES} | var:{v_idx+1}/{len(variants)} "
                      f"| esc:{scale} | detecciones:{len(faces)} | {elapsed:.1f} ms")

                if faces:
                    faces.sort(key=lambda f: f.det_score, reverse=True)
                    valid = [f for f in faces if f.det_score >= float(MIN_DET_SCORE)]
                    if valid:
                        best_faces = valid
                        best_info = (v_idx, scale, elapsed)
                        if SHOW_PREVIEW and ORIG_PREVIEW_EACH_HIT:
                            show_preview(test_img, valid, f"Original OK - {filename}", 1000)
                        break
            if attempts > ORIG_DET_RETRIES or best_faces is not None:
                break

        if best_faces:
            v_idx, scale, elapsed = best_info
            print(f"✅ Detección lograda en intento {attempts} (var:{v_idx+1}, esc:{scale}) "
                  f"con {len(best_faces)} cara(s) válidas")
            return best_faces
        else:
            print("⚠️ No se detectaron caras en la imagen original tras reintentos")
            if SHOW_PREVIEW:
                show_preview(base_bgr, [], f"Original (sin caras) - {filename}", 1000)
            return None

    # ----------------- Augmentations + embeddings (con buckets exactos) -----------------
    def _render_by_bucket(self, bucket: str, base_bgr: np.ndarray, base_bbox: np.ndarray) -> np.ndarray:
        if bucket == "small":
            return self._render_small(base_bgr, SMALL_CANVAS_MULT)
        elif bucket == "large":
            return self._render_large(base_bgr, base_bbox, LARGE_ZOOM)
        else:
            return self._render_medium(base_bgr)

    def _try_make_embedding(self, image_bgr: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[List[DetectedFace]]]:
        faces = self._detect(image_bgr)
        if faces:
            faces.sort(key=lambda f: f.det_score, reverse=True)
            face = faces[0]
            if face.det_score >= float(MIN_DET_SCORE):
                emb = self._compute_embedding_for_detection(image_bgr, face)
                if emb is not None:
                    return True, emb, faces
        return False, None, None

    def process_augmentations(
        self, img, filename, person_name, base_name, person_dir, per_image_plan: Dict[str, int]
    ) -> int:
        """
        Genera exactamente la cantidad pedida por bucket para ESTA imagen:
        per_image_plan = {'small': n1, 'medium': n2, 'large': n3}
        """
        total_goal = sum(per_image_plan.values())
        if total_goal <= 0:
            print("ℹ️ No hay augs planificados para esta imagen (objetivo por persona alcanzado).")
            return 0

        # BGR base y bbox base (para large)
        bgr_base = self._to_bgr(img)
        base_faces = self._detect(bgr_base)
        if base_faces:
            base_faces.sort(key=lambda f: f.det_score, reverse=True)
            base_bbox = base_faces[0].bbox
        else:
            H, W = bgr_base.shape[:2]
            base_bbox = np.array([W//4, H//4, 3*W//4, 3*H//4], dtype=np.int32)

        print(f"🎛️ Plan por buckets (esta imagen) → small:{per_image_plan['small']}  "
              f"medium:{per_image_plan['medium']}  large:{per_image_plan['large']}  (total={total_goal})")

        successes = 0

        for bucket in BUCKETS:
            target = int(per_image_plan.get(bucket, 0))
            made = 0

            # buscar un índice incremental para nombres por bucket
            # (no es estrictamente necesario que sea correlativo, pero ayuda a ordenar)
            next_idx = 0
            # escanea nombres existentes para este (persona, imagen, bucket)
            prefix = f"{person_name}_{base_name}_aug_{bucket}_"
            existing = [f for f in os.listdir(person_dir) if f.startswith(prefix) and f.endswith(".npy")]
            if existing:
                # busca el mayor índice ya usado
                try:
                    nums = []
                    for fn in existing:
                        tail = fn[len(prefix):-4]  # quita prefijo y .npy
                        if tail.isdigit():
                            nums.append(int(tail))
                    if nums:
                        next_idx = max(nums) + 1
                except Exception:
                    pass

            while made < target:
                save_path = os.path.join(person_dir, f"{prefix}{next_idx}.npy")
                next_idx += 1
                if os.path.exists(save_path):
                    print(f"✅ Ya existía {os.path.basename(save_path)}, saltando")
                    successes += 1
                    made += 1
                    continue

                # 1) Render multiescala de acuerdo al bucket
                rendered = self._render_by_bucket(bucket, bgr_base, base_bbox)

                # 2) Augmentación suave
                transform_factory = random.choice(self.transform_options)
                aug_img = transform_factory()(image=rendered)['image']

                # 3) Detectar + embeder con reintentos
                ok, emb, faces = self._try_make_embedding(aug_img)
                attempts = 1
                while (not ok) and attempts < MAX_ATTEMPTS:
                    attempts += 1
                    transform_factory = random.choice(self.transform_options)
                    aug_img = transform_factory()(image=rendered)['image']
                    ok, emb, faces = self._try_make_embedding(aug_img)

                if ok and emb is not None:
                    np.save(save_path, emb)
                    successes += 1
                    made += 1
                    if SHOW_PREVIEW and (successes % PREVIEW_INTERVAL == 0) and faces is not None:
                        show_preview(aug_img, faces, f"{bucket} #{made} - {filename}", 800)
                    print(f"✅ [{bucket}] {made}/{target}  (global {successes}/{total_goal})")
                else:
                    print(f"⚠️ [{bucket}] no se pudo generar (tras {MAX_ATTEMPTS} intentos).")
                    # si no pudimos, sal del bucket para evitar bucles infinitos
                    break

        return successes

    # ----------------- Proceso completo de una imagen -----------------
    def process_single_image(self, img_path, filename, output_dir, max_width, max_height):
        """Procesa una imagen: detección original + embedding + augmentations por bucket exactos."""
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ No se pudo leer {img_path}")
            return False

        print(f"\n📸 Procesando: {filename}")
        img, size, was_resized = resize_image_if_needed(img, max_width, max_height)
        if was_resized:
            print(f"🔄 Redimensionada: {filename} → {size}")

        # Detección en original con reintentos
        faces = self.detect_faces_in_original(img, filename)
        if not faces:
            return False

        # Nombres y carpetas
        base_name = os.path.splitext(filename)[0]
        root = os.path.dirname(img_path)
        rel_dir = os.path.relpath(root, INPUT_DIR)
        person_name = rel_dir.split(os.sep)[0]
        print(f"📂 Person Name: {person_name}")

        person_dir = os.path.join(OUTPUT_DIR, person_name)
        if os.path.isdir(person_dir):
            print(f"✅ Carpeta de embeddings existe para '{person_name}', revisando imágenes...")
        else:
            person_dir = create_output_directory(person_name, OUTPUT_DIR)

        # Guardar embedding ORIGINAL (no cuenta a ningún bucket)
        emb_name = f"{person_name}_{base_name}_original.npy"
        save_path = os.path.join(person_dir, emb_name)
        if not os.path.exists(save_path):
            faces.sort(key=lambda f: f.det_score, reverse=True)
            top_face = faces[0]
            bgr = self._to_bgr(img)
            emb = self._compute_embedding_for_detection(bgr, top_face)
            if emb is not None:
                np.save(save_path, emb)
                print("✅ Original embedding guardado")
            else:
                print("⚠️ No se pudo generar embedding del original (landmarks/align)")
                return False
        else:
            print("✅ Original embedding ya existía, saltando")

        # ====== Plan por buckets solo para ESTA imagen ======
        per_image_plan = self._per_image_bucket_plan(person_dir, person_name)
        total_plan = sum(per_image_plan.values())
        if total_plan == 0:
            print(f"🎯 '{person_name}' ya alcanzó su objetivo por persona ({TARGET_EMB_PER_PERSON}).")
        else:
            print(f"🎯 Plan por imagen: {per_image_plan} (total {total_plan})")

        # Evita usar check_if_image_processed con N_AUGMENTATIONS fijo; aquí el objetivo por imagen es total_plan
        successful_augmentations = self.process_augmentations(
            img, filename, person_name, base_name, person_dir, per_image_plan=per_image_plan
        )

        print(f"📊 Resumen {filename}: {successful_augmentations}/{total_plan} embeddings OK")

        # === ELIMINAR IMAGEN FUENTE SEGÚN CONFIG ===
        from main.training.utils import safe_delete_file, remove_dir_if_empty  # utilidades de borrado
        if DELETE_SOURCE_IMAGES and os.path.exists(img_path):
            cond_ok = (successful_augmentations >= total_plan) if DELETE_REQUIRE_FULL_AUGS else True
            if cond_ok:
                ok = safe_delete_file(img_path, move_to_trash=MOVE_TO_TRASH)  # False => permanente
                if ok:
                    parent_dir = os.path.dirname(img_path)
                    remove_dir_if_empty(parent_dir)
            else:
                print("ℹ️ No se eliminará la imagen fuente porque no se completaron todos los augs planificados.")

        return True
