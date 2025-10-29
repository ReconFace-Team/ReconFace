# face_recognizer.py
import os
import cv2
import faiss
import numpy as np
import logging
from dataclasses import dataclass
from collections import defaultdict, deque
from openvino.runtime import Core

from config import *
logger = logging.getLogger(__name__)

# Plantilla de 5 puntos para alineación ArcFace 112x112
ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963],
     [73.5318, 51.5014],
     [56.0252, 71.7366],
     [41.5493, 92.3655],
     [70.7299, 92.2041]], dtype=np.float32
)

@dataclass
class DetectedFace:
    bbox: tuple          # (x1, y1, x2, y2) en pixeles
    det_score: float     # confianza del detector
    landmarks: np.ndarray  # (5,2) en pixeles relativos al recorte
    embedding: np.ndarray  # (512,)

class OptimizedFaceRecognizer:
    def __init__(self):
        # ---- OpenVINO: carga y compila modelos ----
        core = Core()
        compile_cfg = {"PERFORMANCE_HINT": OV_PERFORMANCE_HINT} if OV_PERFORMANCE_HINT else {}
        self.det_model = core.compile_model(OV_DET_MODEL, OV_DEVICE, compile_cfg)
        self.lmk_model = core.compile_model(OV_LMK_MODEL, OV_DEVICE, compile_cfg)
        self.rec_model = core.compile_model(OV_REC_MODEL, OV_DEVICE, compile_cfg)

        self.det_in = self.det_model.inputs[0]
        self.det_out = self.det_model.outputs[0]
        self.lmk_in = self.lmk_model.inputs[0]
        self.lmk_out = self.lmk_model.outputs[0]
        self.rec_in = self.rec_model.inputs[0]
        self.rec_out = self.rec_model.outputs[0]

        # ---- Estructuras de embeddings/FAISS ----
        self.known_faces = {}
        self.face_embeddings = []
        self.face_labels = []
        self.face_counts = defaultdict(int)
        self.index = None

        # ---- Suavizado temporal ----
        self.temporal_buffer = defaultdict(lambda: deque(maxlen=TEMPORAL_WINDOW))

        # ---- Métricas ----
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'faces_detected': 0,
            'faces_recognized': 0
        }

        self.load_embeddings()
        self.build_index()


    def _preprocess_bgr(self, img, size_hw):
        h, w = size_hw
        blob = cv2.resize(img, (w, h))
        # Modelos OMZ IR esperan BGR, NCHW, float32
        blob = blob.transpose(2, 0, 1)[None].astype(np.float32)
        return blob

    def _detect(self, frame_bgr):
        """Devuelve lista de (x1,y1,x2,y2,score) en píxeles."""
        H, W = frame_bgr.shape[:2]
        blob = self._preprocess_bgr(frame_bgr, (256, 256))
        out = self.det_model([blob])[self.det_out]  # shape [1,1,200,7]
        detections = out[0, 0, :, :]
        faces = []
        for det in detections:
            conf = float(det[2])
            if conf < DET_MIN_CONFIDENCE:
                continue
            x1 = max(0, int(det[3] * W))
            y1 = max(0, int(det[4] * H))
            x2 = min(W - 1, int(det[5] * W))
            y2 = min(H - 1, int(det[6] * H))
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2, y2, conf))
        return faces

    def _landmarks(self, face_bgr):
        """Devuelve (5,2) en píxeles relativos al recorte."""
        fh, fw = face_bgr.shape[:2]
        blob = self._preprocess_bgr(face_bgr, LMK_INPUT_SIZE[::-1])  # (48,48)
        out = self.lmk_model([blob])[self.lmk_out].reshape(-1)  # 10 vals normalizados [0..1]
        pts = out.reshape(5, 2)
        pts[:, 0] *= fw
        pts[:, 1] *= fh
        return pts.astype(np.float32)

    def _align112(self, face_bgr, lmk_xy):
        """Alinea a 112x112 usando 5 puntos."""
        dst = ARCFACE_TEMPLATE.copy()
        # Estima transformación afín parcial desde landmarks -> plantilla
        M, _ = cv2.estimateAffinePartial2D(lmk_xy, dst, method=cv2.LMEDS)
        aligned = cv2.warpAffine(face_bgr, M, REC_INPUT_SIZE, flags=cv2.INTER_LINEAR)
        return aligned

    def _embed(self, aligned_bgr):
        """Devuelve embedding 512D normalizado L2."""
        blob = self._preprocess_bgr(aligned_bgr, REC_INPUT_SIZE[::-1])  # (112,112)
        feat = self.rec_model([blob])[self.rec_out].reshape(-1).astype(np.float32)
        # Normaliza L2 (ArcFace IR reporta comparabilidad por coseno) 
        # Doc del modelo indica salida 1x512 comparable en distancia coseno. :contentReference[oaicite:4]{index=4}
        norm = np.linalg.norm(feat) + 1e-9
        return feat / norm

    def get_faces(self, frame_bgr):
        """Devuelve lista de DetectedFace con bbox, score, landmarks, embedding."""
        faces_raw = self._detect(frame_bgr)
        results = []
        for (x1, y1, x2, y2, score) in faces_raw:
            face_roi = frame_bgr[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue
            lmk = self._landmarks(face_roi)
            aligned = self._align112(face_roi, lmk)
            emb = self._embed(aligned)
            results.append(DetectedFace(
                bbox=(x1, y1, x2, y2),
                det_score=score,
                landmarks=lmk,
                embedding=emb
            ))
        return results


    def load_embeddings(self):
        """Carga embeddings .npy desde EMBEDDING_DIR (deben ser 512D)."""
        if not os.path.exists(EMBEDDING_DIR):
            logger.error(f"Directorio {EMBEDDING_DIR} no existe")
            return

        logger.info("Cargando embeddings...")
        for root, _, files in os.walk(EMBEDDING_DIR):
            for file in files:
                if file.endswith('.npy'):
                    try:
                        person_name = os.path.basename(root)
                        emb_path = os.path.join(root, file)
                        emb = np.load(emb_path)
                        if self.validate_embedding(emb):
                            self.known_faces.setdefault(person_name, []).append(emb)
                            self.face_embeddings.append(emb.astype(np.float32))
                            self.face_labels.append(person_name)
                            self.face_counts[person_name] += 1
                        else:
                            logger.warning(f"Embedding inválido: {file}")
                    except Exception as e:
                        logger.error(f"Error cargando {file}: {e}")

        logger.info(f"Cargados {len(self.face_embeddings)} embeddings de {len(self.face_counts)} personas")
        for name, count in self.face_counts.items():
            logger.info(f"  {name}: {count} embeddings")

    def validate_embedding(self, embedding):
        if embedding is None or len(embedding) == 0:
            return False
        if len(embedding) != 512:
            return False
        if np.linalg.norm(embedding) < 0.1:
            return False
        return True

    def build_index(self):
        if len(self.face_embeddings) == 0:
            logger.warning("No hay embeddings para construir índice")
            return
        logger.info("Construyendo índice FAISS...")
        embeddings_array = np.array(self.face_embeddings).astype('float32')
        faiss.normalize_L2(embeddings_array)
        self.index = faiss.IndexFlatIP(embeddings_array.shape[1])
        self.index.add(embeddings_array)
        logger.info(f"Índice construido con {self.index.ntotal} embeddings")

    def identify_face_optimized(self, embedding, adaptive_threshold=None):
        if self.index is None or len(self.face_embeddings) == 0:
            return "Desconocido", 0.0

        current_threshold = adaptive_threshold if adaptive_threshold else THRESHOLD
        query = embedding.astype('float32').copy()
        faiss.normalize_L2(query.reshape(1, -1))
        k = min(TOP_K_CANDIDATES, len(self.face_embeddings))
        scores, indices = self.index.search(query.reshape(1, -1), k)

        best_match = ("Desconocido", 0.0)
        candidates = defaultdict(list)

        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score > 0:
                label = self.face_labels[idx]
                similarity = float(score) * 100  # producto interno -> % coseno
                candidates[label].append(similarity)

        for label, similarities in candidates.items():
            if len(similarities) == 0:
                continue
            avg_sim = np.mean(similarities)
            max_sim = np.max(similarities)
            std_sim = np.std(similarities) if len(similarities) > 1 else 0
            count = len(similarities)

            consistency_threshold = (CONSISTENCY_THRESHOLD_PERMISSIVE 
                                     if adaptive_threshold and adaptive_threshold > THRESHOLD
                                     else CONSISTENCY_THRESHOLD_NORMAL)
            if std_sim > consistency_threshold and count > 3:
                continue

            consistency_bonus = max(0, 1 - std_sim/25)
            count_bonus = min(0.3, count * 0.05)
            combined_score = (avg_sim * 0.5 + max_sim * 0.5) * (1 + consistency_bonus + count_bonus)

            dynamic_threshold = (100 - current_threshold * 100) - (count * DYNAMIC_THRESHOLD_REDUCTION)
            if combined_score > best_match[1] and max_sim > dynamic_threshold:
                best_match = (label, combined_score)

        return best_match

    def apply_temporal_smoothing(self, face_id, identity, confidence):
        self.temporal_buffer[face_id].append((identity, confidence))
        if len(self.temporal_buffer[face_id]) < 3:
            return identity, confidence

        recent_data = list(self.temporal_buffer[face_id])
        identity_counts = defaultdict(int)
        confidence_sums = defaultdict(list)
        for ident, conf in recent_data:
            identity_counts[ident] += 1
            confidence_sums[ident].append(conf)

        most_common = max(identity_counts.items(), key=lambda x: x[1])
        min_required = (MIN_TEMPORAL_CONSISTENCY if most_common[0] != "Desconocido"
                        else MIN_TEMPORAL_CONSISTENCY_UNKNOWN)

        if most_common[1] >= min_required:
            relevant_confidences = confidence_sums[most_common[0]]
            weights = np.linspace(0.5, 1.0, len(relevant_confidences))
            weighted_confidence = np.average(relevant_confidences, weights=weights)
            return most_common[0], weighted_confidence
        return identity, confidence

    def get_adaptive_threshold(self, face_size):
        if not DISTANCE_ADAPTIVE_THRESHOLD:
            return THRESHOLD
        if face_size >= BASE_FACE_SIZE:
            return THRESHOLD
        elif face_size >= MEDIUM_FACE_SIZE:
            return THRESHOLD + THRESHOLD_ADJUSTMENT_MEDIUM
        elif face_size >= SMALL_FACE_SIZE:
            return THRESHOLD + THRESHOLD_ADJUSTMENT_SMALL
        else:
            return THRESHOLD + THRESHOLD_ADJUSTMENT_VERY_SMALL
