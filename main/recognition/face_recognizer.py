"""
Optimized Face Recognition Class for long-distance recognition
"""

import cv2
import numpy as np
import os
import logging
from collections import defaultdict, deque
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis
import faiss

from config import *

logger = logging.getLogger(__name__)


class OptimizedFaceRecognizer:
    def __init__(self):
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=CTX_ID, det_size=DETECTION_SIZE)
        
        # Estructuras de datos optimizadas
        self.known_faces = {}
        self.face_embeddings = []
        self.face_labels = []
        self.face_counts = defaultdict(int)
        self.index = None
        
        # Suavizado temporal
        self.temporal_buffer = defaultdict(lambda: deque(maxlen=TEMPORAL_WINDOW))
        
        # Métricas de rendimiento
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'faces_detected': 0,
            'faces_recognized': 0
        }
        
        self.load_embeddings()
        self.build_index()
    
    def load_embeddings(self):
        """Carga embeddings desde el directorio de embeddings"""
        if not os.path.exists(EMBEDDING_DIR):
            logger.error(f"Directorio {EMBEDDING_DIR} no existe")
            return

        logger.info("Cargando embeddings...")
    
        for root, dirs, files in os.walk(EMBEDDING_DIR):
            for file in files:
                if file.endswith('.npy'):
                    try:
                        # Obtener nombre de la persona desde el subdirectorio
                        person_name = os.path.basename(root)
                        emb_path = os.path.join(root, file)
                        emb = np.load(emb_path)

                        # Validar embedding
                        if self.validate_embedding(emb):
                            self.known_faces.setdefault(person_name, []).append(emb)
                            self.face_embeddings.append(emb)
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
        """Valida calidad del embedding"""
        if embedding is None or len(embedding) == 0:
            return False
        
        # Verificar que no sea un vector nulo
        norm = np.linalg.norm(embedding)
        if norm < 0.1:
            return False
        
        # Verificar dimensionalidad
        if len(embedding) != 512:  # Buffalo_l usa 512 dimensiones
            return False
        
        return True
    
    def build_index(self):
        """Construye índice FAISS para búsqueda rápida"""
        if len(self.face_embeddings) == 0:
            logger.warning("No hay embeddings para construir índice")
            return
        
        logger.info("Construyendo índice FAISS...")
        embeddings_array = np.array(self.face_embeddings).astype('float32')
        
        # Normalizar embeddings para mejor rendimiento con cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Usar índice de producto interno (equivalente a cosine con vectores normalizados)
        self.index = faiss.IndexFlatIP(embeddings_array.shape[1])
        self.index.add(embeddings_array)
        
        logger.info(f"Índice construido con {self.index.ntotal} embeddings")
    
    def identify_face_optimized(self, embedding, adaptive_threshold=None):
        """Identificación optimizada con múltiples validaciones"""
        if self.index is None or len(self.face_embeddings) == 0:
            return "Desconocido", 0.0
        
        # Usar threshold adaptativo si se proporciona
        current_threshold = adaptive_threshold if adaptive_threshold else THRESHOLD
        
        # Normalizar embedding de consulta
        query_emb = embedding.copy().astype('float32')
        faiss.normalize_L2(query_emb.reshape(1, -1))
        
        # Buscar top-k candidatos
        k = min(TOP_K_CANDIDATES, len(self.face_embeddings))
        scores, indices = self.index.search(query_emb.reshape(1, -1), k)
        
        best_match = ("Desconocido", 0.0)
        candidates = defaultdict(list)
        
        # Agrupar candidatos por persona
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score > 0:  # Solo scores positivos
                label = self.face_labels[idx]
                similarity = float(score) * 100  # FAISS devuelve producto interno
                candidates[label].append(similarity)
        
        # Encontrar mejor match usando estadísticas mejoradas
        for label, similarities in candidates.items():
            if len(similarities) == 0:
                continue
                
            # Estadísticas robustas
            avg_sim = np.mean(similarities)
            max_sim = np.max(similarities)
            std_sim = np.std(similarities) if len(similarities) > 1 else 0
            count = len(similarities)
            
            # Filtro de consistencia
            consistency_threshold = (CONSISTENCY_THRESHOLD_PERMISSIVE 
                                   if adaptive_threshold and adaptive_threshold > THRESHOLD 
                                   else CONSISTENCY_THRESHOLD_NORMAL)
            if std_sim > consistency_threshold and count > 3:
                continue
            
            # Puntuación combinada mejorada
            consistency_bonus = max(0, 1 - std_sim/25)
            count_bonus = min(0.3, count * 0.05)
            
            combined_score = (avg_sim * 0.5 + max_sim * 0.5) * (1 + consistency_bonus + count_bonus)
            
            # Threshold dinámico basado en cantidad de embeddings
            dynamic_threshold = (100 - current_threshold * 100) - (count * DYNAMIC_THRESHOLD_REDUCTION)
            
            if combined_score > best_match[1] and max_sim > dynamic_threshold:
                best_match = (label, combined_score)
        
        return best_match
    
    def apply_temporal_smoothing(self, face_id, identity, confidence):
        """Suavizado temporal mejorado para reducir fluctuaciones"""
        self.temporal_buffer[face_id].append((identity, confidence))
        
        if len(self.temporal_buffer[face_id]) < 3:
            return identity, confidence
        
        # Análisis de consistencia temporal
        recent_data = list(self.temporal_buffer[face_id])
        
        # Identidad más frecuente
        identity_counts = defaultdict(int)
        confidence_sums = defaultdict(list)
        
        for ident, conf in recent_data:
            identity_counts[ident] += 1
            confidence_sums[ident].append(conf)
        
        # Encontrar identidad dominante
        most_common = max(identity_counts.items(), key=lambda x: x[1])
        
        # Requerir mayor consistencia para identidades conocidas
        min_required = (MIN_TEMPORAL_CONSISTENCY if most_common[0] != "Desconocido" 
                       else MIN_TEMPORAL_CONSISTENCY_UNKNOWN)
        
        if most_common[1] >= min_required:
            # Usar confianza promedio ponderada
            relevant_confidences = confidence_sums[most_common[0]]
            weights = np.linspace(0.5, 1.0, len(relevant_confidences))
            weighted_confidence = np.average(relevant_confidences, weights=weights)
            return most_common[0], weighted_confidence
        
        return identity, confidence
    
    def get_adaptive_threshold(self, face_size):
        """Threshold adaptativo basado en tamaño de cara"""
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