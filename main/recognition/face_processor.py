"""
Face processing logic - extends the face recognizer with processing capabilities
"""

import cv2
import numpy as np
import logging
from collections import deque

from config import *
from image_processor import *

logger = logging.getLogger(__name__)


class FaceProcessor:
    """Handles face processing logic for the OptimizedFaceRecognizer"""
    
    def __init__(self, recognizer):
        self.recognizer = recognizer
    
    def process_frame(self, frame):
        """Procesamiento principal del frame con optimizaciones para larga distancia"""
        processed_frame = preprocess_frame(frame)
        
        try:
            faces = self.recognizer.app.get(processed_frame)
            results = []
            
            for i, face in enumerate(faces):
                # Filtrar por confianza
                if face.det_score < MIN_CONFIDENCE:
                    continue
                
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                face_width = x2 - x1
                face_height = y2 - y1
                face_size = min(face_width, face_height)
                
                if face_size < MIN_FACE_SIZE:
                    continue
                
                # Mejorar región facial para caras pequeñas
                if face_size < BASE_FACE_SIZE:
                    processed_frame = enhance_face_region(processed_frame, (x1, y1, x2, y2))
                    # Redetectar con imagen mejorada
                    enhanced_faces = self.recognizer.app.get(processed_frame[y1:y2, x1:x2])
                    if enhanced_faces:
                        face = enhanced_faces[0]  # Usar cara mejorada
                
                # Identificar cara con threshold adaptativo
                adaptive_threshold = self.recognizer.get_adaptive_threshold(face_size)
                identity, confidence = self.recognizer.identify_face_optimized(face.embedding, adaptive_threshold)
                
                # Aplicar suavizado temporal
                face_id = f"{x1}_{y1}_{x2}_{y2}"
                identity, confidence = self.recognizer.apply_temporal_smoothing(face_id, identity, confidence)
                
                # Penalizar confianza para caras muy pequeñas
                confidence = apply_confidence_penalty(confidence, face_size)
                
                results.append({
                    'bbox': (x1, y1, x2, y2),
                    'identity': identity,
                    'confidence': confidence,
                    'det_score': face.det_score,
                    'face_size': face_size
                })
                
                # Actualizar estadísticas
                self.recognizer.stats['faces_detected'] += 1
                if identity != "Desconocido":
                    self.recognizer.stats['faces_recognized'] += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Error procesando frame: {e}")
            return []
    
    def draw_results(self, frame, results):
        """Dibuja resultados en el frame con información de distancia"""
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            identity = result['identity']
            confidence = result['confidence']
            det_score = result['det_score']
            face_size = result.get('face_size', 0)

            # Determinar color basado en identidad y confianza
            if identity != "Desconocido":
                distance_est = get_distance_estimation(face_size)

                # Lógica de color según confianza y tamaño del rostro
                if confidence > 85 and face_size > 80:
                    color = (0, 255, 0)  # Verde fuerte - alta confianza, cerca
                elif confidence > 70:
                    color = (0, 255, 255)  # Amarillo
                elif confidence > 50:
                    color = (0, 165, 255)  # Naranja
                else:
                    color = (0, 0, 255)  # Rojo

                label = f"{identity} ({confidence:.1f}%) - {distance_est}"
            else:
                color = (255, 0, 0)  # Azul para desconocidos
                label = f"Desconocido (det: {det_score:.2f}) - Size: {face_size}px"

            # Grosor de línea basado en confianza
            thickness = 3 if confidence > 80 else 2

            # Dibujar rectángulo
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Fondo para la etiqueta
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)[0]
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)

        # Mostrar estadísticas generales
        stats_text = f"Frames: {self.recognizer.stats['processed_frames']}/{self.recognizer.stats['total_frames']} | "
        stats_text += f"Caras: {self.recognizer.stats['faces_detected']} | "
        stats_text += f"Reconocidas: {self.recognizer.stats['faces_recognized']}"
        cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), 1)

        return frame