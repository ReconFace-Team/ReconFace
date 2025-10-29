"""
Image processing functions for face recognition optimization
"""

import cv2
import numpy as np
import logging

from config import *

logger = logging.getLogger(__name__)


def preprocess_frame(frame):
    """Preprocesamiento optimizado para larga distancia"""
    # Ajusta orden de color según config
    if EXPECTED_COLOR_ORDER.upper() == "RGB":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Si BGR, no convertimos

    # Mejora de contraste/nitidez idéntica a la tuya
    frame = cv2.convertScaleAbs(frame, alpha=CONTRAST_ALPHA, beta=BRIGHTNESS_BETA)
    frame = cv2.bilateralFilter(frame, BILATERAL_FILTER_D,
                                BILATERAL_FILTER_SIGMA_COLOR,
                                BILATERAL_FILTER_SIGMA_SPACE)
    if ENHANCED_PREPROCESSING:
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        frame = cv2.filter2D(frame, -1, kernel)
    return frame



def enhance_face_region(frame, bbox):
    """Mejora la región facial para larga distancia"""
    x1, y1, x2, y2 = bbox
    
    # Extraer región facial con padding
    padding = 20
    x1_pad = max(0, x1 - padding)
    y1_pad = max(0, y1 - padding)
    x2_pad = min(frame.shape[1], x2 + padding)
    y2_pad = min(frame.shape[0], y2 + padding)
    
    face_region = frame[y1_pad:y2_pad, x1_pad:x2_pad]
    
    if face_region.size == 0:
        return frame
    
    # Super-resolución simple (interpolación bicúbica)
    if ENABLE_SUPER_RESOLUTION:
        face_width = x2_pad - x1_pad
        face_height = y2_pad - y1_pad
        
        if face_width < SUPER_RESOLUTION_THRESHOLD or face_height < SUPER_RESOLUTION_THRESHOLD:
            scale_factor = SUPER_RESOLUTION_SCALE
            new_width = int(face_width * scale_factor)
            new_height = int(face_height * scale_factor)
            
            enhanced_face = cv2.resize(face_region, (new_width, new_height), 
                                     interpolation=cv2.INTER_CUBIC)
            
            # Sharpening adicional
            kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
            enhanced_face = cv2.filter2D(enhanced_face, -1, kernel)
            
            # Redimensionar de vuelta
            enhanced_face = cv2.resize(enhanced_face, (face_width, face_height), 
                                     interpolation=cv2.INTER_CUBIC)
            
            # Reemplazar en frame original
            frame[y1_pad:y2_pad, x1_pad:x2_pad] = enhanced_face
    
    return frame


def get_color_by_confidence(identity, confidence, face_size):
    """Determina el color basado en confianza y distancia"""
    if identity != "Desconocido":
        if confidence > HIGH_CONFIDENCE_THRESHOLD and face_size > BASE_FACE_SIZE:
            return COLOR_HIGH_CONFIDENCE  # Verde fuerte - alta confianza, cerca
        elif confidence > MEDIUM_CONFIDENCE_THRESHOLD:
            return COLOR_MEDIUM_CONFIDENCE  # Amarillo - confianza media
        elif confidence > LOW_CONFIDENCE_THRESHOLD:
            return COLOR_LOW_CONFIDENCE  # Naranja - baja confianza
        else:
            return COLOR_VERY_LOW_CONFIDENCE  # Naranja oscuro - muy baja confianza
    else:
        return COLOR_UNKNOWN  # Rojo


def get_distance_estimation(face_size):
    """Estima la distancia basada en el tamaño de la cara"""
    if face_size > BASE_FACE_SIZE:
        return "Cerca"
    elif face_size > MEDIUM_FACE_SIZE:
        return "Medio"
    else:
        return "Lejos"


def apply_confidence_penalty(confidence, face_size):
    """Aplica penalización por distancia a la confianza"""
    if face_size < MEDIUM_FACE_SIZE:
        return confidence * 0.8  # Reducir confianza 20%
    elif face_size < BASE_FACE_SIZE:
        return confidence * 0.9  # Reducir confianza 10%
    return confidence