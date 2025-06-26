# Sistema de Reconocimiento Facial Optimizado

Sistema de reconocimiento facial en tiempo real optimizado para detección a larga distancia con múltiples mejoras de rendimiento y precisión.

## Características Principales

- **Reconocimiento a larga distancia**: Optimizado para detectar caras pequeñas y lejanas
- **Procesamiento en tiempo real**: Optimizaciones de rendimiento para video en vivo
- **Threshold adaptativo**: Ajuste automático de umbrales según el tamaño de la cara
- **Suavizado temporal**: Reduce flickering en las identificaciones
- **Super-resolución**: Mejora automática de caras pequeñas
- **Índice FAISS**: Búsqueda rápida en grandes conjuntos de embeddings
- **Métricas de rendimiento**: Monitoreo en tiempo real del sistema

## Estructura del Proyecto

```
face_recognition_system/
├── main.py                 # Aplicación principal
├── config.py               # Configuración del sistema
├── face_recognizer.py      # Clase principal de reconocimiento
├── face_processor.py       # Lógica de procesamiento de caras
├── image_processor.py      # Funciones de procesamiento de imagen
├── camera_manager.py       # Gestión de cámara
├── utils.py               # Funciones utilitarias
├── requirements.txt       # Dependencias
├── README.md             # Documentación
└── embeddings/           # Directorio de embeddings (crear manualmente)
    ├── persona1/
    │   ├── emb1.npy
    │   └── emb2.npy
    └── persona2/
        ├── emb1.npy
        └── emb2.npy
```

## Instalación

1. **Clonar o descargar los archivos del proyecto**

2. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

3. **Crear directorio de embeddings**:
```bash
mkdir embeddings
```

4. **Organizar embeddings por persona**:
   - Crear una carpeta por persona en `embeddings/`
   - Colocar archivos `.npy` de embeddings dentro de cada carpeta
   - El nombre de la carpeta será usado como identificador de la persona

## Configuración

Editar `config.py` para ajustar parámetros:

### Configuración de Cámara
```python
USE_RTSP = 0  # 0 para cámara local, 1 para RTSP
RTSP_URL = "rtsp://usuario:contraseña@ip:puerto/stream"
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
```

### Parámetros de Reconocimiento
```python
THRESHOLD = 0.50  # Umbral de similitud (0.0-1.0)
MIN_CONFIDENCE = 0.85  # Confianza mínima de detección
MIN_FACE_SIZE = 30  # Tamaño mínimo de cara en píxeles
```

### Optimizaciones para Larga Distancia
```python
ENABLE_SUPER_RESOLUTION = True  # Mejorar caras pequeñas
DISTANCE_ADAPTIVE_THRESHOLD = True  # Umbral dinámico
ENHANCED_PREPROCESSING = True  # Preprocesamiento avanzado
```

## Uso

### Ejecutar el Sistema
```bash
python main.py
```

### Controles Durante Ejecución
- **'q'**: Salir del sistema
- **'s'**: Mostrar estadísticas detalladas
- **'r'**: Resetear estadísticas

### Interpretación de Colores
- **Verde**: Alta confianza, cara cerca
- **Amarillo**: Confianza media
- **Naranja**: Baja confianza o cara lejos
- **Rojo**: Persona desconocida

## Optimizaciones Implementadas

### 1. Procesamiento de Imagen
- Mejora de contraste y brillo
- Filtro bilateral para reducir ruido
- Sharpening para mejorar detalles
- Super-resolución para caras pequeñas

### 2. Algoritmo de Reconocimiento
- Índice FAISS para búsqueda rápida
- Threshold adaptativo por tamaño de cara
- Suavizado temporal para estabilidad
- Validación de calidad de embeddings

### 3. Rendimiento
- Procesamiento selectivo de frames
- Búsqueda optimizada con top-k candidatos
- Estadísticas robustas para mejor matching
- Gestión eficiente de memoria

## Parámetros Técnicos

### Umbrales de Distancia
- **Cerca**: > 80px (threshold normal)
- **Medio**: 50-80px (threshold +0.05)
- **Lejos**: 30-50px (threshold +0.10)
- **Muy lejos**: < 30px (threshold +0.15)

### Métricas de Calidad
- Embedding válido: 512 dimensiones, norma > 0.1
- Consistencia temporal: mínimo 3 detecciones
- Penalización por distancia: hasta 20% menos confianza

## Solución de Problemas

### Error: "Directorio embeddings no existe"
Crear el directorio manualmente:
```bash
mkdir embeddings
```

### Error: "No se encontraron embeddings válidos"
- Verificar que existen archivos `.npy` en subdirectorios de `embeddings/`
- Verificar que los embeddings tienen 512 dimensiones
- Verificar que los archivos no están corruptos

### Baja tasa de reconocimiento
- Ajustar `THRESHOLD` (valores más altos = más estricto)
- Aumentar `MIN_CONFIDENCE` para mejor calidad de detección
- Verificar iluminación y calidad de video
- Añadir más embeddings de entrenamiento

### Rendimiento lento
- Aumentar `PROCESS_EVERY_N_FRAMES` (procesar menos frames)
- Reducir resolución de cámara
- Desactivar `ENABLE_SUPER_RESOLUTION`
- Usar GPU con `CTX_ID = 0` en lugar de CPU

## Archivos de Configuración

### config.py
Contiene todas las configuraciones del sistema, organizadas por categorías.

### requirements.txt
Lista todas las dependencias necesarias con versiones específicas.

## Arquitectura del Sistema

El sistema está diseñado con una arquitectura modular:

1. **main.py**: Orquesta todos los componentes
2. **OptimizedFaceRecognizer**: Maneja embeddings e identificación
3. **FaceProcessor**: Procesa frames y aplica lógica de reconocimiento
4. **CameraManager**: Gestiona la fuente de video
5. **ImageProcessor**: Funciones de mejora de imagen
6. **Utils**: Herramientas de monitoreo y logging

Esta separación permite fácil mantenimiento, testing y extensión del sistema.