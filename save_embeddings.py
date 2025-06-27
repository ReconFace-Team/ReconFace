import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import albumentations as A
import random
import time

# === CONFIGURACIÓN ===
INPUT_DIR = "images"
OUTPUT_DIR = "embeddings"
N_AUGMENTATIONS = 100
MAX_WIDTH = 1000
MAX_HEIGHT = 1000
MIN_DET_SCORE = 0.70
SHOW_PREVIEW = True  # Cambiar a False para desactivar previsualización
PREVIEW_DURATION = 2000  # Duración en milisegundos (2 segundos)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === TRANSFORMACIONES MÁS SUAVES PARA MEJOR DETECCIÓN ===
transform_options = [
    # Transformaciones suaves que preservan características faciales
    lambda: A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
    lambda: A.GaussianBlur(blur_limit=(1, 3), p=1.0),
    lambda: A.GaussNoise(var_limit=(5.0, 20.0), p=1.0),
    lambda: A.HorizontalFlip(p=1.0),
    lambda: A.Rotate(limit=15, p=1.0),  # Rotación suave
    lambda: A.RandomScale(scale_limit=0.1, p=1.0),  # Escala ligera
    
    # Combinaciones suaves
    lambda: A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
        A.GaussianBlur(blur_limit=(1, 2), p=0.5),
    ]),
    lambda: A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        A.GaussNoise(var_limit=(5.0, 15.0), p=0.5),
        A.HorizontalFlip(p=0.5)
    ]),
    lambda: A.Compose([
        A.Rotate(limit=10, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.7),
    ]),
    
    # Transformación identidad (sin cambios) para tener más variedad
    lambda: A.Compose([]),
]

def show_preview(img, faces, window_name="Preview", duration=2000):
    """Muestra una previsualización con los rostros detectados"""
    if not SHOW_PREVIEW:
        return
    
    preview_img = img.copy()
    
    # Dibujar rectángulos alrededor de los rostros detectados
    for face in faces:
        bbox = face.bbox.astype(int)
        cv2.rectangle(preview_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        # Mostrar puntuación de confianza
        cv2.putText(preview_img, f"{face.det_score:.2f}", 
                   (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Redimensionar para mejor visualización si es muy grande
    h, w = preview_img.shape[:2]
    if w > 800 or h > 600:
        scale = min(800/w, 600/h)
        new_w, new_h = int(w*scale), int(h*scale)
        preview_img = cv2.resize(preview_img, (new_w, new_h))
    
    cv2.imshow(window_name, preview_img)
    cv2.waitKey(duration)
    cv2.destroyAllWindows()

# === INICIALIZAR MODELO DE DETECCIÓN DE ROSTROS ===
print("🔄 Inicializando modelo de detección facial...")
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)  # Usa 0 si tienes GPU disponible
print("✅ Modelo inicializado")

# === RECORRER TODAS LAS IMÁGENES EN SUBCARPETAS ===
for root, _, files in os.walk(INPUT_DIR):
    for filename in files:
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        path = os.path.join(root, filename)
        img = cv2.imread(path)

        if img is None:
            print(f"❌ No se pudo leer {path}")
            continue

        print(f"\n📸 Procesando: {filename}")

        # Redimensionar si excede el tamaño máximo
        height, width = img.shape[:2]
        if width > MAX_WIDTH or height > MAX_HEIGHT:
            scale = min(MAX_WIDTH / width, MAX_HEIGHT / height)
            new_size = (int(width * scale), int(height * scale))
            img = cv2.resize(img, new_size)
            print(f"🔄 Imagen redimensionada: {filename} → {new_size}")

        # Probar detección en imagen original primero
        print("🔍 Probando detección en imagen original...")
        original_faces = app.get(img)
        if original_faces:
            print(f"✅ {len(original_faces)} rostro(s) detectado(s) en imagen original")
            show_preview(img, original_faces, f"Original - {filename}")
        else:
            print("⚠️ No se detectaron rostros en imagen original")
            show_preview(img, [], f"Original (sin rostros) - {filename}")
            continue  # Saltar si no hay rostros en la original

        base_name = os.path.splitext(filename)[0]
        rel_dir = os.path.relpath(root, INPUT_DIR)
        person_name = rel_dir.split(os.sep)[0]

        person_dir = os.path.join(OUTPUT_DIR, person_name)
        os.makedirs(person_dir, exist_ok=True)

        # Guardar embedding de la imagen original
        if original_faces[0].det_score >= MIN_DET_SCORE:
            emb = original_faces[0].embedding
            emb_name = f"{person_name}_{base_name}_original.npy"
            save_path = os.path.join(person_dir, emb_name)
            np.save(save_path, emb)
            print(f"✅ Embedding original guardado")

        count = 0
        successful_augmentations = 0
        
        print(f"🔄 Generando {N_AUGMENTATIONS} augmentaciones...")
        
        while count < N_AUGMENTATIONS:
            single_attempts = 0
            success = False
            max_attempts = 50  

            # Elegir una función generadora de transformación
            transform_factory = random.choice(transform_options)
            transform = transform_factory()

            while single_attempts < max_attempts:
                single_attempts += 1

                # Regenerar transformación cada 25 intentos
                if single_attempts % 25 == 1:
                    transform = transform_factory()

                # Aplicar transformación a la imagen
                aug_img = transform(image=img)['image']

                # Detección de rostro
                faces = app.get(aug_img)

                if faces:
                    face = faces[0]
                    if face.det_score >= MIN_DET_SCORE:
                        # Mostrar previsualización solo cada 10 embeddings exitosos
                        if successful_augmentations % 10 == 0:
                            show_preview(aug_img, faces, f"Aug {count+1} - {filename}", 1000)
                        
                        emb = face.embedding
                        emb_name = f"{person_name}_{base_name}_aug_{count}.npy"
                        save_path = os.path.join(person_dir, emb_name)
                        np.save(save_path, emb)
                        
                        successful_augmentations += 1
                        print(f"✅ Embedding {count + 1}/{N_AUGMENTATIONS} guardado (confianza: {face.det_score:.2f}, intentos: {single_attempts})")
                        success = True
                        break
                    else:
                        if single_attempts % 10 == 0:  # Mostrar menos mensajes
                            print(f"[!] Baja confianza ({face.det_score:.2f}) intento {single_attempts}/{max_attempts}")
                else:
                    if single_attempts % 10 == 0:  # Mostrar menos mensajes
                        print(f"[!] Sin rostro detectado, intento {single_attempts}/{max_attempts}")

            if not success:
                print(f"⚠️ No se pudo generar embedding {count + 1} tras {max_attempts} intentos")

            count += 1

        print(f"📊 Resumen para {filename}: {successful_augmentations}/{N_AUGMENTATIONS} embeddings exitosos")

print("🏁 Procesamiento finalizado.")
print("💡 Consejos para mejorar detección:")
print("   - Usar imágenes con rostros bien iluminados y frontales")
print("   - Verificar que MIN_DET_SCORE no sea muy alto (actual: {})".format(MIN_DET_SCORE))
print("   - Considerar usar imágenes de mayor calidad")