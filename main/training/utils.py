import cv2
import os
import numpy as np
from config import SHOW_PREVIEW
from send2trash import send2trash
import sys
import shutil
import pathlib

def show_preview(img, faces, window_name="Preview", duration=2000):
    if not SHOW_PREVIEW:
        return

    preview_img = img.copy()

    for face in faces:
        # Obtener bbox y score de forma robusta
        bbox = getattr(face, "bbox", None)
        det = getattr(face, "det_score", None)
        if bbox is None and isinstance(face, dict):
            bbox = face.get("bbox")
            det  = face.get("det_score")
        if bbox is None:
            continue

        # Asegurar ints
        x1, y1, x2, y2 = [int(v) for v in (bbox if isinstance(bbox, (list, tuple, np.ndarray)) else list(bbox))]
        cv2.rectangle(preview_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if det is not None:
            cv2.putText(preview_img, f"{float(det):.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    h, w = preview_img.shape[:2]
    if w > 800 or h > 600:
        scale = min(800/w, 600/h)
        new_w, new_h = int(w*scale), int(h*scale)
        preview_img = cv2.resize(preview_img, (new_w, new_h))

    cv2.imshow(window_name, preview_img)
    cv2.waitKey(duration)
    cv2.destroyAllWindows()


def resize_image_if_needed(img, max_width, max_height):
    """Resize image if it exceeds maximum dimensions"""
    height, width = img.shape[:2]
    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        new_size = (int(width * scale), int(height * scale))
        img = cv2.resize(img, new_size)
        return img, new_size, True
    return img, (width, height), False

def is_image_file(filename):
    """Check if file is a valid image format"""
    return filename.lower().endswith((".jpg", ".png", ".jpeg"))

def create_output_directory(person_name, output_dir):
    """Create output directory for a person if it doesn't exist"""
    person_dir = os.path.join(output_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)
    return person_dir

def count_existing_embeddings(person_dir):
    """Count existing .npy embedding files in a person's directory"""
    if not os.path.exists(person_dir):
        return 0
    
    embedding_files = [f for f in os.listdir(person_dir) if f.endswith('.npy')]
    return len(embedding_files)

def get_existing_image_embeddings(person_dir, person_name, base_name):
    """Get list of existing embeddings for a specific image"""
    if not os.path.exists(person_dir):
        return []
    
    existing_embeddings = []
    # Look for embeddings that match the exact naming pattern
    original_pattern = f"{person_name}_{base_name}_original.npy"
    aug_pattern_prefix = f"{person_name}_{base_name}_aug_"
    
    for filename in os.listdir(person_dir):
        if filename.endswith('.npy'):
            if filename == original_pattern or filename.startswith(aug_pattern_prefix):
                existing_embeddings.append(filename)
    
    return existing_embeddings

def check_if_image_processed(person_dir, person_name, base_name, n_augmentations):
    """Check if an image has already been fully processed"""
    existing = get_existing_image_embeddings(person_dir, person_name, base_name)
    
    # Check for original embedding
    original_name = f"{person_name}_{base_name}_original.npy"
    has_original = original_name in existing
    
    # Count augmentation embeddings
    aug_pattern_prefix = f"{person_name}_{base_name}_aug_"
    aug_count = len([f for f in existing if f.startswith(aug_pattern_prefix)])
    
    return has_original, aug_count

def should_skip_person(person_dir, min_embeddings_threshold=50):
    """Check if a person already has sufficient embeddings"""
    total_embeddings = count_existing_embeddings(person_dir)
    return total_embeddings >= min_embeddings_threshold

def print_final_tips(min_det_score):
    """Print helpful tips for improving detection"""
    print("🏁 Processing completed.")
    print("💡 Tips for improving detection:")
    print("   - Use images with well-lit and frontal faces")
    print(f"   - Verify that MIN_DET_SCORE is not too high (current: {min_det_score})")
    print("   - Consider using higher quality images")


def safe_delete_file(path, move_to_trash=False):
    """Elimina el archivo. Si move_to_trash=False → borrado permanente."""
    try:
        if not os.path.exists(path):
            return True
        if move_to_trash:
            try:
                
                send2trash(path)
                print(f"🗑️  Enviado a Papelera: {path}")
                return True
            except Exception as e:
                print(f"ℹ️  No se pudo usar Papelera ({e}), borrando permanente: {path}")
        os.remove(path)
        print(f"🧹 Borrado permanente: {path}")
        return True
    except Exception as e:
        print(f"❌ Error al eliminar {path}: {e}")
        return False

def remove_dir_if_empty(dir_path):
    """Elimina la carpeta si quedó vacía."""
    try:
        if os.path.isdir(dir_path) and len(os.listdir(dir_path)) == 0:
            os.rmdir(dir_path)
            print(f"🧹 Carpeta vacía eliminada: {dir_path}")
            return True
    except Exception as e:
        print(f"⚠️ No se pudo eliminar carpeta vacía {dir_path}: {e}")
    return False

def compute_dynamic_augs_for_image(person_dir, current_person_total, originals_left):
    """
    Calcula cuántas augmentations generar para ESTA imagen, para acercarse
    a TARGET_EMB_PER_PERSON, repartiendo el faltante entre las imágenes que quedan.
    Respeta AUGS_PER_IMAGE_MIN/MAX.
    """
    from config import TARGET_EMB_PER_PERSON, AUGS_PER_IMAGE_MIN, AUGS_PER_IMAGE_MAX

    remaining = max(0, TARGET_EMB_PER_PERSON - current_person_total)
    if remaining == 0 or originals_left <= 0:
        return 0

    per_image_need = int(np.ceil(remaining / originals_left))
    per_image_need = max(AUGS_PER_IMAGE_MIN, min(AUGS_PER_IMAGE_MAX, per_image_need))
    return per_image_need
