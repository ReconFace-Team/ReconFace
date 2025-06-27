import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import random
from transformations import get_transform_options
from utils import (
    show_preview, resize_image_if_needed, create_output_directory,
    count_existing_embeddings, check_if_image_processed, should_skip_person
)
from config import (
    MIN_DET_SCORE, MAX_ATTEMPTS, REGENERATE_TRANSFORM_INTERVAL, 
    PREVIEW_INTERVAL, LOG_INTERVAL, N_AUGMENTATIONS,
    CHECK_EXISTING_EMBEDDINGS, MIN_EMBEDDINGS_THRESHOLD, SKIP_COMPLETED_IMAGES,
    INPUT_DIR, OUTPUT_DIR
)

class FaceProcessor:
    """Handles face detection and embedding generation"""
    
    def __init__(self):
        """Initialize the face analysis model"""
        print("🔄 Initializing facial detection model...")
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0)  # Use 0 if GPU is available
        self.transform_options = get_transform_options()
        print("✅ Model initialized")
    
    def detect_faces_in_original(self, img, filename):
        """Detect faces in original image"""
        print("🔍 Testing detection on original image...")
        original_faces = self.app.get(img)
        
        if original_faces:
            print(f"✅ {len(original_faces)} face(s) detected in original image")
            show_preview(img, original_faces, f"Original - {filename}")
            return original_faces
        else:
            print("⚠️ No faces detected in original image")
            show_preview(img, [], f"Original (no faces) - {filename}")
            return None
    
    def save_original_embedding(self, faces, person_name, base_name, person_dir):
        """Save embedding from original image"""
        if faces[0].det_score >= MIN_DET_SCORE:
            # Check if original embedding already exists
            emb_name = f"{person_name}_{base_name}_original.npy"
            save_path = os.path.join(person_dir, emb_name)
            
            if os.path.exists(save_path):
                print("✅ Original embedding already exists, skipping")
                return True
            
            emb = faces[0].embedding
            np.save(save_path, emb)
            print("✅ Original embedding saved")
            return True
        return False
    
    def generate_augmented_embedding(self, img, transform_factory, attempt_num):
        """Generate a single augmented embedding"""
        # Regenerate transformation periodically
        if attempt_num % REGENERATE_TRANSFORM_INTERVAL == 1:
            transform = transform_factory()
        else:
            transform = transform_factory()
        
        # Apply transformation to image
        aug_img = transform(image=img)['image']
        
        # Face detection
        faces = self.app.get(aug_img)
        
        if faces:
            face = faces[0]
            if face.det_score >= MIN_DET_SCORE:
                return aug_img, face, faces
        
        return None, None, None
    
    def process_augmentations(self, img, filename, person_name, base_name, person_dir):
        """Process all augmentations for an image"""
        
        # Check if we should skip completed images
        if CHECK_EXISTING_EMBEDDINGS and SKIP_COMPLETED_IMAGES:
            has_original, existing_aug_count = check_if_image_processed(
                person_dir, person_name, base_name, N_AUGMENTATIONS
            )
            
            if has_original and existing_aug_count >= N_AUGMENTATIONS:
                print(f"✅ Image already fully processed with {existing_aug_count} augmentations, skipping")
                return existing_aug_count
            elif existing_aug_count > 0:
                print(f"📊 Found {existing_aug_count} existing augmentations, generating {N_AUGMENTATIONS - existing_aug_count} more")
        
        count = 0
        successful_augmentations = 0
        
        print(f"🔄 Generating {N_AUGMENTATIONS} augmentations...")
        
        while count < N_AUGMENTATIONS:
            # Check if this specific augmentation already exists
            emb_name = f"{person_name}_{base_name}_aug_{count}.npy"
            save_path = os.path.join(person_dir, emb_name)
            
            if os.path.exists(save_path):
                print(f"✅ Embedding {count + 1}/{N_AUGMENTATIONS} already exists, skipping")
                successful_augmentations += 1
                count += 1
                continue
            
            single_attempts = 0
            success = False
            
            # Choose a transformation generator function
            transform_factory = random.choice(self.transform_options)
            
            while single_attempts < MAX_ATTEMPTS:
                single_attempts += 1
                
                aug_img, face, faces = self.generate_augmented_embedding(
                    img, transform_factory, single_attempts
                )
                
                if face is not None:
                    # Show preview only every N successful embeddings
                    if successful_augmentations % PREVIEW_INTERVAL == 0:
                        show_preview(aug_img, faces, f"Aug {count+1} - {filename}", 1000)
                    
                    # Save embedding
                    emb = face.embedding
                    np.save(save_path, emb)
                    
                    successful_augmentations += 1
                    print(f"✅ Embedding {count + 1}/{N_AUGMENTATIONS} saved (confidence: {face.det_score:.2f}, attempts: {single_attempts})")
                    success = True
                    break
                else:
                    # Show fewer log messages
                    if single_attempts % LOG_INTERVAL == 0:
                        if faces:  # Face detected but low confidence
                            print(f"[!] Low confidence, attempt {single_attempts}/{MAX_ATTEMPTS}")
                        else:  # No face detected
                            print(f"[!] No face detected, attempt {single_attempts}/{MAX_ATTEMPTS}")
            
            if not success:
                print(f"⚠️ Could not generate embedding {count + 1} after {MAX_ATTEMPTS} attempts")
            
            count += 1
        
        return successful_augmentations
    
    def process_single_image(self, img_path, filename, output_dir, max_width, max_height):
        """Process a single image file"""
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"❌ Could not read {img_path}")
            return False
        
        print(f"\n📸 Processing: {filename}")
        
        # Resize if exceeds maximum size
        img, size, was_resized = resize_image_if_needed(img, max_width, max_height)
        if was_resized:
            print(f"🔄 Image resized: {filename} → {size}")

        # Test detection on original image
        original_faces = self.detect_faces_in_original(img, filename)
        if not original_faces:
            return False  # Skip if no faces in original

        # Setup paths and directories
        base_name = os.path.splitext(filename)[0]
        root = os.path.dirname(img_path)
        rel_dir = os.path.relpath(root, INPUT_DIR)  # Relative path from INPUT_DIR
        person_name = rel_dir.split(os.sep)[0]  # Extract the top-level folder name

        print(f"📂 Person Name: {person_name}")

        # Check if a folder with the same name as person_name exists in the embeddings directory
        person_dir = os.path.join(OUTPUT_DIR, person_name)
        if os.path.isdir(person_dir):
            print(f"✅ Embeddings folder already exists for '{person_name}', checking individual images...")
        else:
            # Create output directory for the person if it doesn't exist
            person_dir = create_output_directory(person_name, OUTPUT_DIR)

        # Save original embedding
        self.save_original_embedding(original_faces, person_name, base_name, person_dir)

        # Process augmentations
        successful_augmentations = self.process_augmentations(
            img, filename, person_name, base_name, person_dir
        )

        print(f"📊 Summary for {filename}: {successful_augmentations}/{N_AUGMENTATIONS} successful embeddings")
        return True