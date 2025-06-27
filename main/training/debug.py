#!/usr/bin/env python3
"""
Debug script to check existing embeddings and verify naming patterns
"""
import os
from config import INPUT_DIR, OUTPUT_DIR

def debug_folder_structure():
    """Debug the folder structure and naming patterns"""
    print("🔍 DEBUGGING FOLDER STRUCTURE")
    print("=" * 50)
    
    # Check input structure
    print(f"📁 INPUT_DIR: {INPUT_DIR}")
    if os.path.exists(INPUT_DIR):
        for person_folder in os.listdir(INPUT_DIR):
            person_path = os.path.join(INPUT_DIR, person_folder)
            if os.path.isdir(person_path):
                images = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                print(f"   👤 {person_folder}/")
                for img in images[:3]:  # Show first 3 images
                    print(f"      📷 {img}")
                if len(images) > 3:
                    print(f"      ... and {len(images) - 3} more images")
    else:
        print(f"   ❌ {INPUT_DIR} does not exist!")
    
    print()
    
    # Check output structure
    print(f"📁 OUTPUT_DIR: {OUTPUT_DIR}")
    if os.path.exists(OUTPUT_DIR):
        for person_folder in os.listdir(OUTPUT_DIR):
            person_path = os.path.join(OUTPUT_DIR, person_folder)
            if os.path.isdir(person_path):
                embeddings = [f for f in os.listdir(person_path) if f.endswith('.npy')]
                print(f"   👤 {person_folder}/ ({len(embeddings)} embeddings)")
                
                # Group by type
                originals = [f for f in embeddings if '_original.npy' in f]
                augmentations = [f for f in embeddings if '_aug_' in f]
                
                print(f"      📊 Original: {len(originals)}")
                print(f"      📊 Augmentations: {len(augmentations)}")
                
                # Show examples
                if originals:
                    print(f"      📄 Example original: {originals[0]}")
                if augmentations:
                    print(f"      📄 Example aug: {augmentations[0]}")
                
                print()
    else:
        print(f"   ❌ {OUTPUT_DIR} does not exist!")

def check_specific_image(person_name, image_name):
    """Check embeddings for a specific image"""
    print(f"\n🔍 CHECKING SPECIFIC IMAGE: {person_name}/{image_name}")
    print("=" * 50)
    
    person_dir = os.path.join(OUTPUT_DIR, person_name)
    if not os.path.exists(person_dir):
        print(f"❌ Person directory doesn't exist: {person_dir}")
        return
    
    base_name = os.path.splitext(image_name)[0]
    
    # Check what embeddings exist
    all_files = os.listdir(person_dir)
    all_embeddings = [f for f in all_files if f.endswith('.npy')]
    
    print(f"📊 Total embeddings in {person_name}/: {len(all_embeddings)}")
    
    # Look for this specific image
    original_name = f"{person_name}_{base_name}_original.npy"
    aug_prefix = f"{person_name}_{base_name}_aug_"
    
    related_embeddings = []
    for emb in all_embeddings:
        if emb == original_name or emb.startswith(aug_prefix):
            related_embeddings.append(emb)
    
    print(f"📊 Embeddings for {image_name}: {len(related_embeddings)}")
    
    if original_name in related_embeddings:
        print(f"✅ Original embedding exists: {original_name}")
    else:
        print(f"❌ Original embedding missing: {original_name}")
    
    aug_embeddings = [f for f in related_embeddings if f.startswith(aug_prefix)]
    print(f"📊 Augmentation embeddings: {len(aug_embeddings)}")
    
    if aug_embeddings:
        print("   Sample augmentations:")
        for aug in aug_embeddings[:5]:
            print(f"      {aug}")
        if len(aug_embeddings) > 5:
            print(f"      ... and {len(aug_embeddings) - 5} more")

def main():
    """Main debug function"""
    debug_folder_structure()
    
    # You can manually check specific images here
    # Example: check_specific_image("john_doe", "photo1.jpg")
    
    print("\n💡 TIPS:")
    print("   - Check if person names match between input and output folders")
    print("   - Verify image names don't have special characters")
    print("   - Make sure the naming pattern matches exactly")
    print("   - Run this script to debug before running the main processing")

if __name__ == "__main__":
    main()