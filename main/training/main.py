import os
from face_processor import FaceProcessor
from utils import is_image_file, print_final_tips, count_existing_embeddings
from config import INPUT_DIR, OUTPUT_DIR, MAX_WIDTH, MAX_HEIGHT, MIN_DET_SCORE, CHECK_EXISTING_EMBEDDINGS

def print_existing_embeddings_summary():
    """Print summary of existing embeddings"""
    if not os.path.exists(OUTPUT_DIR):
        print("📁 No existing embeddings directory found")
        return
    
    print("📊 Existing Embeddings Summary:")
    total_embeddings = 0
    
    for person_folder in os.listdir(OUTPUT_DIR):
        person_path = os.path.join(OUTPUT_DIR, person_folder)
        if os.path.isdir(person_path):
            count = count_existing_embeddings(person_path)
            if count > 0:
                print(f"   {person_folder}: {count} embeddings")
                total_embeddings += count
    
    print(f"   Total existing embeddings: {total_embeddings}\n")

def main():
    """Main function to process all images in subdirectories"""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Show existing embeddings summary if checking is enabled
    if CHECK_EXISTING_EMBEDDINGS:
        print_existing_embeddings_summary()
    
    # Initialize face processor
    processor = FaceProcessor()
    
    # Process all images in subdirectories
    processed_count = 0
    skipped_count = 0
    total_count = 0
    
    for root, _, files in os.walk(INPUT_DIR):
        for filename in files:
            if not is_image_file(filename):
                continue
            
            total_count += 1
            img_path = os.path.join(root, filename)
            
            success = processor.process_single_image(
                img_path, filename, OUTPUT_DIR, MAX_WIDTH, MAX_HEIGHT, INPUT_DIR
            )
            
            if success:
                # Check if it was actually processed or skipped
                if "skipping" in str(success).lower():
                    skipped_count += 1
                else:
                    processed_count += 1
            else:
                skipped_count += 1
    
    # Print final statistics and tips
    print(f"\n📊 Final Statistics:")
    print(f"   Total images found: {total_count}")
    print(f"   Successfully processed: {processed_count}")
    print(f"   Skipped: {skipped_count}")
    
    # Show final embeddings summary
    if CHECK_EXISTING_EMBEDDINGS:
        print("\n" + "="*50)
        print_existing_embeddings_summary()
    
    print_final_tips(MIN_DET_SCORE)

if __name__ == "__main__":
    main()