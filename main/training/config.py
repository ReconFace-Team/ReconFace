# === CONFIGURATION SETTINGS ===

# Directory paths
INPUT_DIR = "./src/images"
OUTPUT_DIR = "./src/embeddings"

# Augmentation settings
N_AUGMENTATIONS = 100

# Image processing settings
MAX_WIDTH = 1000
MAX_HEIGHT = 1000
MIN_DET_SCORE = 0.70

# Preview settings
SHOW_PREVIEW = True  # Change to False to disable preview
PREVIEW_DURATION = 2000  # Duration in milliseconds (2 seconds)

# Processing settings
MAX_ATTEMPTS = 50  # Maximum attempts per augmentation
REGENERATE_TRANSFORM_INTERVAL = 25  # Regenerate transform every N attempts
PREVIEW_INTERVAL = 10  # Show preview every N successful augmentations
LOG_INTERVAL = 10  # Show log messages every N attempts

# Embedding checking settings
CHECK_EXISTING_EMBEDDINGS = True  # Check for existing embeddings before processing
MIN_EMBEDDINGS_THRESHOLD = 50  # Skip person if they already have this many embeddings
SKIP_COMPLETED_IMAGES = True  # Skip images that are already fully processed