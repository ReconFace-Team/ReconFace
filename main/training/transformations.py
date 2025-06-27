import albumentations as A

# === GENTLE TRANSFORMATIONS FOR BETTER DETECTION ===
def get_transform_options():
    """
    Returns a list of transformation functions that preserve facial features
    for better face detection and recognition.
    """
    return [
        # Gentle transformations that preserve facial features
        lambda: A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
        lambda: A.GaussianBlur(blur_limit=(1, 3), p=1.0),
        lambda: A.GaussNoise(var_limit=(5.0, 20.0), p=1.0),
        lambda: A.HorizontalFlip(p=1.0),
        lambda: A.Rotate(limit=15, p=1.0),  # Gentle rotation
        lambda: A.RandomScale(scale_limit=0.1, p=1.0),  # Light scaling
        
        # Gentle combinations
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
        
        # Identity transformation (no changes) for more variety
        lambda: A.Compose([]),
    ]