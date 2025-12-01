import albumentations as A

# === GENTLE TRANSFORMATIONS FOR BETTER DETECTION ===
def get_transform_options():
    """
    Returns a list of transformation functions that preserve facial features
    for better face detection and recognition.
    Each item is a factory (lambda) that returns an Albumentations transform
    used as: transform_factory()(image=img)["image"].
    """
    return [
        # --- Transformaciones suaves individuales ---
        lambda: A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=1.0
        ),

        lambda: A.GaussianBlur(
            blur_limit=(1, 3),
            p=1.0
        ),

        # GaussNoise con API nueva (std_range en vez de var_limit)
        # Aproximación de (5..20) / 255 => (0.02..0.08)
        lambda: A.GaussNoise(
            std_range=(0.02, 0.08),
            mean_range=(0.0, 0.0),
            p=1.0
        ),

        lambda: A.HorizontalFlip(p=1.0),

        lambda: A.Rotate(
            limit=15,
            p=1.0
        ),

        lambda: A.RandomScale(
            scale_limit=0.1,
            p=1.0
        ),

        # --- Combinaciones suaves ---
        lambda: A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=1.0
            ),
            A.GaussianBlur(
                blur_limit=(1, 2),
                p=0.5
            ),
        ]),

        lambda: A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            # Antes: var_limit=(5.0, 15.0) → ahora aprox (0.02..0.06)
            A.GaussNoise(
                std_range=(0.02, 0.06),
                mean_range=(0.0, 0.0),
                p=0.5
            ),
            A.HorizontalFlip(p=0.5),
        ]),

        lambda: A.Compose([
            A.Rotate(limit=10, p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.7
            ),
        ]),

        # Identidad (no cambia nada) para tener también casos "limpios"
        lambda: A.Compose([]),
    ]
