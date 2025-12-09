import albumentations as A

# === GENTLE TRANSFORMATIONS FOR BETTER DETECTION ===
def get_transform_options():
    """
    Devuelve una lista de *factories* de transformaciones de Albumentations
    que preservan los rasgos faciales para detección/reconocimiento.
    Cada elemento de la lista es un `lambda: A.Compose(...)` o un transform
    simple que luego se usará como:  transform_factory()(image=img)["image"].
    """
    return [
        # Transformaciones suaves individuales
        lambda: A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=1.0
        ),

        lambda: A.GaussianBlur(
            blur_limit=(1, 3),
            p=1.0
        ),

        # GaussNoise sin var_limit para evitar warnings en tu versión
        lambda: A.GaussNoise(p=1.0),

        # Flip horizontal correcto
        lambda: A.HorizontalFlip(p=1.0),

        # Rotación suave
        lambda: A.Rotate(
            limit=15,
            p=1.0
        ),

        # Escalado ligero
        lambda: A.RandomScale(
            scale_limit=0.1,
            p=1.0
        ),

        # Combinaciones suaves
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
            A.GaussNoise(p=0.5),
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
