# main/recognition/transformations.py
import albumentations as A

def get_transform_options():
    """
    Devuelve una lista de fábricas de transformaciones. Cada fábrica, al llamarla,
    retorna un objeto "transform" que acepta: transform(image=RGB)['image'].
    Mantenemos cambios suaves para no distorsionar rasgos faciales.
    """
    return [
        # Transformaciones suaves
        lambda: A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
        lambda: A.GaussianBlur(blur_limit=(1, 3), p=1.0),
        lambda: A.GaussNoise(var_limit=(5.0, 20.0), p=1.0),
        lambda: A.HorizontalFlip(p=1.0),
        lambda: A.Rotate(limit=15, p=1.0),
        lambda: A.RandomScale(scale_limit=0.1, p=1.0),

        # Combinaciones suaves
        lambda: A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
            A.GaussianBlur(blur_limit=(1, 2), p=0.5),
        ]),
        lambda: A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.GaussNoise(var_limit=(5.0, 15.0), p=0.5),
            A.HorizontalFlip(p=0.5),
        ]),
        lambda: A.Compose([
            A.Rotate(limit=10, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.7),
        ]),

        # Identidad
        lambda: A.Compose([]),
    ]
