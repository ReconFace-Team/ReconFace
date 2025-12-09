# ReconFace

**ReconFace** es un sistema integral de reconocimiento facial en tiempo real. Esta versión evolucionada del proyecto introduce una arquitectura modular y una interfaz gráfica de usuario (GUI) dedicada para facilitar la gestión de identidades, el monitoreo en vivo y la administración del sistema.

## 📋 Descripción

El proyecto ha sido reestructurado para separar la lógica de procesamiento visual de la interfaz de usuario, ofreciendo una solución más robusta y mantenible. Utiliza una base de datos ligera basada en JSON para la persistencia de datos, eliminando la necesidad de configuraciones de bases de datos complejas para despliegues locales.

## ✨ Características Principales

* **Interfaz Gráfica (GUI)**: Punto de entrada unificado a través de `run_gui.py`, permitiendo una interacción visual amigable en lugar de comandos de consola.
* **Gestión de Datos Simplificada**: Almacenamiento de embeddings faciales y metadatos de usuarios en `database.json`.
* **Sistema de Auditoría**: Generación automática de registros de actividad y errores en el directorio `logs/`.
* **Arquitectura Modular**:
  * **`src/`**: Núcleo del procesamiento y algoritmos de reconocimiento.
  * **`gui/`**: Componentes visuales y ventanas.
  * **`main/`**: Scripts de ejecución lógica.
* **Soporte de Pruebas**: Incluye un directorio `test/` para validación de funcionalidades.

## 📂 Estructura del Proyecto

La organización actual del repositorio es la siguiente:

```text
ReconFace/
├── run_gui.py              # Script principal de ejecución (Entry Point)
├── database.json           # Base de datos de identidades
├── requirements.txt        # Dependencias del proyecto
├── src/                    # Código fuente del motor de reconocimiento
├── gui/                    # Código fuente de la interfaz gráfica
├── main/                   # Módulos principales de lógica
├── logs/                   # Archivos de log (creado en runtime)
├── test/                   # Scripts de pruebas unitarias
└── THIRD_PARTY_NOTICES.txt # Licencias de terceros
````

## 🛠️ Instalación

### Requisitos Previos

  * Python 3.10
  * CUDA 12.2, cuDNN 9.0.X y TensorRT 10.X
  * Webcam o cámara IP disponible
  * Git instalado

### Pasos de Instalación

1.  **Clonar el repositorio:**

    ```bash
    git clone [https://github.com/ReconFace-Team/ReconFace.git](https://github.com/ReconFace-Team/ReconFace.git)
    cd ReconFace
    ```

2.  **Crear y activar un entorno virtual (Opcional pero recomendado):**

    ```bash
    python -m venv venv
    
    # En Windows:
    venv\Scripts\activate
    
    # En Linux/Mac:
    source venv/bin/activate
    ```

3.  **Instalar dependencias:**

    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Uso

Para iniciar la aplicación con la interfaz gráfica, ejecute el siguiente comando desde la raíz del proyecto:

```bash
python run_gui.py
```

### Funcionalidades Esperadas en la GUI:

  * **Registro**: Captura y almacenamiento de nuevos rostros según entrenamiento.
  * **Monitoreo**: Visualización en tiempo real con bounding boxes e identificación.
  * **Logs**: Revisión de eventos pasados (dependiendo de la implementación de la GUI).

## 📄 Licencia y Avisos

Revise el archivo `THIRD_PARTY_NOTICES.txt` para información sobre las licencias de las librerías y componentes de terceros utilizados en este proyecto.

Copyright © 2024-2025 ReconFace Team
