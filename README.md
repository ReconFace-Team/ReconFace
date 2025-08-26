# Sistema de Reconocimiento Facial Optimizado

Sistema de reconocimiento facial en tiempo real optimizado para detección a larga distancia con múltiples mejoras de rendimiento y precisión.
Patch C++

## Requerimientos
- Visual Studio Community
- [OpenCV](https://github.com/opencv/opencv/releases/latest)

## Instrucciones (Windows)
1. Descargar el ejecutable de la latest build de OpenCV.
2. Extraer los archivos a cualquier carpeta. Debe de quedar así:
```
C:/
├─ opencv/
│  ├─ build/
│  ├─ sources/
│  ├─ LICENSE.txt
│  ├─ LICENSE_FFMPEG.txt
│  ├─ README.md.txt
```
3. Configurar en las *Variables de Entorno* los siguientes valores:
```
Variables de usuario (Path) y Variables del sistema (Path)
├─ C:\opencv\build\x64\vcXX\bin

Donde XX corresponde a la versión. En este caso, vc16.
```
4. Crear (o abrir el proyecto en Visual Studio)
5. Configurar el proyecto de la siguiente forma:
```
Proyecto -> Propiedades -> Directorios de VC++
├─ General
│  ├─ Directorios de archivos de inclusión -> Editar y añadir: C:\opencv\build\include
│  ├─ Directorios de archivos de bibliotecas -> Editar y añadir: C:\opencv\build\x64\vc16\lib

Proyecto -> Propiedades -> Vinculador
├─ Entrada
│  ├─ Dependencias adicionales -> Editar y añadir: opencv_worldXXXXd.lib

Donde XXXX es un número basado en la versión, en este caso, 4120
(Este archivo se encuentra en C:\opencv\build\x64\vc16\lib)
```
