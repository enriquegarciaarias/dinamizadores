# 🏛️ Modelo de LLM para Dinamizadores

Este repositorio contiene el código para la generación automatizada de descripciones de imágenes.  
El sistema se estructura en 3 fases:
- 🧮 **Fase I. Características y Agrupamiento**
- 🧭 **Fase II. Contextualización**
- 🧠 **Fase III. Ingeniería de prompts**

## 🗂️ Dataset
El repositorio incluye un dataset en el directorio `process/input`.  
Su estructura consiste en un conjunto de imágenes y un documento Word con información de las imágenes que pueda ser utilizada como marco contextual.

## 🛠️ Instalación
El proyecto utiliza Python 3.19. Si no dispone de Python instalado, puede descargarlo de la web oficial.  
Para la instalación de las librerías necesarias, puede utilizarse `pip` con el archivo `requirements.txt` incluido.

1. Clonar el repositorio:
```bash
git clone https://github.com/enriquegarciaarias/DescripImg.git
2. Instalar las dependencias:
bash
cd DescripImg
pip install -r requirements.txt
```

2.  Miscelanea

Instalación de llama-cpp
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

## ⚙️ Configuración
La configuración se realiza en config.js, que **no se indexa en el repositorio**. Este archivo se genera a partir del archivo config.ORG incluido.

1. Copiar config_example.json y pegarlo a la misma altura que el original. Cambiarle el nombre a la copia por config.js.
2. Ajustar los valores en config.js según el entorno de tu equipo.

## 🚀 Ejecución
los procesos se lanzan con el comando:
```bash
cd DescripImg
python3.10 main.py --proc=OPCION
```

Donde "OPCION" puede ser:
- MODEL: Construye los modelos base de características y clústering
- APPLY: Aplica los modelos sobre nuevas imágenes FIN MARKDOWN

