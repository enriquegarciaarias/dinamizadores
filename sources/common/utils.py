from sources.common.common import logger, processControl, log_
import json

import time
import os
from os.path import isdir
from PyPDF2 import PdfReader
from docx import Document
import shutil
from collections import Counter
import re


def mkdir(dir_path):
    """
    @Desc: Creates directory if it doesn't exist.
    @Usage: Ensures a directory exists before proceeding with file operations.
    """
    if not isdir(dir_path):
        os.makedirs(dir_path)


def dbTimestamp():
    """
    @Desc: Generates a timestamp formatted as "YYYYMMDDHHMMSS".
    @Result: Formatted timestamp string.
    """
    timestamp = int(time.time())
    formatted_timestamp = str(time.strftime("%Y%m%d%H%M%S", time.gmtime(timestamp)))
    return formatted_timestamp

class configLoader:
    """
    @Desc: Loads and provides access to JSON configuration data.
    @Usage: Instantiates with path to config JSON file.
    """
    def __init__(self, config_path='config.json'):
        self.base_path = os.path.realpath(os.getcwd())
        realConfigPath = os.path.join(self.base_path, config_path)
        self.config = self.load_config(realConfigPath)

    def load_config(self, realConfigPath):
        with open(realConfigPath, 'r') as config_file:
            return json.load(config_file)

    def get_environment(self):
        environment =  self.config.get("environment", None)
        environment["realPath"] = self.base_path
        return environment

    def get_defaults(self):
        return self.config.get("defaults", {})

    def get_models(self):
        return self.config.get("models", {})

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out

def huggingface_login(token):
    from huggingface_hub import login
    try:
        # Add your Hugging Face token here, or retrieve it from environment variables
        token = processControl.defaults['huggingface_login']
        login(token)
        print("Successfully logged in to Hugging Face.")
    except Exception as e:
        print("Error logging into Hugging Face:", str(e))
        raise


def extraer_texto(archivo):
    """Extrae texto de PDF o Word"""
    if archivo.endswith('.pdf'):
        with open(archivo, 'rb') as f:
            reader = PdfReader(f)
            return "\n".join([page.extract_text() for page in reader.pages])
    elif archivo.endswith(('.docx', '.doc')):
        doc = Document(archivo)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        raise ValueError("Formato de archivo no soportado")


def grabaJson(data, path):
    try:

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    except Exception as e:
        log_("error", logger, f"Error write json path:{path}, error:{e}")
        return False

    log_("info", logger, f"JSON written path:{path}")
    return True


def clean_and_move(path, filepath1, filepath2):
    # Validate inputs
    if not os.path.exists(path):
        print(f"Error: Path {path} does not exist")
        return False

    try:
        # Recursively delete all contents under path
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                try:
                    os.unlink(file_path)
                except Exception as e:
                    raise Exception(f"Failed to delete {file_path}: {e}")


            for name in dirs:
                dir_path = os.path.join(root, name)
                try:
                    os.rmdir(dir_path)
                except Exception as e:
                    raise Exception(f"Failed to delete {dir_path}: {e}")

        print(f"Successfully cleaned directory: {path}")

        if filepath1 is None:
            return True

        if not os.path.exists(filepath1):
            raise Exception(f"Error: Source file {filepath1} does not exist")

        if not os.path.exists(filepath2):
            raise Exception(f"Error: Source file {filepath2} does not exist")

        # Move filepath1 to filepath2
        shutil.move(filepath1, filepath2)
        print(f"Successfully moved {filepath1} to {filepath2}")

        return True

    except Exception as e:
        log_("exception", logger, f"Operation failed: {e}")
        return False

def determinarTema(texto):
    # Palabras clave para cada tema (puedes expandirlas según necesidad)
    palabras_ciberinteligencia = {"ciberinteligencia", "inteligencia", "osint", "amenazas", "estrategica", "tactica", "soc", "vigilancia"}
    palabras_ransomware = {"ransomware", "malware", "cifrado", "rescate", "eternalblue", "wannacry", "bitcoin", "secuestro"}

    # Convertir texto a minúsculas y eliminar puntuación
    texto_limpio = re.sub(r'[^\w\s]', '', texto.lower())
    palabras_texto = texto_limpio.split()

    # Contar coincidencias
    contador_ci = Counter(palabras_texto) & Counter(palabras_ciberinteligencia)
    contador_r = Counter(palabras_texto) & Counter(palabras_ransomware)

    # Calcular puntuación (similitud basada en número de palabras clave)
    similitud_ci = sum(contador_ci.values()) / len(palabras_ciberinteligencia) if palabras_ciberinteligencia else 0
    similitud_r = sum(contador_r.values()) / len(palabras_ransomware) if palabras_ransomware else 0

    # Determinar tema con mayor similitud
    if similitud_ci > similitud_r and similitud_ci > 0.1:  # Umbral mínimo de 10% para certeza
        return "ciberinteligencia", similitud_ci
    elif similitud_r > similitud_ci and similitud_r > 0.1:
        return "ransomware", similitud_r
    else:
        return "desconocido", max(similitud_ci, similitud_r)