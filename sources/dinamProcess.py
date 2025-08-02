from sources.common.common import logger, processControl, log_

from sources.common.utils import grabaJson, dbTimestamp, clean_and_move
from sources.modelProcess import asignaClaseModelo

import os
import zipfile

def obtieneEjerciciosInstrucciones():
    input_path = processControl.env['inputPath']
    act_dir = processControl.args.act
    output_dir = os.path.join(input_path, act_dir, "ejercicios")
    os.makedirs(output_dir, exist_ok=True)

    # Detectar el único archivo ZIP en input_path
    zip_files = [f for f in os.listdir(input_path) if f.lower().endswith('.zip')]
    if len(zip_files) != 1:
        raise FileNotFoundError(f"Se esperaba exactamente 1 archivo ZIP en {input_path}, encontrados {len(zip_files)}")
    zip_path = os.path.join(input_path, zip_files[0])

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    """
    # Cargar instrucciones
    instrucciones_path = os.path.join(input_path, act_dir, f"{act_dir}PlanteamientoYSolucionario.pdf")
    if not os.path.exists(instrucciones_path):
        raise FileNotFoundError(f"No se encontraron las instrucciones en {instrucciones_path}")
    instrucciones = extraer_texto(instrucciones_path)    
    """


    # Recorrer subcarpetas (alumnos) y preparar lista de ejercicios
    ejercicios_por_alumno = []
    for alumno_dir in os.listdir(output_dir):
        alumno_path = os.path.join(output_dir, alumno_dir)
        if os.path.isdir(alumno_path):
            ejercicios = []
            for filename in os.listdir(alumno_path):
                file_path = os.path.join(alumno_path, filename)
                if filename.lower().endswith(('.pdf', '.docx', '.doc')):
                    # Renombrar a estandar
                    base_name = "parte1" if len(ejercicios) == 0 else "parte2"
                    new_name = os.path.join(alumno_path, f"{base_name}.pdf")
                    os.rename(file_path, new_name)
                    ejercicios.append(new_name)
                else:
                    raise ValueError(f"Formato no válido para {filename} en {alumno_path}")

            if len(ejercicios) != 2:
                raise ValueError(f"Se esperaban 2 ejercicios en {alumno_path}, encontrados {len(ejercicios)}")

            alumno_clean = alumno_dir.split("_")[0]
            ejercicios_por_alumno.append({
                "alumno": alumno_clean,
                "ejercicios": ejercicios
            })

    return ejercicios_por_alumno, zip_files[0], output_dir


def dinamizaProcess():
    resultados_globales = []
    inputZipPath = outputZipPath = output_dir = None
    try:
        """Preprocesa archivos y procesa los ejercicios de los alumnos"""
        ejercicios_por_alumno, zipFileName, output_dir = obtieneEjerciciosInstrucciones()

        # Instanciar evaluador
        evaluador = asignaClaseModelo()  # Asume que esta función devuelve una instancia de EvaluadorLlama3
        log_("info", logger, "\nProcesando ejercicios...")

        # Procesar lote para cada alumno
        for entry in ejercicios_por_alumno:
            alumno = entry["alumno"]
            ejercicios = entry["ejercicios"]
            resultados = evaluador.procesar_lote(ejercicios)
            resultados_globales.append({
                "alumno": alumno,
                "resultados": resultados
            })

        periodo = dbTimestamp()
        jsonPath = os.path.join(processControl.env['outputPath'], f"{processControl.args.act}_resultados-{periodo}.json")
        grabaJson(resultados_globales, jsonPath)
        inputZipPath = os.path.join(processControl.env['inputPath'], zipFileName)
        outputZipPath = os.path.join(processControl.env['outputPath'], f"{periodo}_{zipFileName}")

    except Exception as e:
        log_("exception", logger, f"Error en proceso Dinamiza {e}")


    clean_and_move(output_dir, inputZipPath, outputZipPath)
    return len(resultados_globales)

