from sources.common.common import logger, processControl, log_

from sources.common.utils import (
    grabaJson,
    dbTimestamp,
    clean_and_move,
    read_json,
    upsertAlumnoHistorico,
    extraer_texto_y_metas,
    texto_hash,
    existe_hash_en_otro_alumno,
    calcular_semaforos,
)
from sources.modelProcess import asignaClaseModeloEvaluacion, inicializaEntorno
from sources.modelPerplexity import inicializaAItest, calcular_perplejidad

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
                    original_extension = os.path.splitext(filename)[1].lower()  # Obtiene la extensión original
                    new_name = os.path.join(alumno_path, f"{base_name}{original_extension}")
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
    historicoPath = os.path.join(processControl.env['outputPath'], f"historico_{processControl.args.act}")
    historicoResults = read_json(historicoPath)
    inputZipPath = outputZipPath = output_dir = None
    inicializaEntorno()

    #Primer paso para evaluación
    try:
        """Preprocesa archivos y procesa los ejercicios de los alumnos"""
        ejercicios_por_alumno, zipFileName, output_dir = obtieneEjerciciosInstrucciones()

        # Instanciar evaluador
        evaluador = asignaClaseModeloEvaluacion()  # Asume que esta función devuelve una instancia de EvaluadorLlama3
        log_("info", logger, "\nProcesando ejercicios...")

        # Procesar lote para cada alumno
        for entry in ejercicios_por_alumno:
            alumno = entry["alumno"]
            log_("info", logger, f"Procesando ejercicios {alumno}")
            ejercicios = entry["ejercicios"]
            resultados = evaluador.procesar_lote(ejercicios)
            resultados_globales.append({
                "alumno": alumno,
                "resultados": resultados
            })


            elemento = {
                "alumno": alumno,
                "resultados": resultados,

            }
            historicoResults = upsertAlumnoHistorico(historicoResults, elemento)



    except Exception as e:
        log_("exception", logger, f"Error en proceso Dinamiza {e}")

    try:
        model, tokenizer = inicializaAItest()
        for entry in resultados_globales:
            alumno = entry["alumno"]
            resultados = entry["resultados"]

            newElement = {
                "alumno": alumno,
                "resultados": resultados,
            }

            for idx, elemento in enumerate(resultados):
                extraccion = extraer_texto_y_metas(elemento['archivo']) # usa "general" si no existe
                ejercicio = extraccion['texto']
                metas = extraccion['metadatos']
                semaforos = calcular_semaforos(metas)
                log_("info", logger, f"Semáforos {semaforos}")

                hashValue = texto_hash(ejercicio)
                repetido = existe_hash_en_otro_alumno(historicoResults, alumno, hashValue)
                if repetido:
                    log_("info", logger, f"Se encontrado repetido {repetido}")
                perplexity = calcular_perplejidad(ejercicio, model, tokenizer)
                perplexity = round(perplexity, 2)
                log_("debug", logger, f"Perplexity {alumno}-{perplexity}")
                newElement["resultados"][idx].setdefault("perplexity", None)
                newElement["resultados"][idx].setdefault("hash", None)
                newElement["resultados"][idx].setdefault("metas", None)
                newElement['resultados'][idx]['perplexity'] = perplexity
                newElement['resultados'][idx]['hash'] = hashValue
                newElement['resultados'][idx]['metas'] = metas

            historicoResults = upsertAlumnoHistorico(historicoResults, newElement)

    except Exception as e:
        log_("exception", logger, f"Error en proceso Dinamiza AI tester {e}")

    periodo = dbTimestamp()
    jsonPath = os.path.join(processControl.env['outputPath'], f"{processControl.args.act}_resultados-{periodo}.json")
    grabaJson(resultados_globales, jsonPath)
    grabaJson(historicoResults, historicoPath)
    inputZipPath = os.path.join(processControl.env['inputPath'], zipFileName)
    outputZipPath = os.path.join(processControl.env['outputPath'], f"{periodo}_{zipFileName}")
    clean_and_move(output_dir, inputZipPath, outputZipPath)
    return len(resultados_globales)

