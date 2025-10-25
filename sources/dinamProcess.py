from sources.common.common import logger, processControl, log_

from sources.common.utils import (
    grabaJson,
    dbTimestamp,
    clean_and_move,
    read_json,
    upsertAlumnoHistorico,
    extraerTextoMetas,
    texto_hash,
    existe_hash_en_otro_alumno,
    calcular_semaforos,
    determinarTema,
    extraer_texto
)
from sources.modelProcess import asignaClaseModeloEvaluacion, inicializaEntorno, EvaluadorLlama3
from sources.modelPerplexity import inicializaAItest, calcular_perplejidad

import os
import zipfile

import os
import json
import time
import gc
import torch
from contextlib import contextmanager
import threading
import queue

# helpers: ajusta si ya tienes funciones similares en tu código base
def append_jsonl(path, obj):
    """Añade un objeto JSON como línea nueva (JSONL). Crea el archivo si no existe."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_historico_index(historico_path, key_for_hash="hash"):
    """
    Construye un índice mínimo a partir del histórico en disco para detectar repetidos.
    Soporta dos formatos: JSON array o JSONL (una línea por JSON).
    Devuelve: set de hashes encontrados.
    """
    seen_hashes = set()
    if not os.path.exists(historico_path):
        return seen_hashes

    try:
        with open(historico_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if not text:
                return seen_hashes

            # Intentar JSON array completo
            if text.startswith("["):
                arr = json.loads(text)
                for item in arr:
                    if isinstance(item, dict) and key_for_hash in item:
                        seen_hashes.add(item.get(key_for_hash))
                return seen_hashes

            # Si no es array, tratar como JSONL (una línea = un JSON)
            f.seek(0)
            for line in f:
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        # hay casos donde el hash esté dentro de 'resultados' -> extraer si existe
                        if key_for_hash in obj:
                            seen_hashes.add(obj.get(key_for_hash))
                        else:
                            # buscar hashes dentro de estructura 'resultados' si aplica
                            if "resultados" in obj and isinstance(obj["resultados"], list):
                                for r in obj["resultados"]:
                                    if isinstance(r, dict) and key_for_hash in r:
                                        seen_hashes.add(r.get(key_for_hash))
                except Exception:
                    # línea no JSON válida -> ignorar
                    continue
    except Exception:
        # lectura fallida -> devolver set vacío
        return set()

    return seen_hashes

def simple_chunk_text(text, max_chars=3000):
    """
    Chunking simple por párrafos para mantener prompts debajo de un tamaño (caracteres).
    Devuelve lista de chunks. No intenta token counting; ajusta max_chars según tu tokenizador.
    """
    if not text:
        return []
    paras = [p.strip() for p in text.splitlines() if p.strip()]
    chunks = []
    cur = ""
    for p in paras:
        if len(cur) + len(p) + 1 > max_chars:
            if cur:
                chunks.append(cur.strip())
            cur = p
        else:
            cur = (cur + "\n\n" + p).strip() if cur else p
    if cur:
        chunks.append(cur.strip())
    return chunks

@contextmanager
def llama3_evaluator_context():
    """
    Context manager que instancia EvaluadorLlama3 y garantiza limpieza al salir.
    (Asume que EvaluadorLlama3() es la clase que proporcionas).
    """
    evaluador = None
    try:
        # autentica si es necesario antes de la carga (tu función huggingface_login)
        # huggingface_login(processControl.defaults['huggingface_login'])
        evaluador = asignaClaseModeloEvaluacion()
        yield evaluador
    finally:
        # intentar liberar referencias y caché CUDA
        try:
            del evaluador
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()



def obtieneEjerciciosInstrucciones():
    input_path = processControl.env['inputPath']
    act_dir = processControl.args.act
    output_dir = os.path.join(input_path, act_dir, "ejercicios")
    os.makedirs(output_dir, exist_ok=True)

    # Detectar el único archivo ZIP en input_path
    if processControl.args.unzip:
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




GPU_BATCH_SIZE = 4                # cuantos items tokenizar/generar por batch en GPU
QUEUE_MAXSIZE = 64                # tamaño máximo de la cola (control de memoria)
JSONL_HISTORICO = True            # si True escribe JSONL incremental
HISTORICO_PATH = lambda: os.path.join(processControl.env['outputPath'], f"historico_{processControl.args.act}")
INFER_MAX_NEW_TOKENS = 512
PROMPT_MAX_CHARS = 3500

def writer_thread_fn(result_q, stop_event, historico_path):
    """
    Writer que consume de result_q y persiste resultados en histórico (JSON o CSV)
    """
    historico = []  # lista acumulativa en memoria

    while not stop_event.is_set() or not result_q.empty():
        try:
            elemento = result_q.get(timeout=1)
        except queue.Empty:
            continue

        try:
            # Actualiza histórico en memoria
            historico = upsertAlumnoHistorico(historico, elemento)
            # Persistir cada iteración o cada N elementos según conveniencia
            grabaJson(historico, historico_path)

            datos = []
            if os.path.exists(processControl.env['resultPath']):
                try:
                    with open(processControl.env['resultPath'], 'r', encoding='utf-8') as archivo:
                        datos = json.load(archivo)
                except (json.JSONDecodeError, FileNotFoundError):
                    # Si el archivo está corrupto o vacío, empezar con lista vacía
                    continue

            # Agregar el nuevo elemento
            evaluacion = elemento['evaluacion']
            if (not isinstance(evaluacion, dict) or
                    not all(k in evaluacion for k in ("evaluacion", "comentario")) or
                    evaluacion.get("evaluacion", "Error") == "Error"):
                evaluacion = elemento["respuesta_modelo"]
            newElement = {
                "alumno": elemento['alumno'],
                "tema": elemento['tema'],
                "evaluacion": evaluacion,
            }
            datos.append(newElement)
            grabaJson(datos, processControl.env['resultPath'])
            log_("info", logger, f"Resultados: {elemento.get('alumno')} - {elemento.get('tema')} - {elemento.get('evaluacion').get('evaluacion')}")


        except Exception as e:
            log_("exception", logger, f"Error escribiendo histórico {elemento.get('archivo')}: {e}")

        result_q.task_done()
        # liberar memoria intermedia
        del elemento
        gc.collect()


def gpu_worker(task_q, result_q, stop_event, evaluador, instrucciones_por_tema):
    """
    Worker que consume tareas de la cola, genera respuestas en GPU y produce resultados
    """
    while not stop_event.is_set() or not task_q.empty():
        try:
            task = task_q.get(timeout=1)
        except queue.Empty:
            continue

        prompt = task['prompt']
        prompt_full = ""
        alumno = task['alumno']
        archivo = task['archivo']
        tema = task['meta']['tema']
        metas = task['meta']['metas']
        hashValue = task['meta']['hash']

        instrucciones = instrucciones_por_tema.get(tema, "")
        res_text, res_json = "", None

        try:

            res_text, res_json  = evaluador.evaluar_ejercicio_prompt(prompt, instrucciones)


        except Exception as e:
            log_("exception", logger, f"Error evaluando {archivo}: {e}")

        result = {
            "alumno": alumno,
            "archivo": archivo,
            "tema": tema,
            "hash": hashValue,
            "metas": metas,
            "respuesta_modelo": res_text,
            "evaluacion": res_json or {"evaluacion": "Error", "comentario": ""}
        }

        result_q.put(result)
        task_q.task_done()

        # liberar memoria intermedia
        del prompt, prompt_full, res_text, res_json, result
        torch.cuda.empty_cache()
        gc.collect()




def procesar_lote_pipeline(evaluador, ejercicios_por_alumno, instrucciones_por_tema):
    """
    Flujo principal: produce tareas, gpu_worker consume y writer escribe.
    Devuelve resumen compacto para grabaJson.
    """
    task_q = queue.Queue(maxsize=QUEUE_MAXSIZE)
    result_q = queue.Queue(maxsize=QUEUE_MAXSIZE)
    stop_event = threading.Event()

    historico_path = HISTORICO_PATH()

    # --- Writer thread ---
    writer_thread = threading.Thread(
        target=writer_thread_fn,
        args=(result_q, stop_event, historico_path),
        daemon=True
    )
    writer_thread.start()

    # --- GPU worker thread ---
    gpu_thread = threading.Thread(
        target=gpu_worker,
        args=(task_q, result_q, stop_event, evaluador, instrucciones_por_tema),
        daemon=True
    )
    gpu_thread.start()

    resumen_global = []

    try:
        for entry in ejercicios_por_alumno:
            alumno = entry.get("alumno")
            ejercicios = entry.get("ejercicios", [])
            resumen_alumno = {"alumno": alumno, "resultados": []}
            temaEjercicios = []
            for elemento in ejercicios:
                # extracción ligera
                extraccion = extraerTextoMetas(elemento)
                ejercicio_texto = extraccion.get("texto", "")
                metas = extraccion.get("metadatos", {})
                temaEjercicios.append({
                    "ejercicioTexto": ejercicio_texto,
                    "metas": metas,
                    "elemento": elemento,
                })
            temaEjercicios = determinarTema(temaEjercicios, processControl.args.act)

            for item in temaEjercicios:
                elemento = item["elemento"]
                ejercicio_texto = item["ejercicioTexto"]
                metas = item["metas"]
                tema = item["tema"]
                # determinar tema y hash localmente
                #tema = determinarTema(ejercicio_texto, processControl.args.act)
                hashValue = texto_hash(ejercicio_texto)

                # chunk de prompt para evitar OOM
                prompt_chunks = simple_chunk_text(ejercicio_texto, max_chars=PROMPT_MAX_CHARS)
                chosen_chunk = prompt_chunks[0] if prompt_chunks else ejercicio_texto[:PROMPT_MAX_CHARS]

                task = {
                    "alumno": alumno,
                    "archivo": elemento,
                    "prompt": chosen_chunk,
                    "meta": {"metas": metas, "hash": hashValue, "tema": tema}
                }

                task_q.put(task)

                resumen_alumno["resultados"].append({
                    "archivo": elemento,
                    "tema": tema,
                    "hash": hashValue
                })

            resumen_global.append(resumen_alumno)

        # esperar a que todos los tasks se procesen
        task_q.join()
        result_q.join()

    finally:
        stop_event.set()
        gpu_thread.join(timeout=5)
        writer_thread.join(timeout=5)
        torch.cuda.empty_cache()
        gc.collect()

    return resumen_global



def dinamizaProcess():
    """
    Flujo principal: obtiene ejercicios, inicializa evaluador y procesa todo en pipeline
    con GPU worker y writer thread.
    """
    inicializaEntorno()
    processControl.env['resultPath'] = os.path.join(processControl.env['outputPath'], f"{processControl.args.act}_resultados-{dbTimestamp()}.json")

    # --- Obtener ejercicios y paths ---
    try:
        ejercicios_por_alumno, zipFileName, output_dir = obtieneEjerciciosInstrucciones()
    except Exception as e:
        log_("exception", logger, f"Error obteniendo ejercicios: {e}")
        return 0

    # --- Preparar instrucciones por tema ---
    instrucciones_por_tema = {}
    for idx, tema in enumerate(processControl.defaults['identificador'][processControl.args.act]):
        indice = idx + 1
        instrucciones_path = os.path.join(
            processControl.env['inputPath'],
            processControl.args.act,
            f"{processControl.args.act}PlanteamientoYSolucionario{indice}.pdf"
        )
        if not os.path.exists(instrucciones_path):
            log_("exception", logger, f"No se encontraron las instrucciones en {instrucciones_path}")
            continue
        instrucciones_por_tema[tema] = extraer_texto(instrucciones_path)

    # --- Inicializar evaluador (modelo cargado una vez) ---
    evaluador = EvaluadorLlama3()

    # --- Ejecutar pipeline con threads ---
    resumen = procesar_lote_pipeline(evaluador, ejercicios_por_alumno, instrucciones_por_tema)

    # --- Mover y limpiar input/output ---
    """
    inputZipPath = os.path.join(processControl.env['inputPath'], zipFileName)
    outputZipPath = os.path.join(processControl.env['outputPath'], f"{periodo}_{zipFileName}")
    clean_and_move(output_dir, inputZipPath, outputZipPath)    
    """


    # --- Liberar modelo y memoria ---
    try:
        del evaluador
        torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass

    return len(resumen)

