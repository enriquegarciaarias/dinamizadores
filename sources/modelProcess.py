from sources.common.common import logger, processControl, log_

from sources.common.utils import huggingface_login, extraer_texto, determinarTema, extraer_json_de_texto
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

from sentence_transformers import SentenceTransformer, util
from huggingface_hub import hf_hub_download
import time
import os
import json
import re
import gc

from llama_cpp import Llama

def getModel():
    filename="Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
    filePath = os.path.join(processControl.env['models'], filename)
    if os.path.exists(filePath):
        return filePath

    model_path = hf_hub_download(
        repo_id="bartowski/Meta-Llama-3-8B-Instruct-GGUF",
        filename=filename,
        local_dir=processControl.env['models'],  # Custom directory
        local_dir_use_symlinks=False,  # Store actual file (not symlink)
        token=processControl.defaults['huggingface_login']  # Only needed if not logged in via CLI
    )
    return model_path



class EvaluadorLlama3Local:
    def __init__(self):
        # Load GGUF model (CPU version)
        self.model_path = getModel()
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=2048,  # Context window size
            n_threads=8,  # Use all CPU cores
            n_gpu_layers=0  # 0 = CPU-only mode
        )

        # Model for semantic similarity (unchanged)
        self.sim_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def calcular_similitud(self, texto1, texto2):
        """Calcula similitud semántica entre textos (unchanged)"""
        emb1 = self.sim_model.encode(texto1, convert_to_tensor=True)
        emb2 = self.sim_model.encode(texto2, convert_to_tensor=True)
        return util.pytorch_cos_sim(emb1, emb2).item()

    def generar_respuesta(self, prompt):
        """Genera respuesta usando Llama 3 GGUF (modified for CPU)"""
        response = self.llm.create_chat_completion(
            messages=[{
                "role": "system",
                "content": "Eres un profesor asistente experto en evaluación académica de ciberseguridad."
            }, {
                "role": "user",
                "content": prompt
            }],
            max_tokens=512,
            temperature=0.7,
            top_p=0.9
        )
        return response['choices'][0]['message']['content']

    def evaluar_ejercicio(self, instrucciones, ejercicio):
        """Evalúa el ejercicio según las instrucciones (modified prompt format)"""

        summary_prompt = f"Resume este texto académico en menos de 1000 palabras:\n\n{ejercicio[:10000]}"  # First 10k chars

        summary = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=1000
        )['choices'][0]['message']['content']

        ejercicio = summary
        toEliminate = "Aquí te presento un resumen del texto académico en menos de 1000 palabras:"
        similitud = self.calcular_similitud(instrucciones, ejercicio)

        prompt = f"""
        INSTRUCCIONES:
        {instrucciones}

        EJERCICIO DEL ESTUDIANTE:
        {ejercicio}

        Analiza este ejercicio y proporciona:
        1. Evaluación (Apto/No apto)
        2. Comentario con retroalimentación específica (máx. 50 palabras)
        3. Detección de posible uso de IA (Sí/No)       
        """

        evaluacion = self.generar_respuesta(prompt)

        return {
            "evaluacion": evaluacion,
            "similitud": similitud
        }

    def procesar_lote(self, archivo_instrucciones, archivos_ejercicios):
        """Procesa múltiples ejercicios (unchanged)"""
        instrucciones = extraer_texto(archivo_instrucciones)
        resultados = []

        for archivo in archivos_ejercicios:
            try:
                ejercicio = extraer_texto(archivo)
                inicio = time.time()
                tema, similitud = determinarTema(ejercicio, processControl.args.act)
                if not tema:
                    raise Exception("Tema incorrecto in {archivo}")

                resultado = self.evaluar_ejercicio(instrucciones, ejercicio)
                tiempo = time.time() - inicio

                resultados.append({
                    "archivo": archivo,
                    "evaluacion": resultado["evaluacion"],
                    "similitud": resultado["similitud"],
                    "tiempo_procesamiento": tiempo,
                    "tema": tema,
                })

            except Exception as e:
                log_("exception", logger, f"Error procesando {archivo}: {str(e)}")
                resultados.append({
                    "archivo": archivo,
                    "error": str(e)
                })

        return resultados



def inicializaEntorno():
    huggingface_login(processControl.defaults['huggingface_login'])
    processControl.defaults['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    if processControl.defaults['device'] == "cuda":
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.cuda.set_per_process_memory_fraction(0.98, device=0)
        torch.backends.cuda.max_split_size_mb = 64
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    log_("info", logger, f"Inicializando modelos en Device: {processControl.defaults['device']}")


# Función externa para instanciar la clase (simulada)
def asignaClaseModeloEvaluacion():
    huggingface_login(processControl.defaults['huggingface_login'])
    #evaluador = EvaluadorLlama3Local()
    evaluador = EvaluadorLlama3()
    log_("info", logger, "Modelo cargado LLama3 correctamente")
    return evaluador



GPU_BATCH_SIZE = 4                # cuantos items tokenizar/generar por batch en GPU
QUEUE_MAXSIZE = 64                # tamaño máximo de la cola (control de memoria)
JSONL_HISTORICO = True            # si True escribe JSONL incremental
HISTORICO_PATH = lambda: os.path.join(processControl.env['outputPath'], f"historico_{processControl.args.act}")
INFER_MAX_NEW_TOKENS = 512
PROMPT_MAX_CHARS = 40000  # Ajustado para prompts largos

class EvaluadorLlama3:
    def __init__(self):
        self.model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantización 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.float16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            offload_folder="/tmp/llama_offload",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

        # SentenceTransformer en CPU
        self.sim_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')

        self.gen_kwargs = {
            "max_new_tokens": INFER_MAX_NEW_TOKENS,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id
        }

        # --- Cargar instrucciones por tema ---
        self.instrucciones = {}
        for idx, tema in enumerate(processControl.defaults['identificador'][processControl.args.act]):
            indice = idx + 1
            instrucciones_path = os.path.join(
                processControl.env['inputPath'],
                processControl.args.act,
                f"{processControl.args.act}PlanteamientoYSolucionario{indice}.pdf"
            )
            if os.path.exists(instrucciones_path):
                self.instrucciones[tema] = extraer_texto(instrucciones_path)
            else:
                log_("warning", logger, f"No se encontraron las instrucciones en {instrucciones_path}")

    def calcular_similitud(self, texto1, texto2):
        emb1 = self.sim_model.encode(texto1, convert_to_tensor=True)
        emb2 = self.sim_model.encode(texto2, convert_to_tensor=True)
        return util.pytorch_cos_sim(emb1, emb2).item()

    def generar_batch(self, prompts: list[str]):
        results = []
        device = next(self.model.parameters()).device

        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **self.gen_kwargs
            )

        for out in outputs:
            txt = self.tokenizer.decode(out, skip_special_tokens=True)
            try:
                parsed = extraer_json_de_texto(txt)
            except Exception:
                parsed = None
            results.append((txt, parsed))

        del input_ids, attention_mask, outputs, encoded
        torch.cuda.empty_cache()
        gc.collect()

        return results

    def evaluar_ejercicio_prompt(self, prompt, tema):
        """
        Construye prompt final usando instrucciones internas por tema.
        Retorna (texto, JSON)
        """
        instrucciones = self.instrucciones.get(tema, "")
        prompt_full = f"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
Eres un profesor asistente que evalúa ejercicios de estudiantes en ciberseguridad.
Evalúa de forma objetiva según las instrucciones dadas más abajo.
Devuelve **únicamente** un JSON con estos campos:
{{
  "evaluacion": "Apto", "No Apto" o "Sobresaliente",
  "comentario": "Si 'No Apto' expon las razones. Si 'Apto' o 'Sobresaliente' expon sugerencias de mejora relacionadas con las instrucciones. Máximo 25 palabras."
}}

Instrucciones para la evaluación:
{instrucciones}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Aquí tienes el ejercicio del estudiante que debes evaluar:

EJERCICIO DEL ESTUDIANTE:
{prompt}

---
Evalúa ahora. Devuelve únicamente un JSON **válido**, completo, que comience con `{{` y termine con `}}`.
No añadas ningún otro texto antes o después.
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
        if len(prompt_full) > PROMPT_MAX_CHARS:
            prompt_full = prompt_full[:PROMPT_MAX_CHARS]

        return self.generar_batch([prompt_full])[0]  # (texto, JSON)



