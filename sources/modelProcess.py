from sources.common.common import logger, processControl, log_

from sources.common.utils import huggingface_login, extraer_texto
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

from sentence_transformers import SentenceTransformer, util
from huggingface_hub import hf_hub_download
import time
import os

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
                resultado = self.evaluar_ejercicio(instrucciones, ejercicio)
                tiempo = time.time() - inicio

                resultados.append({
                    "archivo": archivo,
                    "evaluacion": resultado["evaluacion"],
                    "similitud": resultado["similitud"],
                    "tiempo_procesamiento": tiempo
                })

            except Exception as e:
                print(f"Error procesando {archivo}: {str(e)}")
                resultados.append({
                    "archivo": archivo,
                    "error": str(e)
                })

        return resultados


class EvaluadorLlama3:
    def __init__(self):
        # Cargar modelo y tokenizer
        self.model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=processControl.defaults['device'],
            torch_dtype=torch.float16
        )
        # Modelo para similitud semántica
        self.sim_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        # Configuración de generación
        self.gen_config = {
            "max_length": 2000,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
        instrucciones_path = os.path.join(
            processControl.env['inputPath'],
            processControl.args.act,
            f"{processControl.args.act}PlanteamientoYSolucionario.pdf"
        )
        if not os.path.exists(instrucciones_path):
            raise FileNotFoundError(f"No se encontraron las instrucciones en {instrucciones_path}")
        self.instrucciones = extraer_texto(instrucciones_path)


    def calcular_similitud(self, texto1, texto2):
        """Calcula similitud semántica entre textos"""
        emb1 = self.sim_model.encode(texto1, convert_to_tensor=True)
        emb2 = self.sim_model.encode(texto2, convert_to_tensor=True)
        return util.pytorch_cos_sim(emb1, emb2).item()

    def detectar_IA(self, texto, similitud):
        """Detecta si el texto parece generado por IA o copiado de chat"""
        # Umbral de similitud baja y texto largo sugieren IA
        if similitud < 0.5 and len(texto.split()) > 100:
            return "Posible uso de IA detectado por baja similitud y longitud excesiva."
        # Patrones comunes de IA (e.g., repetición, frases genéricas)
        if "en general" in texto.lower() and "sugerencias de mejora" in texto.lower():
            return "Posible uso de IA detectado por patrones genéricos."
        # Detectar copy/paste de chat (tags o frases específicas)
        chat_tags = ["<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>"]
        chat_phrases = ["user:", "assistant:", "por favor, provee", "responde solo con"]
        if any(tag in texto for tag in chat_tags) or any(phrase in texto.lower() for phrase in chat_phrases):
            return "Posible copy/paste de chat detectado por tags o frases específicas."
        return "No se detecta uso claro de IA ni copy/paste de chat."

    def generar_respuesta(self, prompt):
        """Genera respuesta usando Llama 3 local"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(processControl.defaults['device'])
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def evaluar_ejercicio(self, ejercicio):
        """Evalúa el ejercicio según las instrucciones con formato conciso"""
        # Calcular similitud semántica
        similitud = self.calcular_similitud(self.instrucciones, ejercicio)
        # Detectar posible uso de IA o copy/paste
        ia_deteccion = self.detectar_IA(ejercicio, similitud)

        # Construir prompt de evaluación conciso
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> Eres un profesor asistente. Evalúa el ejercicio comparándolo con las instrucciones. Responde solo con:
1. Evaluación: Apto/No Apto
2. Comentario: breve (máx. 30 palabras) con mejoras si Apto, razones si No Apto.
3. IA: {ia_deteccion}
INSTRUCCIONES: {self.instrucciones}
EJERCICIO DEL ESTUDIANTE: {ejercicio}
<|eot_id|><|start_header_id|>user<|end_header_id|> Proporciona la evaluación.<|eot_id|>"""

        # Generar la evaluación
        respuesta = self.generar_respuesta(prompt)
        # Extraer la sección de evaluación desde la última ocurrencia de "Evaluación:"
        evaluacion = ""
        if "Evaluación:" in respuesta:
            start_idx = respuesta.rindex("Evaluación:")  # Usa rindex para la última ocurrencia
            evaluacion = respuesta[start_idx:].strip()
        return {
            "respuesta": respuesta,
            "evaluacion": evaluacion,
            "ia": ia_deteccion,
            "similitud": similitud
        }

    def procesar_lote(self, archivos_ejercicios):
        """Procesa múltiples ejercicios"""
        #instrucciones = extraer_texto(archivo_instrucciones)
        resultados = []
        for archivo in archivos_ejercicios:
            try:
                ejercicio = extraer_texto(archivo)
                inicio = time.time()
                resultado = self.evaluar_ejercicio(ejercicio)
                tiempo_procesamiento = time.time() - inicio
                resultados.append({
                    "archivo": archivo,
                    "respuesta": resultado["respuesta"],
                    "evaluacion": resultado["evaluacion"],
                    "ia": resultado["ia"],
                    "similitud": resultado["similitud"],
                    "tiempo_procesamiento": tiempo_procesamiento
                })
            except Exception as e:
                print(f"Error procesando {archivo}: {str(e)}")
                resultados.append({
                    "archivo": archivo,
                    "error": str(e)
                })
        return resultados


# Función externa para instanciar la clase (simulada)
def asignaClaseModelo():
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

    log_("info", logger, f"Inicializando modelo Llama 3 en Device: {processControl.defaults['device']}")




    huggingface_login(processControl.defaults['huggingface_login'])
    #evaluador = EvaluadorLlama3Local()
    evaluador = EvaluadorLlama3()
    log_("info", logger, "Modelo cargado correctamente")
    return evaluador

class OLDEvaluadorLlama3:
    def __init__(self):
        # Cargar modelo y tokenizer
        self.model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=processControl.defaults['device'],
            torch_dtype=torch.float16
        )
        # Modelo para similitud semántica
        self.sim_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        # Configuración de generación
        self.gen_config = {
            "max_length": 2000,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }

    def calcular_similitud(self, texto1, texto2):
        """Calcula similitud semántica entre textos"""
        emb1 = self.sim_model.encode(texto1, convert_to_tensor=True)
        emb2 = self.sim_model.encode(texto2, convert_to_tensor=True)
        return util.pytorch_cos_sim(emb1, emb2).item()

    def detectar_IA(self, texto, similitud):
        """Detecta si el texto parece generado por IA o copiado de chat"""
        # Umbral de similitud baja y texto largo sugieren IA
        if similitud < 0.5 and len(texto.split()) > 100:
            return "Posible uso de IA detectado por baja similitud y longitud excesiva."
        # Patrones comunes de IA (e.g., repetición, frases genéricas)
        if "en general" in texto.lower() and "sugerencias de mejora" in texto.lower():
            return "Posible uso de IA detectado por patrones genéricos."
        # Detectar copy/paste de chat (tags o frases específicas)
        chat_tags = ["<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>"]
        chat_phrases = ["user:", "assistant:", "por favor, provee", "responde solo con"]
        if any(tag in texto for tag in chat_tags) or any(phrase in texto.lower() for phrase in chat_phrases):
            return "Posible copy/paste de chat detectado por tags o frases específicas."
        return "No se detecta uso claro de IA ni copy/paste de chat."

    def generar_respuesta(self, prompt):
        """Genera respuesta usando Llama 3 local"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(processControl.defaults['device'])
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def evaluar_ejercicio(self, instrucciones, ejercicio):
        """Evalúa el ejercicio según las instrucciones con formato conciso"""
        # Calcular similitud semántica
        similitud = self.calcular_similitud(instrucciones, ejercicio)
        # Detectar posible uso de IA o copy/paste
        ia_deteccion = self.detectar_IA(ejercicio, similitud)

        # Construir prompt de evaluación conciso
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> Eres un profesor asistente. Evalúa el ejercicio comparándolo con las instrucciones. Responde solo con:
1. Evaluación: Apto/No Apto
2. Comentario: breve (máx. 30 palabras) con mejoras si Apto, razones si No Apto.
3. IA: {ia_deteccion}
INSTRUCCIONES: {instrucciones}
EJERCICIO DEL ESTUDIANTE: {ejercicio}
<|eot_id|><|start_header_id|>user<|end_header_id|> Proporciona la evaluación.<|eot_id|>"""

        # Generar la evaluación
        respuesta = self.generar_respuesta(prompt)
        # Extraer la sección de evaluación si existe
        evaluacion = ""
        if "Evaluación:" in respuesta:
            start_idx = respuesta.rindex("Evaluación:")
            evaluacion = respuesta[start_idx:].strip()
        return {
            "respuesta": respuesta,
            "evaluacion": evaluacion,
            "ia": ia_deteccion,
            "similitud": similitud
        }

    def procesar_lote(self, archivo_instrucciones, archivos_ejercicios):
        """Procesa múltiples ejercicios"""
        instrucciones = extraer_texto(archivo_instrucciones)
        resultados = []
        for archivo in archivos_ejercicios:
            try:
                ejercicio = extraer_texto(archivo)
                inicio = time.time()
                resultado = self.evaluar_ejercicio(instrucciones, ejercicio)
                tiempo_procesamiento = time.time() - inicio
                resultados.append({
                    "archivo": archivo,
                    "respuesta": resultado["respuesta"],
                    "evaluacion": resultado["evaluacion"],
                    "ia": resultado["ia"],
                    "similitud": resultado["similitud"],
                    "tiempo_procesamiento": tiempo_procesamiento
                })
            except Exception as e:
                print(f"Error procesando {archivo}: {str(e)}")
                resultados.append({
                    "archivo": archivo,
                    "error": str(e)
                })
        return resultados