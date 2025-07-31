from sources.common.common import logger, processControl, log_
from sources.common.utils import convert_docx_to_txt, huggingface_login
import os

"""
    Metadatos:
    - Etiqueta de clúster: "Panorámica"
    - Nombre del yacimiento: "Santuario de Némesis"
    - Título de la imagen: "Planta del Santuario de Némesis"
    - Zona del yacimiento: "Ática"
    
    Texto largo: [Inserte aquí el texto de 3000 palabras sobre el yacimiento]

Instrucciones:
Extrae del texto largo una descripción relevante para la imagen, teniendo en cuenta que es una "Panorámica" del "Santuario de Némesis" ubicado en "Ática". La descripción debe ser concisa (máximo 100 palabras) y enfocarse en los elementos visuales o contextuales que podrían aparecer en una imagen panorámica del yacimiento.
Modelo de LLaMA 2 a Utilizar
Modelo recomendado: LLaMA 2 13B (equilibrio entre rendimiento y requisitos computacionales).

Si tienes recursos limitados, puedes usar LLaMA 2 7B.

Si necesitas máxima precisión y tienes recursos suficientes, usa LLaMA 2 70B.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch
from sentence_transformers import SentenceTransformer, util
import re

def processMistral():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_("info", logger, f"Using device: {device}")

    long_text = convert_docx_to_txt(os.path.join(processControl.env['inputPath'], 'RAMNOUS.docx'))

    MODEL_NAME_LLM = "mistralai/Mistral-Nemo-Instruct-2407"
    MODEL_NAME_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"

    huggingface_login()

    embedding_model = SentenceTransformer(MODEL_NAME_EMBEDDING)
    embedding_model.to(device)  # Mover embeddings a GPU si está disponible

    metadata = {
        "cluster": "Panorámica",
        "site_name": "Santuario de Némesis",
        "image_title": "Planta del Santuario de Némesis",
        "site_zone": "Ática"
    }

    keywords = ["Némesis", "Santuario de Némesis", "Planta del Santuario de Némesis"]
    relevant_paragraphs = extract_relevant_paragraphs(embedding_model, long_text, keywords, top_n=5)

    extracted_text = "\n\n".join(relevant_paragraphs)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_LLM)
    if processControl.env['systemName'] == "tesla.informatica.uned.es":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME_LLM, torch_dtype=torch.float16, device_map="auto"
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME_LLM, torch_dtype=torch.float16, device_map="auto"
        )

    caption = generate_caption(tokenizer, model, metadata, extracted_text)
    print("\n📝 Caption Generado:\n", caption)

    prompt = f"""
    Metadatos:
    {metadata}

    Texto largo:
    {caption}

    Instrucciones:
    Extrae del texto largo una descripción relevante para la imagen, teniendo en cuenta que es una "{metadata['cluster']}" del "{metadata['site_name']}" correspondiente a "{metadata['image_title']}" ubicado en "{metadata['site_zone']}". La descripción debe ser concisa (máximo 50 palabras) y enfocarse en los elementos visuales o contextuales que podrían aparecer en la imagen indicada.
    """

    model_device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(model_device)  # Enviar a GPU
    outputs = model.generate(**inputs, max_new_tokens=300)
    descripcion = tokenizer.decode(outputs[0], skip_special_tokens=True)

    log_("info", logger, f"Descripción generada: {descripcion}")




def generate_caption(tokenizer, model, metadata, extracted_text, max_words=300):
    """
    Genera un caption usando el LLM basado en los párrafos extraídos.
    """
    prompt = f"""
    **Metadatos**:
    - 📌 Etiqueta de clúster: {metadata["cluster"]}
    - 🏛️ Nombre del yacimiento: {metadata["site_name"]}
    - 🖼️ Título de la imagen: {metadata["image_title"]}
    - 📍 Zona del yacimiento: {metadata["site_zone"]}

    **Extracto relevante**:
    {extracted_text}

    📖 **Tarea**:
    Basado en el extracto relevante, genera un caption claro y conciso (máx. {max_words} palabras).
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
    output_ids = model.generate(**inputs, max_new_tokens=400, temperature=0.7)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption


def processLlama2():
    huggingface_login()
    # Cargar el tokenizador y el modelo LLaMA 2
    model_name = "meta-llama/Llama-2-13b-chat-hf"  # Ajusta el tamaño del modelo según tus recursos

    text = ""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Metadatos y texto largo
    metadata = """
    - Etiqueta de clúster: "Panorámica"
    - Nombre del yacimiento: "Santuario de Némesis"
    - Título de la imagen: "Planta del Santuario de Némesis"
    - Zona del yacimiento: "Ática"
    """

    texto_largo = f"""
    {text}
    """

    # Crear el prompt
    prompt = f"""
    Metadatos:
    {metadata}
    
    Texto largo:
    {texto_largo}
    
    Instrucciones:
    Extrae del texto largo una descripción relevante para la imagen, teniendo en cuenta que es una "Panorámica" del "Santuario de Némesis" ubicado en "Ática". La descripción debe ser concisa (máximo 100 palabras) y enfocarse en los elementos visuales o contextuales que podrían aparecer en una imagen panorámica del yacimiento.
    """

    # Tokenizar y generar la descripción
    inputs = tokenizer(prompt, return_tensors="pt")
    #outputs = model.generate(**inputs, max_length=300)  # Ajusta max_length según sea necesario
    outputs = model.generate(**inputs, max_new_tokens=300)
    descripcion = tokenizer.decode(outputs[0], skip_special_tokens=True)

    log_("info", logger, f"Descripción generada: {descripcion}")

