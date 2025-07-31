from sources.common.common import logger, processControl, log_
from sources.dinamProcess import buildContextData
from sources.common.utils import image_parser, load_images
from sources.processFeatures import extractFeatures, assign_to_cluster
from sources.dataManager import readResults, writeResultsData
import os
import time
import json
import numpy as np

import torch
from tqdm import tqdm

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

import re

def eval_model(args, commonArgs):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(commonArgs["model_path"])
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        commonArgs["model_path"], commonArgs["model_base"], model_name
    )
    qs = args["query"]
    image_path = args["image_file"]
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    args["conv_mode"] = conv_mode

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    images = load_images([image_path])
    image_sizes = [x.size for x in images]
    #EGA get the device from model
    device = model.device
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        #EGA .cuda()
        .to(device)
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if commonArgs["temperature"]> 0 else False,
            temperature=commonArgs["temperature"],
            top_p=commonArgs["top_p"],
            num_beams=commonArgs["num_beams"],
            max_new_tokens=commonArgs["max_new_tokens"],
            use_cache=False,  #EGA disabled from True
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    #EGA elimino el print y añado return
    #print(outputs)
    #return outputs
    return {
            "image": args["image_file"],
            "prompt": args["query"],
            "answer": outputs
        }

def eval_model_batch(args_list, commonArgs):
    disable_torch_init()
    results = []

    # Load model and tokenizer once
    model_name = get_model_name_from_path(commonArgs["model_path"])
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        commonArgs["model_path"], commonArgs["model_base"], model_name
    )

    device = model.device
    batch_input_ids = []
    batch_images = []
    batch_image_sizes = []
    batch_prompts = []
    batch_metadata = []

    for args in args_list:
        qs = args["query"]
        image_path = args["image_file"]

        # Process prompt
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        # Conversation mode selection
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        args["conv_mode"] = conv_mode

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Process image
        images = load_images([image_path])  # Load as list
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            image_processor,
            model.config
        ).to(device, dtype=torch.float16)

        # Tokenize input
        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(device)
        )

        batch_input_ids.append(input_ids)
        batch_images.append(images_tensor)
        batch_image_sizes.append(image_sizes)
        batch_prompts.append(prompt)
        batch_metadata.append(args)

    # Process each image-prompt pair sequentially (if memory is limited)
    for i in tqdm(range(len(batch_input_ids)), desc="Processing Batches", unit="batch"):
        with torch.inference_mode():
            output_ids = model.generate(
                batch_input_ids[i],
                images=batch_images[i],
                image_sizes=batch_image_sizes[i],
                do_sample=True if commonArgs["temperature"] > 0 else False,
                temperature=commonArgs["temperature"],
                top_p=commonArgs["top_p"],
                num_beams=commonArgs["num_beams"],
                max_new_tokens=commonArgs["max_new_tokens"],
                use_cache=False,
            )

        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        results.append({
            "image": batch_metadata[i]["image_file"],
            "prompt": batch_metadata[i]["query"],
            "answer": output_text
        })

    return results


def buildContentProcess():
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    doc_extensions = {".doc", ".docx"}

    images_list = []
    doc_file = None

    for filename in os.listdir(processControl.env['inputPath']):
        file_path = os.path.join(processControl.env['inputPath'], filename)

        # Si es un archivo de imagen
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in image_extensions):
            # Extraer el título (nombre completo del archivo)
            title = filename

            # Extraer el nombre sin "Diapo 99.99" al inicio
            match = re.match(r"Diapo \d+\.\d+\s+(.+)", filename)
            name = match.group(1) if match else filename  # Si hay coincidencia, extraer nombre limpio
            name, _ = os.path.splitext(name)  # Eliminar la extensión

            # Si el nombre queda vacío, asignar "vacio"
            if not name.strip():
                name = "vacio"

            images_list.append({
                "image": title,
                "imagePath": file_path,
                "name": name,
                "yacimiento": "RAMNOUS",
                "zona": "ÁTICA"
            })

        # Si es un archivo .doc o .docx (tomamos el primero que encontremos)
        elif os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in doc_extensions):
            if doc_file is None:  # Solo guardamos el primer archivo .doc/docx encontrado
                doc_file = file_path

    return {"images":images_list, "doc":doc_file}


def commonVars():
    model_path = "liuhaotian/llava-v1.5-7b"
    commonArgs = {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "conv_mode": None,
        "sep": ",",
        "temperature": 0.2,  # 0
        "top_p": None,
        "num_beams": 3,  # 1
        "max_new_tokens": 256  # 512
    }
    metas = {
        "yacimiento": "RAMNOUS",
        "region": "ÁTICA"
    }
    personalization = {
        "Panorámica": ["Para esta fotografía panorámica",
                       "**Ubicación y entorno**: Describe el paisaje y el tipo de terreno",
                       "**Elemento principal yacimiento arqueológico**: Explica la estructura, su disposición y qué simboliza o representa culturalmente",
                       "es un yacimiento arqueológico en su vista general y amplia no son sólo piedras"],
        "Dibujos": ["Para este dibujo que muestra con detalle un elemento o una estructura",
                    "**Composición y contorno**: Describe su estructura, composición, apariencia",
                    "**Elemento principal**: Explica qué simboliza o representa culturalmente",
                    "es la representación de una estructura arqueológica singular y que puede estar incompleta"],
        "Detalles": ["Para esta fotografía que enfoca un detalle",
                     "**Ubicación y entorno**: Describe el entorno y cómo se ubica el elemento principal",
                     "**Elemento principal que protagoniza la imagen**: Explica la estuctura y qué simboliza o representa culturalmente el elemento principal",
                     "es un yacimiento arqueológico con su elemento principal no son sólo piedras"],
        "Diapositivas": ["Para esta fotografía de exposición",
                         "**Composición**: Describe su composición y contorno",
                         "**Elemento principal**: Explica sus características y qué simboliza o representa culturalmente",
                         "es un objeto arqueológico de valor singular"]
    }
    return commonArgs, metas, personalization


def processPrompt1(contentProcess, imageFeatures, personalization):
    processArgs = []
    for content in contentProcess["images"]:
        features = imageFeatures[content['name']]
        assigned_label, closest_cluster_idx = assign_to_cluster(features)
        contextText, keywords = buildContextData(contentProcess["doc"], content["name"], top_n=5)
        #log_("info", logger, f"Contexto generado: {contextText}")
        prompt1 = (f"{personalization[assigned_label][0]} representando a '{content['name']}', "
                   f"Ten en cuenta sin mencionar explícitamente que {personalization[assigned_label][3]} "
                   f"y realiza una descripción en castellano, cíñete a estos dos items:\n"
                   f"Item 1 {personalization[assigned_label][1]}. "
                   f"Item 2 {personalization[assigned_label][2]}. Máximo 20 palabras, no mencionar deteriorado y/o antiguo.")

        args = {
            "query": prompt1,
            "image_file": content["imagePath"],
        }
        processArgs.append(args)
    return processArgs


def processPrompt2(data):
    processArgs = []
    log_("info", logger, f"Start process LLaVA")
    for element in data:
        patron = r'Item\s*[12]:?'  # Coincide con "Item 1", "Item 1:", "Item 2", "Item 2:"
        descripInicial = re.sub(patron, '', element['answer']).strip()
        patron = r'\*\*.*?\*\*'
        descripInicial = re.sub(patron, '', descripInicial).strip()

        prompt2 = (f"Partiendo de la descripción inicial: '{descripInicial}', adopta un perfil de arqueólogo y mejórala evitando subjetividades y suposiciones, "
                  f"no menciones evidencias para un arqueólogo como que es antiguo o deteriorado")
        if element['context'] is not None:
            prompt2 += f" y asocia conceptos que esten relacionados con la imagen de este contexto: '{element['context']}'. Utiliza un máximo de 50 palabras."
        else:
            prompt2 += ". Utiliza un máximo de 20 palabras "
        prompt2 += f"y construye un texto enlazado"

        args = {
            "query": prompt2,
            "image_file": element["imagePath"],
        }
        processArgs.append(args)
    return processArgs


def processPrompt3(data):
    processArgs = []
    log_("info", logger, f"Start process LLaVA")
    for element in data:
        prompt2 = (f"Partiendo de la imagen y de su descripción inicial: '{element['answer2']}', mejora la descripción evitando inferencias, subjetividades, "
                   f"selecciona los argumentos positivos evita los argumentos de duda o pregunta como 'podría ser',..."
                f"Se trata de dar un enfoque de descripción de arqueología, no menciones evidencias para un arqueólogo como que es antiguo o deteriorado, "
                f"no menciones piedras sino restos arqueológicos y redacta un texto con continuidad")
        if element['context'] is not None:
            prompt2 += f". Utiliza un máximo de 50 palabras."
        else:
            prompt2 += ". Utiliza un máximo de 20 palabras "
        prompt2 += f"y construye un texto enlazado"

        args = {
            "query": prompt2,
            "image_file": element["imagePath"],
        }
        processArgs.append(args)
    return processArgs


def checkStage():
    data = readResults(2)
    if data:
        return 2, data
    data = readResults(1)
    if data:
        return 1, data
    return 0, None


def processLLaVA():
    commonArgs, metas, personalization = commonVars()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    stage, data = checkStage()
    log_("info", logger, f"Stage: {stage}")
    if stage == 1:
        start_time = time.time()
        processArgs = processPrompt2(data)
        results = eval_model_batch(processArgs, commonArgs)
        for idx, image_data in enumerate(data):
            for idx2, result in enumerate(results):
                if image_data["imagePath"] == result["image"]:
                    data[idx]["prompt2"] = result["prompt"]
                    data[idx]["answer2"] = f"{data[idx]['name']}, yacimiento de {data[idx]['yacimiento']} en zona de {data[idx]['zona']}. {result['answer']}"

        stage += 1
        result = sorted(data, key=lambda x: (x['label'] is None, x['label']))
        writeResultsData(result, stage)
        end_time = time.time()  # End timing
        duration = end_time - start_time
        log_("info", logger, f"Duration Process 1: {duration}")
        return True

    elif stage == 0:
        start_time = time.time()
        contentProcess = buildContentProcess()
        imageFeatures = extractFeatures(contentProcess["images"], processControl.args.featuresmodel)
        doc = contentProcess["doc"]
        for index, content in enumerate(contentProcess["images"]):
            log_('info', logger, f'processing image {content["name"]}')
            features = imageFeatures[content['name']]
            assigned_label, closest_cluster_idx = assign_to_cluster(features)
            contextText, keywords = buildContextData(doc, content["name"], top_n=5)
            contentProcess["images"][index]["label"] = assigned_label
            #contentProcess["images"][index]["clusterIDX"] = closest_cluster_idx
            contentProcess["images"][index]["context"] = contextText
            contentProcess["images"][index]["keywords"] = keywords

        end_time = time.time()  # End timing
        duration = end_time - start_time
        log_("info", logger, f"Duration Process Features-Cluster: {duration}")

        processArgs = processPrompt1(contentProcess, imageFeatures, personalization)
        results = eval_model_batch(processArgs, commonArgs)
        for idx, image_data in enumerate(contentProcess["images"]):
            for idx2, result in enumerate(results):
                if image_data["imagePath"] == result["image"]:
                    contentProcess["images"][idx]["prompt"] = result["prompt"]
                    # Elimino los patrones de respuesta indicados en el prompt
                    patron = r'Item\s*[12]:?'  # Coincide con "Item 1", "Item 1:", "Item 2", "Item 2:"
                    descripInicial = re.sub(patron, '', result['answer']).strip()
                    patron = r'\*\*.*?\*\*'
                    descripInicial = re.sub(patron, '', descripInicial).strip()
                    contentProcess["images"][idx]["answer"] = descripInicial

        stage += 1

        result = sorted(contentProcess['images'], key=lambda x: (x['label'] is None, x['label']))
        writeResultsData(result, stage)

        end_time2 = time.time()  # End timing
        duration = end_time2 - end_time
        log_("info", logger, f"Duration Process 2: {duration}")

        return True

