from sources.common.common import logger, processControl, log_
from sources.dataManager import readResults, writeResultsData

from sentence_transformers import SentenceTransformer, util
import re
import math
from collections import Counter


def calculaSimilitudSemantica(element, model):
    if not element['context']:
        return 0

    embedding_ref = model.encode(element["context"], convert_to_tensor=True)
    embedding_cmp = model.encode(element["answer2"], convert_to_tensor=True)

    # Calcular similitud coseno
    similitud = util.pytorch_cos_sim(embedding_ref, embedding_cmp)
    return round(similitud.item(), 3)
    # Mostrar resultado
    #print(f"Similitud semántica: {similitud.item():.4f}")

def calcular_entropia(tokens):
    counter = Counter(tokens)
    total_tokens = len(tokens)
    probs = [count / total_tokens for count in counter.values()]
    return -sum(p * math.log2(p) for p in probs)

def evaluaRiquezaLexica(element, nlp):
    # Procesar los textos con spaCy
    # Elimino la primera parte compuesta del tipo "Reconstrucción de la casa oeste, yacimiento de RAMNOUS en zona de ÁTICA." para no interferir en el resultado del LLM
    # De hecho produciria repeticion de terminos clave reduciendo la entropia (variedad léxica)
    patron = r"^[^.]*\."
    answer2 = re.sub(patron, "", element['answer2']).strip()

    doc_con = nlp(answer2)
    doc_sin = nlp(element['answer'])

    # Obtener solo las palabras (excluyendo puntuación y espacios)
    tokens_sin = [token.text.lower() for token in doc_sin if token.is_alpha]
    tokens_con = [token.text.lower() for token in doc_con if token.is_alpha]

    #Entropia lexica (riqueza)
    entropia_sin = calcular_entropia(tokens_sin)
    entropia_con = calcular_entropia(tokens_con)

    # Total de palabras en el texto
    total_palabras_sin = len(tokens_sin)
    total_palabras_con = len(tokens_con)

    # Número total de entidades detectadas
    num_entidades_sin = len(doc_sin.ents)
    num_entidades_con = len(doc_con.ents)

    # Probabilidad de encontrar entidades en el texto (Relevancia semantica)
    prob_sin = round(num_entidades_sin / total_palabras_sin, 3) if total_palabras_sin > 0 else 0
    prob_con = round(num_entidades_con / total_palabras_con, 3) if total_palabras_con > 0 else 0

    # Diversidad léxica (TTR) ( no lo uso)
    diversidadTTR_sin = round(len(set(tokens_sin)) / total_palabras_sin, 3) if total_palabras_sin > 0 else 0
    diversidadTTR_con = round(len(set(tokens_con)) / total_palabras_con, 3) if total_palabras_con > 0 else 0

    return prob_sin, prob_con, entropia_sin , entropia_con

def gptVSproy():
    data = [
        {
            'img':"Diapo 13.69 Planta del Anfiareio",
            'context': "Casas parcelas construcciones agrícolas pozos vallas y almacenes daban otro aspecto al lugar Quizás el Amfiareio se utilizase como hospital para los heridos de la fortaleza El Anfiareio era en un principio un santuario medicinal y ctónico del héroe médico Aristómaco como sabemos por las epigrafías y las fuentes filológicas con la cabeza de Apolo saliendo de su útero Este conjunto escultórico uno de los más bellos del Ática se encuentra en el Museo Arqueológico Nacional de Atenas.",
            'answer': "El plano representa la planta del Amfiareio de Ramnous, un santuario dedicado a Amphiaraos. Se observan dos edificios principales con muros gruesos, conectados por una zona abierta con elementos dispersos. Incluye detalles arquitectónicos como accesos y estructuras internas, reflejando la disposición del espacio dentro del contexto religioso del sitio.",
            'answer2': "Planta del Amfiareio, yacimiento de RAMNOUS en zona de ÁTICA. El dibujo representa una estructura arqueológica singular, posiblemente incompleta, que podría ser parte de un amfiteatro romano. La estructura tiene una forma de planta del amfiteatro, con un contorno definido por líneas rectas y curvas. El elemento principal es la base de la estructura, que podría haber sido utilizada para albergar a los espectadores. La composición y apariencia del dibujo son detalladas, lo que permite una comprensión clara de la estructura representada. Casas, parcelas, construcciones agrícolas, pozos, vallas y almacenes daban otro aspecto al lugar. Quizás el Amfiareio se utilizase como hospital para los heridos de la fortaleza. El Anfiareio era en un principio un santuario medicinal y ctónico del héroe médico Aristómaco, como sabemos por las epigrafías y las fuentes filológicas. Con la cabeza de Apolo saliendo de su útero, este conjunto escultórico uno de los más bellos del Ática",

        },
        {
            'img': "Diapo 13.36 Templete del recinto de Diogiton",
            'context': "",
            'answer': "El templete pertenece al recinto de Diogiton en Ramnous. Su estructura rectangular con columnas estriadas y entablamento decorado enmarca una escena esculpida en altorrelieve, donde una figura femenina drapeada interactúa con un niño. Representa un contexto votivo o funerario, destacando la tradición escultórica y arquitectónica del santuario.",
            'answer2': "Templete del recinto de Diogiton, yacimiento de RAMNOUS en zona de ÁTICA. La escultura representa a un hombre y una niña, con la niña sosteniendo un libro. La escultura está hecha de piedra y tiene un contorno elegante.",

        },
        {
            'img': "Diapo 13.68 El Anfiareio",
            'context': "El Anfiareio era en un principio un santuario medicinal y ctónico del héroe médico Aristómaco como sabemos por las epigrafías y las fuentes filológicas El pequeño santuario de Anfiarao se encuentra al SO de la puerta principal de entrada a la fortaleza encima de las rocas de la colina.",
            'answer': "La imagen muestra una estructura de grandes bloques de piedra dispuestos en forma de muro, con signos de erosión y posibles restos de colapso. El entorno es montañoso y seco, con vegetación baja. En el documento, se menciona que Ramnous albergaba fortificaciones y templos, lo que sugiere que esta estructura podría formar parte de un sitio arqueológico de defensa o culto.",
            'answer2': "El Anfiareio, yacimiento de RAMNOUS en zona de ÁTICA. El elemento principal se encuentra en un yacimiento arqueológico, rodeado de otras piedras y rocas. El Anfiareio era en un principio un santuario medicinal y ctónico del héroe médico Aristómaco como sabemos por las epigrafías y las fuentes filológicas. El pequeño santuario de Anfiarao se encuentra al SO de la puerta principal de entrada a la fortaleza encima de las rocas de la colina.",

        }

    ]
    return data

def procesoEvaluacion():
    data = gptVSproy()
    import spacy
    nlp = spacy.load("es_core_news_md")
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    #data = readResults(2)
    if not data:
        log_("info", logger, "No results")
        return True
    result = []

    for idx, element in enumerate(data):
        newElement = element.copy()
        newElement['similitudContext'] = calculaSimilitudSemantica(element, model)
        prob_sin, prob_con, diversidadTTR_sin, diversidadTTR_con = evaluaRiquezaLexica(element, nlp)
        newElement['probSin'] = prob_sin
        newElement['probCon'] = prob_con
        newElement['diversidadTTR_sin'] = diversidadTTR_sin
        newElement['diversidadTTR_con'] = diversidadTTR_con

        result.append(newElement)

    writeResultsData(result, 3)
