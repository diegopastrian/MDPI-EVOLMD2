# operadores/mutacion_nico.py
from typing import List, Dict
import random
import nltk
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agents'))
from synonym_selection import seleccionar_sinonimo

def procesar_synsets_wordnet(palabra: str) -> List[str]:
    """
    Procesa los synsets de WordNet para una palabra y los convierte
    en una lista plana de sinónimos únicos (lemmas).
    """
    synsets = wn.synsets(palabra)
    sinonimos_unicos = set()
    
    for syn in synsets:
        for lemma in syn.lemmas():
            lemma_name = lemma.name().replace('_', ' ')
            if lemma_name.lower() != palabra.lower():
                sinonimos_unicos.add(lemma_name)
    
    return list(sinonimos_unicos)


async def obtener_sinonimo(palabra: str, individuo: Dict, parametro_a_mutar: str, llm_agent: 'LLMAgent') -> str:
    """
    Obtiene un sinónimo de una palabra usando un agente de selección.
    Ahora recibe la instancia de llm_agent.
    """
    pos_tags = pos_tag([palabra])
    word, pos = pos_tags[0]
    
    tipos_mutables = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS'}
    
    if pos not in tipos_mutables:
        return palabra
    
    sinonimos_disponibles = procesar_synsets_wordnet(palabra)
    
    if not sinonimos_disponibles:
        return palabra
    
    try:
        resultado = await seleccionar_sinonimo(
            individuo=individuo,
            parametro_a_mutar=parametro_a_mutar,
            palabra_a_sustituir=palabra,
            sinonimos_disponibles=sinonimos_disponibles,
            llm_agent=llm_agent # Se pasa el agente
        )
        
        if resultado:
            return resultado.selected_synonym
        else:
            return palabra
            
    except Exception as e:
        print(f"Error en el agente de selección: {e}")
        return palabra



async def mutar_keywords(keywords: List[str], individuo: Dict, llm_agent: 'LLMAgent') -> List[str]:
    """
    Muta keywords y pasa el llm_agent a obtener_sinonimo.
    """
    if not keywords: return keywords
    
    num_a_mutar = random.randint(1, len(keywords))
    indices_a_mutar = random.sample(range(len(keywords)), num_a_mutar)
    
    nuevas_keywords = keywords.copy()
    for idx in indices_a_mutar:
        sinonimo = await obtener_sinonimo(keywords[idx], individuo, "keywords", llm_agent)
        nuevas_keywords[idx] = sinonimo
    
    return nuevas_keywords


async def mutar_texto(texto: str, individuo: Dict, parametro: str, llm_agent: 'LLMAgent') -> str:
    """
    Muta un texto y pasa el llm_agent a obtener_sinonimo.
    """
    if not texto: return texto
    
    palabras = texto.split()
    if not palabras: return texto
    
    num_a_mutar = random.randint(1, len(palabras))
    indices_a_mutar = random.sample(range(len(palabras)), num_a_mutar)
    
    nuevas_palabras = palabras.copy()
    for idx in indices_a_mutar:
        sinonimo = await obtener_sinonimo(palabras[idx], individuo, parametro, llm_agent)
        nuevas_palabras[idx] = sinonimo
    
    return ' '.join(nuevas_palabras)


async def mutacion(individuo: Dict, llm_agent: 'LLMAgent') -> Dict:
    """
    Realiza mutación y pasa la instancia del llm_agent a las sub-funciones.
    """
    individuo_mutado = individuo.copy()
    parametros = ["rol", "topic", "keywords"]
    
    p = random.randint(1, len(parametros))
    parametros_a_mutar = random.sample(parametros, p)
      
    for parametro in parametros_a_mutar:
        if parametro == "rol":
            individuo_mutado["rol"] = await mutar_texto(individuo["rol"], individuo, "rol", llm_agent)
        elif parametro == "topic":
            individuo_mutado["topic"] = await mutar_texto(individuo["topic"], individuo, "topic", llm_agent)
        elif parametro == "keywords":
            individuo_mutado["keywords"] = await mutar_keywords(individuo["keywords"], individuo, llm_agent)
    
    individuo_mutado["prompt"] = ""
    individuo_mutado["generated_data"] = ""
    individuo_mutado["fitness"] = 0
    
    return individuo_mutado


'''def crear_individuo_ejemplo(rol: str, topic: str, keywords: List[str], fitness: float = 0.0) -> Dict:
    """Crea un individuo de ejemplo"""
    return {
        "rol": rol,
        "topic": topic,
        "prompt": f"Example prompt for {rol} about {topic}",
        "keywords": keywords,
        "generated_data": "",
        "fitness": fitness,
    }


def mostrar_individuo(individuo: Dict, nombre: str):
    """Muestra la información de un individuo"""
    print(f"{nombre}:")
    print(f"  Rol: {individuo['rol']}")
    print(f"  Topic: {individuo['topic']}")
    print(f"  Keywords: {individuo['keywords']}")
    print(f"  Prompt: {individuo['prompt']}")
    print(f"  Generated Data: {individuo['generated_data']}")
    print(f"  Fitness: {individuo['fitness']}")


def test():
    individuo = crear_individuo_ejemplo(
        rol="an authority figure",
        topic="mask wearing guidelines",
        keywords=["mask", "guidelines", "authority", "figure", "wearing", "protection"],
        fitness=0.85,
    )

    mostrar_individuo(individuo, "Individuo No Mutado")
    print("=" * 60)
    print("Aplicar Mutación:")
    print("=" * 60)
    individuo_mutado = mutacion(individuo)
    mostrar_individuo(individuo_mutado, "Individuo Mutado")

if __name__ == "__main__":
    test()'''