# initial_population.py
from typing import Optional, List, Dict
import random
import json
import argparse
from pathlib import Path
import asyncio

# Agentes
from agents.llm_agent import LLMAgent
from agents.initial_prompts import obtener_prompts_agente
from agents.keywords import extraer_keywords_con_ollama

# === Conjuntos por defecto ===
ROLES_DEFAULT = [
    "a normal person",
    "an authority figure",
    "an expert",
    "a healthcare worker",
    "an organization",
    "a policymaker",
    "first responder",
    "a politician",
    "a public figure",
    "an artist",
    "a police officer",
    "an influencer",
]

TOPICS_DEFAULT = [
    "COVID-19 prevention measures",
    "mask wearing guidelines",
    "hand hygiene and sanitization",
    "social distancing practices",
    "COVID-19 testing and tracing",
    "vaccination campaign for COVID-19",
    "public health advisory on coronavirus",
    "quarantine and isolation protocols",
    "remote work and online schooling",
    "mental health during the pandemic",
    "travel restrictions and safety",
    "hospital preparedness for COVID-19",
    "community resilience during lockdowns",
    "misinformation and fake news about COVID-19",
    "economic impact of the coronavirus pandemic",
]

# === Persistencia del texto de referencia ===
REF_PATH = Path("data/reference.txt")

def guardar_referencia_txt(texto: str, ruta: Path = REF_PATH) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    # overwrite explícito
    if ruta.exists():
        ruta.unlink()
    ruta.write_text(texto, encoding="utf-8")

def cargar_texto_unico(archivo: str = "corpus_reducido_v2.txt") -> str:
    p = Path(archivo)
    if not p.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {archivo}")
    lineas = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lineas:
        raise ValueError(f"El archivo {archivo} está vacío o sin líneas válidas.")
    return random.choice(lineas)

# Función asincrona para crear un individuo
async def crear_un_individuo(
    llm_agent: 'LLMAgent',
    texto_referencia: str,
    rol_fijo: Optional[str] = None,
    topic_fijo: Optional[str] = None,
    temp_prompts: float = 0.9,
    temp_keywords: float = 0.3
) -> Optional[Dict]:
    """
    Encapsula la lógica asíncrona para crear un único individuo.
    """

    rol_i = rol_fijo or random.choice(ROLES_DEFAULT)
    topic_i = topic_fijo or random.choice(TOPICS_DEFAULT)
    
    prompts_i = await obtener_prompts_agente(
        texto_referencia=texto_referencia,
        rol=rol_i,
        task=topic_i,
        n=1,
        llm_agent=llm_agent,
        temperatura=temp_prompts,
    )
    
    if not prompts_i:
        return None
    
    p = prompts_i[0]
    kws = await extraer_keywords_con_ollama(
        prompt=p,
        topic=topic_i,
        rol=rol_i,
        llm_agent=llm_agent,
        temperatura=temp_keywords,
    )
    return construir_individuo(rol=rol_i, topic=topic_i, prompt=p, keywords=kws)

def construir_individuo(rol: str, topic: str, prompt: str, keywords: List[str]) -> Dict:
    return {
        "rol": rol,
        "topic": topic,
        "prompt": prompt,
        "keywords": keywords,
        "fitness": 0,
        "generated_data": "",
    }

def guardar_json(individuos, archivo="data.json"):
    with open(archivo, "w", encoding="utf-8") as f:
        json.dump(individuos, f, ensure_ascii=False, indent=2)

async def generar_poblacion_inicial(
    n: int,
    llm_agent: 'LLMAgent', # Recibe la instancia del agente
    rol: Optional[str] = None,
    topic: Optional[str] = None,
    texto_referencia: Optional[str] = None,
    archivo_salida: str = "data.json",
    temp_prompts: float = 0.9,
    temp_keywords: float = 0.3,
):
    """
    1) Si rol/topic no se pasan, se eligen aleatoriamente de ROLES_DEFAULT/TOPICS_DEFAULT.
       Además, si faltan, se pueden variar por cada prompt.
    2) Genera n prompts con el agente de prompts.
    3) Para cada prompt, extrae keywords con el agente de keywords.
    4) Construye y guarda la población en data.json.
    5) Guarda el texto de referencia en data/reference.txt sobrescribiendo en cada ejecución.
    """
    if not texto_referencia:
        texto_referencia = cargar_texto_unico()

    # Persistir referencia en disco, sobrescribiendo
    guardar_referencia_txt(texto_referencia)

    print(f"Generando la población inicial...")
    
    # Creamos una lista de tareas, una por cada individuo a crear
    tasks = [
        crear_un_individuo(
            llm_agent=llm_agent,
            texto_referencia=texto_referencia,
            rol_fijo=rol,
            topic_fijo=topic,
            temp_prompts=temp_prompts,
            temp_keywords=temp_keywords
        ) for _ in range(n)
    ]

    # Ejecutamos todas las tareas en paralelo
    resultados = await asyncio.gather(*tasks)
    
    # Filtramos los posibles resultados None si alguna creación falló
    individuos = [ind for ind in resultados if ind is not None]
    
    print(f"Población de {len(individuos)} individuos generada.")

    guardar_json(individuos, archivo_salida)
    return individuos


async def main():
    parser = argparse.ArgumentParser(description="Genera población inicial.")
    parser.add_argument("--n", type=int, required=True, help="Cantidad de individuos.")
    parser.add_argument("--out", default="data.json", help="Archivo JSON de salida.")
    parser.add_argument("--model", default="llama3", help="Modelo LLM a utilizar.")
    parser.add_argument("--texto-referencia", type=str, default=None, help="Texto de referencia específico a utilizar.")
    args = parser.parse_args()

    # Se crea una única instancia del agente
    llm_agent = LLMAgent(model=args.model)

    await generar_poblacion_inicial(
        n=args.n,
        llm_agent=llm_agent,
        archivo_salida=args.out,
        texto_referencia=args.texto_referencia
    )

if __name__ == "__main__":
    asyncio.run(main())
