# ga.py

import random
import time
import asyncio
from pathlib import Path
from typing import Dict, Any

from agents.llm_agent import LLMAgent
from agents.generate_data import generar_data_con_ollama
from agents.regenerate_prompt import obtener_prompt_regenerado
from metrics.bert import bertscore_individuos
from operadores.crossover import crossover
from operadores.mutation import mutacion
from metrics.reports import append_metrics

async def generar_data_para_individuo(individuo, ref_text, llm_agent: 'LLMAgent', temperatura=0.7):
    out = await generar_data_con_ollama(
        individuo, texto_referencia=ref_text, llm_agent=llm_agent, temperatura=temperatura
    )
    individuo["generated_data"] = out.data.strip() if out else ""
    return individuo

async def regenerar_prompt(individuo, ref_text, llm_agent: 'LLMAgent', temperatura=0.9):
    if individuo.get("keywords"):
        prompts = await obtener_prompt_regenerado(
            texto_referencia=ref_text, role=individuo["role"], topic=individuo["topic"],
            keywords=individuo["keywords"], n=1, llm_agent=llm_agent, temperatura=temperatura
        )
        individuo["prompt"] = prompts[0] if prompts else ""
    else:
        individuo["prompt"] = ""
    return individuo

def torneo(individuos, k=3):
    candidatos = random.sample(individuos, k)
    return max(candidatos, key=lambda x: x["fitness"])

# Función asíncrona para procesar un solo hijo
async def procesar_hijo(hijo: dict, ref_text: str, llm_agent: 'LLMAgent', prob_mutacion: float):
    """
    Encapsula toda la lógica de procesamiento (mutación, prompt, data) para un individuo.
    """
    if random.random() < prob_mutacion:
        hijo = await mutacion(hijo, llm_agent) # La mutación ahora es asíncrona
    
    hijo = await regenerar_prompt(hijo, ref_text, llm_agent)
    hijo = await generar_data_para_individuo(hijo, ref_text, llm_agent)
    
    return hijo


async def metaheuristica(individuos, ref_text, llm_agent: 'LLMAgent', bert_model: str,
                   generaciones=5, k=3, prob_crossover=0.8, prob_mutacion=0.1, num_elitismo=2,
                   outdir: Path = Path(".")):
    poblacion = individuos[:]
    evo_t0 = time.perf_counter()
    for g in range(generaciones):
        gen_t0 = time.perf_counter()
        print(f"\n🌀 Generación {g+1}/{generaciones}")
        poblacion.sort(key=lambda x: x["fitness"], reverse=True)

        nueva_poblacion = poblacion[:num_elitismo]  # elitismo
        
        # Creamos una lista de hijos sin procesar
        hijos_a_procesar = []

        while len(nueva_poblacion) + len(hijos_a_procesar) < len(poblacion):
            if random.random() < prob_crossover:
                p1, p2 = torneo(poblacion, k=k), torneo(poblacion, k=k)
                hijo = crossover(p1, p2)
            else:
                hijo = torneo(poblacion, k=k).copy()
            hijos_a_procesar.append(hijo)

        # Creamos una lista de tareas asíncronas
        tasks = [
            procesar_hijo(h, ref_text, llm_agent, prob_mutacion) for h in hijos_a_procesar
        ]
        
        # Ejecutamos todas las tareas en paralelo y esperamos los resultados
        poblacion_a_evaluar = await asyncio.gather(*tasks)

        # Evaluamos a TODOS los nuevos individuos en un solo lote
        if poblacion_a_evaluar:
            poblacion_recien_evaluada = bertscore_individuos(poblacion_a_evaluar, ref_text, model_type=bert_model)
            # Añadimos los individuos ya evaluados a la nueva población.
            nueva_poblacion.extend(poblacion_recien_evaluada)

        poblacion = nueva_poblacion
        best = max(poblacion, key=lambda x: x["fitness"])
        gen_dt = time.perf_counter() - gen_t0
        print(f"   → Mejor fitness: {best['fitness']:.4f} (Duración: {gen_dt:.2f}s)")
        append_metrics(outdir, g+1, poblacion, duration_sec=gen_dt)
        
    evo_dt = time.perf_counter() - evo_t0
    with open(outdir / "runtime.tmp", "a", encoding="utf-8") as f:
        f.write(f"evolution_sec={evo_dt:.6f}\n")
    return poblacion