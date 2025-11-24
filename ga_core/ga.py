# ga_core/ga.py

import random
import time
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, List

# --- Componentes Propios (Operadores y Agentes) ---
from agents.llm_agent import LLMAgent
from agents.generate_data import generar_data_con_ollama
from agents.regenerate_prompt import obtener_prompt_regenerado
from operadores.crossover import crossover
from operadores.mutation import mutacion

# --- Métricas (Fase 1: Fidelidad y Diversidad) ---
from metrics.fidelity import calculate_sbert_similarity
from metrics.diversity import get_population_embeddings, calculate_kmeans_inertia, calculate_entity_entropy
from sentence_transformers import SentenceTransformer
import spacy

# --- Pymoo (Fase 2: Motor NSGA-II) ---
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.population import Population
from pymoo.core.individual import Individual as PymooIndividual

# ==========================================
# 1. CARGA DE MODELOS (Singleton para eficiencia)
# ==========================================
print("⚡ Cargando modelos de métricas (SBERT y Spacy)...")

# SBERT: Para fidelidad y vectorización
# Se carga una sola vez al importar el módulo
SBERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# Spacy: Para entropía conceptual
try:
    # Intentamos cargar el modelo pequeño de inglés
    SPACY_MODEL = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️ ERROR CRÍTICO: Modelo 'en_core_web_sm' no encontrado.")
    print("Ejecuta en tu terminal: python -m spacy download en_core_web_sm")
    SPACY_MODEL = None

print("✅ Modelos cargados correctamente.")


# ==========================================
# 2. FUNCIONES AUXILIARES (Ciclo de vida del Individuo)
# ==========================================

async def generar_data_para_individuo(individuo: Dict, ref_text: str, llm_agent: 'LLMAgent', temperatura=0.7) -> Dict:
    """
    Usa el LLM para generar el texto sintético basado en el prompt del individuo.
    """
    out = await generar_data_con_ollama(
        individuo, texto_referencia=ref_text, llm_agent=llm_agent, temperatura=temperatura
    )
    # Si falla el LLM, guardamos string vacío para no romper el flujo
    individuo["generated_data"] = out.data.strip() if out else ""
    return individuo

async def regenerar_prompt(individuo: Dict, ref_text: str, llm_agent: 'LLMAgent', temperatura=0.9) -> Dict:
    """
    Usa el LLM para refinar el prompt del individuo basándose en sus keywords y rol.
    """
    if individuo.get("keywords"):
        prompts = await obtener_prompt_regenerado(
            texto_referencia=ref_text, role=individuo["role"], topic=individuo["topic"],
            keywords=individuo["keywords"], n=1, llm_agent=llm_agent, temperatura=temperatura
        )
        individuo["prompt"] = prompts[0] if prompts else ""
    else:
        individuo["prompt"] = ""
    return individuo

async def procesar_hijo(hijo: Dict, ref_text: str, llm_agent: 'LLMAgent', prob_mutacion: float) -> Dict:
    """
    Ejecuta el pipeline completo de transformación de un nuevo individuo:
    1. Posible Mutación (cambio de palabras/rol).
    2. Regeneración del Prompt (coherencia).
    3. Generación de Data (resultado final).
    """
    # 1. Mutación
    if random.random() < prob_mutacion:
        hijo = await mutacion(hijo, llm_agent)
    
    # 2. Regenerar Prompt (Refleja los cambios de la mutación)
    hijo = await regenerar_prompt(hijo, ref_text, llm_agent)
    
    # 3. Generar Data (El resultado a evaluar)
    hijo = await generar_data_para_individuo(hijo, ref_text, llm_agent)
    
    return hijo


# ==========================================
# 3. EVALUACIÓN MULTIOBJETIVO
# ==========================================

def evaluar_poblacion(poblacion: List[Dict], ref_text: str) -> List[Dict]:
    """
    Calcula los objetivos para toda la población simultáneamente.
    Objetivo 1: Fidelidad (Individual) -> SBERT Similarity
    Objetivo 2: Diversidad (Poblacional) -> K-Means Inertia * Entity Entropy
    """
    if not poblacion: return []
    
    # Extraemos los textos para procesarlos en lote (batch processing)
    textos_generados = [ind["generated_data"] for ind in poblacion]
    
    # A. Vectorización (Embeddings)
    # Convertimos texto a vectores numéricos una sola vez
    pop_embeddings = SBERT_MODEL.encode(textos_generados, convert_to_tensor=True, normalize_embeddings=True)
    ref_embedding = SBERT_MODEL.encode([ref_text], convert_to_tensor=True, normalize_embeddings=True)

    # B. Cálculo Objetivo 1: Fidelidad
    # Similitud del coseno entre cada individuo y la referencia
    scores_fidelidad = calculate_sbert_similarity(pop_embeddings, ref_embedding)
    
    # C. Cálculo Objetivo 2: Diversidad
    # 1. Dispersión Semántica (Inercia)
    # Usamos k=5 como heurística estándar para detectar agrupamientos
    inercia = calculate_kmeans_inertia(pop_embeddings, k=min(5, len(poblacion)))
    
    # 2. Cobertura Conceptual (Entropía)
    if SPACY_MODEL:
        entropia = calculate_entity_entropy(textos_generados)
    else:
        entropia = 0.0
    
    # Score Combinado de Diversidad
    # Multiplicamos para asegurar que ambos factores contribuyan.
    # (entropia + 1.0) evita anular la inercia si la entropía es 0.
    score_diversidad = inercia * (entropia + 1.0)
    
    # D. Asignación de Objetivos
    for i, ind in enumerate(poblacion):
        # Guardamos los objetivos que usará NSGA-II
        ind["objetivos"] = [
            float(scores_fidelidad[i]), # Obj 1
            float(score_diversidad)     # Obj 2 (Mismo valor para toda la gen, fuerza dispersión)
        ]
        
        # Guardamos el desglose para análisis/debugging
        ind["metrics_detail"] = {
            "fidelity_sbert": float(scores_fidelidad[i]),
            "diversity_inertia": float(inercia),
            "diversity_entropy": float(entropia),
            "diversity_combined": float(score_diversidad)
        }

    return poblacion


# ==========================================
# 4. SELECCIÓN AMBIENTAL (MOTOR NSGA-II)
# ==========================================

def seleccion_nsga2(poblacion_combinada: List[Dict], n_survivors: int) -> List[Dict]:
    """
    Aplica el operador de supervivencia de NSGA-II (Rank & Crowding Distance).
    Filtra la población combinada (Padres + Hijos) para quedarse con los mejores.
    """
    # --- CLASE DUMMY PARA ENGAÑAR A PYMOO ---
    class DummyProblem:
        def has_constraints(self):
            return False # ¡No tenemos restricciones!
    # ----------------------------------------
    # 1. Crear objetos Individual de Pymoo manualmente
    pymoo_individuals = []
    for ind_dict in poblacion_combinada:
        # Crear un individuo vacío de Pymoo
        ind = PymooIndividual()
        
        # Asignar nuestros datos:
        # X: El diccionario original (para recuperarlo después)
        # F: Los objetivos (negativos porque Pymoo minimiza)
        # CV: Violación de restricciones (0.0 = válido)
        ind.X = ind_dict
        ind.F = np.array([ -ind_dict["objetivos"][0], -ind_dict["objetivos"][1] ])
        ind.CV = np.array([0.0])
        # NOTA: No asignamos ind.feasible porque es read-only y se deriva de CV.
        
        pymoo_individuals.append(ind)
    
    # 2. Crear la Población Pymoo desde la lista de individuos
    pop = Population.create(pymoo_individuals)
    
    # 3. Ejecutar Algoritmo de Supervivencia
    # RankAndCrowdingSurvival ordena por Frentes de Pareto y desempata por Distancia de Aglomeración
    survivors_pop = RankAndCrowdingSurvival().do(
        problem=DummyProblem(), 
        pop=pop, 
        n_survive=n_survivors
    )
    
    # 4. Desempaquetar y Retornar
    # Recuperamos nuestros diccionarios originales desde el atributo .X de cada individuo
    return [ind.X for ind in survivors_pop]


# ==========================================
# 5. BUCLE PRINCIPAL (METAHEURÍSTICA)
# ==========================================

async def metaheuristica(
    individuos: List[Dict], 
    ref_text: str, 
    llm_agent: 'LLMAgent', 
    bert_model: str, # (Argumento legado, no se usa, pero se mantiene por compatibilidad)
    generaciones: int = 5, 
    k: int = 3, 
    prob_crossover: float = 0.8, 
    prob_mutacion: float = 0.1, 
    num_elitismo: int = 2, # (Argumento legado, NSGA-II maneja el elitismo implícitamente)
    outdir: Path = Path(".")
) -> List[Dict]:
    
    # --- Paso 0: Evaluación Inicial ---
    print("📊 Evaluando población inicial (Multiobjetivo)...")
    poblacion = evaluar_poblacion(individuos, ref_text)
    
    evo_t0 = time.perf_counter()

    for g in range(generaciones):
        gen_t0 = time.perf_counter()
        print(f"\n🌀 Generación {g+1}/{generaciones}")
        
        # --- Paso 1: Selección de Padres y Reproducción ---
        # En NSGA-II estándar, el tamaño de la descendencia (offspring) suele ser igual N.
        padres = poblacion 
        hijos_a_procesar = []
        
        while len(hijos_a_procesar) < len(poblacion):
            # Selección aleatoria para cruce (la presión selectiva viene al final en NSGA-II)
            p1 = random.choice(padres)
            p2 = random.choice(padres)
            
            if random.random() < prob_crossover:
                hijo = crossover(p1, p2)
            else:
                hijo = p1.copy()
            
            hijos_a_procesar.append(hijo)
            
        # --- Paso 2: Procesamiento de Hijos (Paralelo) ---
        # Aquí ocurren las llamadas al LLM (Mutación, Prompt, Data)
        tasks = [procesar_hijo(h, ref_text, llm_agent, prob_mutacion) for h in hijos_a_procesar]
        offspring = await asyncio.gather(*tasks)
        
        # --- Paso 3: Evaluación de Hijos ---
        if offspring:
            offspring = evaluar_poblacion(offspring, ref_text)
            
        # --- Paso 4: Unión (Merge) ---
        # Combinamos Padres (P_t) + Hijos (Q_t) = R_t
        poblacion_combinada = poblacion + offspring
        
        # --- Paso 5: Selección Ambiental (NSGA-II) ---
        # De R_t seleccionamos los N mejores para formar P_{t+1}
        poblacion = seleccion_nsga2(poblacion_combinada, n_survivors=len(individuos))
        
        # --- Reporte de Progreso ---
        # Mostramos los mejores individuos de cada objetivo para monitorear
        best_fid = max(poblacion, key=lambda x: x["objetivos"][0])
        best_div = max(poblacion, key=lambda x: x["objetivos"][1])
        
        gen_dt = time.perf_counter() - gen_t0
        print(f"   → Max Fidelidad: {best_fid['objetivos'][0]:.4f}")
        print(f"   → Max Diversidad: {best_div['objetivos'][1]:.4f}")
        print(f"   → Tiempo Gen: {gen_dt:.2f}s")

    evo_dt = time.perf_counter() - evo_t0
    
    # Guardar tiempo total
    with open(outdir / "runtime.tmp", "a", encoding="utf-8") as f:
        f.write(f"evolution_sec={evo_dt:.6f}\n")
        
    # La población final contiene el Frente de Pareto aproximado
    return poblacion