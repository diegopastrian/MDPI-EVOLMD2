# test_metrics_loop.py
import time
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from metrics.fidelity import calculate_sbert_similarity, calculate_bertscore
from metrics.diversity import get_population_embeddings

# Importamos la librería SBERT directamente para controlar la carga
from sentence_transformers import SentenceTransformer, util


# frase de referencia
ref_text = "The hospital is overwhelmed with new patients, and the lack of medical supplies is compromising patient care."

# textos generados
population_data = [
    # --- FIDELIDAD ALTA / COLAPSO DE MODO ---
    "Patient care is being compromised by the shortage of medical supplies as the hospital struggles with being overwhelmed.", # Paráfrasis 1
    "A lack of medical supplies is hurting patient care because the hospital is completely overwhelmed with new cases.", # Paráfrasis 2
    "Due to the hospital being overwhelmed, a crisis in medical supplies is directly compromising the care of patients.", # Paráfrasis 3
    
    # --- FIDELIDAD MEDIA (Tópico correcto, idea diferente) ---
    "We need to send more funding to hospitals so they can buy medical supplies for future pandemics.", # Relevante, pero es una *solución*, no el *problema*.

    # --- FIDELIDAD BAJA (Irrelevante) ---
    "I saw a new TV show about doctors in a hospital, it was very dramatic.", # Irrelevante (Palabra clave "hospital")
    "The new city budget includes funding for parks and public transport, which is great." # Totalmente irrelevante
]
ref_list = [ref_text] * len(population_data)

N_GENERATIONS = 10  # Simularemos 10 generaciones del AG

print(f"--- Prueba de Costo Amortizado (Simulando {N_GENERATIONS} Generaciones) ---")

# --- 1. PRUEBA BERTScore (Carga en cada bucle) ---
print("\n--- 1. Probando BERTScore... ---")
bert_times = []
t_bert_start = time.perf_counter()

for i in range(N_GENERATIONS):
    t_gen_start = time.perf_counter()

    # calculate_bertscore tiene que cargar/gestionar el modelo CADA VEZ
    bert_scores = calculate_bertscore(population_data, ref_list, model_type="bert-base-uncased")

    t_gen_end = time.perf_counter()
    bert_times.append(t_gen_end - t_gen_start)

t_bert_total = time.perf_counter() - t_bert_start
avg_bert = sum(bert_times) / len(bert_times)

print(f"Tiempo Total (BERTScore): {t_bert_total:.4f} segundos")
print(f"Tiempo Promedio por Gen: {avg_bert:.4f} segundos")
print(f"(Scores Gen 0: {['{:.2f}'.format(s) for s in bert_scores]})")


# --- 2. PRUEBA SBERT (Carga UNA SOLA VEZ) ---
print("\n--- 2. Probando SBERT... ---")
sbert_times = []

# (A) Carga del modelo (¡FUERA DEL BUCLE!)
print("Cargando modelo SBERT (una sola vez)...")
t_sbert_load_start = time.perf_counter()
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
t_sbert_load_end = time.perf_counter()
load_time = t_sbert_load_end - t_sbert_load_start
print(f"Tiempo de Carga SBERT: {load_time:.4f} segundos")

t_sbert_start = time.perf_counter()

for i in range(N_GENERATIONS):
    t_gen_start = time.perf_counter()

    # (B) Cálculo de Embeddings (Dentro del bucle)
    pop_embeddings = sbert_model.encode(population_data, convert_to_tensor=True, normalize_embeddings=True)
    ref_embedding = sbert_model.encode([ref_text], convert_to_tensor=True, normalize_embeddings=True)

    # (C) Cálculo de Similitud (Dentro del bucle)
    sbert_scores = calculate_sbert_similarity(pop_embeddings, ref_embedding)

    t_gen_end = time.perf_counter()
    sbert_times.append(t_gen_end - t_gen_start)

t_sbert_total = time.perf_counter() - t_sbert_start
avg_sbert = sum(sbert_times) / len(sbert_times)

print(f"Tiempo Total (SBERT, solo bucle): {t_sbert_total:.4f} segundos")
print(f"Tiempo Promedio por Gen: {avg_sbert:.4f} segundos")
print(f"(Scores Gen 0: {['{:.2f}'.format(s) for s in sbert_scores]})")


# --- 3. CONCLUSIÓN ---
print("\n--- 3. Conclusión ---")
print(f"Ahorro por generación (SBERT): {avg_bert / avg_sbert:.2f}x más rápido")
print(f"Tiempo total (SBERT, incl. carga): {load_time + t_sbert_total:.4f} segundos")
print(f"Tiempo total (BERTScore): {t_bert_total:.4f} segundos")