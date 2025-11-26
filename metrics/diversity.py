# metrics/diversity.py

import torch
from sentence_transformers import SentenceTransformer
from collections import Counter
from scipy.stats import entropy
from sklearn.cluster import KMeans
import numpy as np
import spacy
from scipy.stats import entropy

NER_MODEL = None

def _load_ner_model(model_name: str = "en_core_web_sm"):
    """Carga el modelo NER de Spacy. Lo hacemos en una función para manejar errores."""
    global NER_MODEL
    if NER_MODEL is None:
        try:
            NER_MODEL = spacy.load(model_name)
        except OSError:
            print(f"ERROR: Modelo de Spacy '{model_name}' no encontrado.")
            print(f"Por favor, descárgalo ejecutando: python -m spacy download {model_name}")
            NER_MODEL = "error" # Marcar como error para no reintentar
    return NER_MODEL

# --- Métricas de Dispersión Semántica (K-Means Inertia) ---


def get_population_embeddings(generated_texts: list[str], model_name: str = 'all-MiniLM-L6-v2') -> torch.Tensor:
    """
    Convierte una lista de textos generados en un tensor de embeddings usando SBERT.
    """
    # Aseguramos que los textos no estén vacíos
    texts = [t if t.strip() else "[texto vacío]" for t in generated_texts]

    # Cargamos el modelo SBERT
    model = SentenceTransformer(model_name)

    # Codificamos los textos. 'normalize_embeddings=True' facilita los cálculos de similitud/distancia.
    embeddings = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    return embeddings

def calculate_kmeans_inertia(embeddings: torch.Tensor, k: int = 5) -> float:
    """
    Calcula la Inercia (dispersión) de un conjunto de embeddings usando K-Means.
    Un valor MÁS ALTO significa MÁS DIVERSIDAD (los clusters están más separados).

    'k' es el número de clusters. Elegir 'k' es un desafío, 
    pero para medir dispersión, un valor fijo (ej. 5) es suficiente.
    """
    n_samples = embeddings.shape[0] 
    
    if n_samples < k:
        k = n_samples

    if k == 0:
        return 0.0

    # Convertimos a numpy para scikit-learn
    embeddings_np = embeddings.cpu().numpy()

    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10) # n_init=10 para evitar malos inicios
    kmeans.fit(embeddings_np)

    total_inertia = float(kmeans.inertia_)
    
    # 2. NORMALIZACIÓN: Dividimos por N
    # Esto nos da la "distancia cuadrada promedio" por individuo
    normalized_inertia = total_inertia / n_samples
    
    return normalized_inertia

# --- Métricas de Cobertura Conceptual (Entity Entropy) ---

def calculate_entity_entropy(generated_texts: list[str]) -> float:
    """
    Calcula la Entropía Conceptual (Sustantivos, Verbos, Adjetivos) de un conjunto de textos.
    Un valor MÁS ALTO significa MÁS DIVERSIDAD (se cubren más temas).
    """
    model = _load_ner_model() # Carga el modelo spacy
    if model == "error" or model is None:
        return 0.0 

    all_concepts = []
    
    # --- MODIFICACIÓN ---
    # Ya no usamos n_process=-1 para evitar las advertencias de "fork"
    # El procesamiento en batch sigue siendo rápido.
    docs = model.pipe(generated_texts)
    
    # Tipos de palabras que queremos extraer (Part-of-Speech)
    POS_KEPT = {"NOUN", "VERB", "ADJ"} 
    
    for doc in docs:
        for token in doc:
            # En lugar de doc.ents, revisamos el Part-of-Speech (POS)
            if token.pos_ in POS_KEPT:
                # Usamos el 'lemma_' (forma base de la palabra)
                # ej: "running" -> "run", "cars" -> "car"
                all_concepts.append(token.lemma_.lower())
    
    if not all_concepts:
        # No se encontraron conceptos
        return 0.0
    
    # 2. Contar la frecuencia de cada concepto
    entity_counts = Counter(all_concepts)
    
    # 3. Calcular la entropía de Shannon
    frequencies = list(entity_counts.values())
    
    return entropy(frequencies, base=2)