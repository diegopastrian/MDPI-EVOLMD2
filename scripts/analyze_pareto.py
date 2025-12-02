import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

def load_json(filepath):
    if not filepath.exists():
        print(f"❌ Error: No se encontró el archivo {filepath}")
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def calculate_entropy_weights(matrix):
    """
    Calcula pesos objetivos usando el Método de Entropía (Shannon).
    Da más peso a los criterios que tienen mayor variabilidad.
    Ref: Sección 4.2.2 del Estado del Arte.
    """
    # 1. Normalización para Entropía (proporción sobre la suma)
    # Sumamos un epsilon pequeño para evitar log(0)
    col_sums = matrix.sum(axis=0)
    # Si una columna suma 0 (improbable), evitamos división por cero
    col_sums[col_sums == 0] = 1 
    
    p_matrix = matrix / col_sums
    
    # 2. Calcular constante k
    m = matrix.shape[0] # número de alternativas
    if m <= 1: return np.array([0.5, 0.5]) # Fallback si solo hay 1 candidato
    k = 1.0 / np.log(m)
    
    # 3. Calcular Entropía por columna
    # E_j = -k * sum(p_ij * ln(p_ij))
    p_matrix_log = np.where(p_matrix > 0, np.log(p_matrix), 0)
    entropy_vals = -k * np.sum(p_matrix * p_matrix_log, axis=0)
    
    # 4. Calcular Divergencia (Grado de Diversidad)
    div = 1.0 - entropy_vals
    
    # 5. Calcular Pesos Normalizados
    weights = div / div.sum()
    
    print(f"⚖️  Pesos Calculados (Entropy Method):")
    print(f"   - Peso Fidelidad: {weights[0]:.4f}")
    print(f"   - Peso Diversidad: {weights[1]:.4f}")
    print(f"   (ℹ️ La {['Fidelidad', 'Diversidad'][np.argmax(weights)]} tiene más peso porque varía más en tus resultados)")
    
    return weights

def calculate_topsis_scores(candidates):
    # --- 1. Preparar Matriz de Decisión ---
    # Convertimos a numpy. Aseguramos valores positivos para Entropía
    # (Sumamos 1 si hay valores negativos por la similitud coseno, aunque SBERT suele ser positivo)
    raw_data = np.array([c["objetivos"] for c in candidates])
    
    # Manejo de negativos para el cálculo de entropía (TOPSIS prefiere positivos)
    # Si hay negativos, hacemos shift para que el min sea > 0
    min_vals = raw_data.min(axis=0)
    shift = np.abs(np.minimum(min_vals, 0)) + 0.0001
    matrix_for_entropy = raw_data + shift

    # --- 2. Calcular Pesos con Entropy Method ---
    weights = calculate_entropy_weights(matrix_for_entropy)

    # --- 3. Normalización Vectorial (L2) para TOPSIS ---
    norm = np.linalg.norm(raw_data, axis=0)
    norm = np.where(norm == 0, 1, norm) 
    normalized_matrix = raw_data / norm

    # --- 4. Aplicar Pesos ---
    weighted_matrix = normalized_matrix * weights

    # --- 5. Definir Soluciones Ideal y Anti-Ideal ---
    ideal_solution = np.max(weighted_matrix, axis=0)
    anti_ideal_solution = np.min(weighted_matrix, axis=0)

    # --- 6. Calcular Distancias Euclidianas ---
    dist_to_ideal = np.sqrt(np.sum((weighted_matrix - ideal_solution)**2, axis=1))
    dist_to_anti_ideal = np.sqrt(np.sum((weighted_matrix - anti_ideal_solution)**2, axis=1))

    # --- 7. Calcular Score TOPSIS ---
    denom = dist_to_ideal + dist_to_anti_ideal
    topsis_scores = np.divide(dist_to_anti_ideal, denom, out=np.zeros_like(dist_to_anti_ideal), where=denom!=0)
    
    return topsis_scores.tolist()

def run_hybrid_selection(data, n_select=5, lambda_param=0.5, max_fidelity_threshold=0.99, min_fidelity_threshold=0.25):
    print(f"🧠 Ejecutando Fase 2: Análisis MCDM (TOPSIS + MMR)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # --- 1. Filtrado de Candidatos ---
    candidates = []
    skipped_high = 0
    skipped_low = 0
    
    for c in data:
        text = c.get("generated_data", "").strip()
        fid_score = c["objetivos"][0]
        
        if not text: continue
        
        if fid_score > max_fidelity_threshold:
            skipped_high += 1
            continue
        if fid_score < min_fidelity_threshold:
            skipped_low += 1
            continue
            
        candidates.append(c)

    print(f"📊 Filtrado: {len(candidates)} candidatos viables.")
    if skipped_high > 0: print(f"   - {skipped_high} descartados por ser copias exactas (>{max_fidelity_threshold})")
    if skipped_low > 0:  print(f"   - {skipped_low} descartados por baja calidad (<{min_fidelity_threshold})")

    if not candidates:
        print("⚠️ No quedaron candidatos viables.")
        return []

    # --- 2. Calcular Relevancia (TOPSIS con Pesos de Entropía) ---
    topsis_vals = calculate_topsis_scores(candidates)
    for i, c in enumerate(candidates):
        c["topsis_score"] = topsis_vals[i]

    # --- 3. Bucle MMR (Diversidad Final) ---
    texts = [c["generated_data"] for c in candidates]
    cand_embeddings = model.encode(texts, convert_to_tensor=True)
    selected_indices = []
    
    print("\n============================================================")
    print(f"🏆  TOP {n_select} SELECCIÓN FINAL (Entropy-TOPSIS + MMR)")
    print("============================================================")
    
    for rank in range(min(n_select, len(candidates))):
        best_mmr_score = -float('inf')
        best_idx = -1
        
        for i in range(len(candidates)):
            if i in selected_indices: continue
                
            relevance = candidates[i]["topsis_score"]
            
            if not selected_indices:
                redundancy = 0.0
            else:
                sims = util.cos_sim(cand_embeddings[i], cand_embeddings[selected_indices])
                redundancy = torch.max(sims).item()
            
            # MMR Score
            mmr_score = (lambda_param * relevance) - ((1 - lambda_param) * redundancy)
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = i
        
        if best_idx != -1:
            selected_indices.append(best_idx)
            c = candidates[best_idx]
            print(f"\n🥇 RANGO #{rank+1} | Score: {c['topsis_score']:.4f}")
            print(f"   Fidelidad: {c['objetivos'][0]:.4f} | Diversidad: {c['objetivos'][1]:.4f}")
            print(f"   Role: {c['role']}")
            print(f"   Prompt: \"{c['prompt']}\"")
            print(f"   Generado (preview): \"{c['generated_data'][:80]}...\"")

    return [candidates[i] for i in selected_indices]

def main():
    parser = argparse.ArgumentParser(description="Selección final Híbrida (TOPSIS + MMR).")
    parser.add_argument("--folder", type=str, required=True, help="Carpeta en 'exec/'")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--lambda-param", type=float, default=0.6)
    parser.add_argument("--max-fidelity", type=float, default=0.99)
    parser.add_argument("--min-fidelity", type=float, default=0.25)
    args = parser.parse_args()

    base_path = Path("exec") / args.folder
    json_path = base_path / "pareto_front.json"
    print(f"📂 Cargando: {json_path}")
    data = load_json(json_path)
    if not data: return

    final_selection = run_hybrid_selection(
        data, 
        n_select=args.top_k, 
        lambda_param=args.lambda_param,
        max_fidelity_threshold=args.max_fidelity,
        min_fidelity_threshold=args.min_fidelity
    )

    out_file = base_path / "pareto_ranked.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_selection, f, indent=2, ensure_ascii=False)
    
    print("-" * 60)
    print(f"✅ Ranking completo guardado en: {out_file}")
    print("El primer elemento de este archivo es tu prompt ganador.")

if __name__ == "__main__":
    main()