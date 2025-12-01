import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

def load_json(filepath):
    if not filepath.exists():
        print(f"❌ Error: No se encontró el archivo {filepath}")
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def calculate_topsis_scores(candidates):
    # --- 1. Preparar Matriz de Decisión ---
    matrix = np.array([c["objetivos"] for c in candidates])
    if len(matrix) == 0: return []

    # --- 2. Normalización Vectorial (L2) ---
    norm = np.linalg.norm(matrix, axis=0)
    norm = np.where(norm == 0, 1, norm) 
    normalized_matrix = matrix / norm

    # --- 3. Definir Soluciones Ideal y Anti-Ideal ---
    ideal_solution = np.max(normalized_matrix, axis=0)
    anti_ideal_solution = np.min(normalized_matrix, axis=0)

    # --- 4. Calcular Distancias Euclidianas ---
    dist_to_ideal = np.sqrt(np.sum((normalized_matrix - ideal_solution)**2, axis=1))
    dist_to_anti_ideal = np.sqrt(np.sum((normalized_matrix - anti_ideal_solution)**2, axis=1))

    # --- 5. Calcular Score TOPSIS ---
    denom = dist_to_ideal + dist_to_anti_ideal
    topsis_scores = np.divide(dist_to_anti_ideal, denom, out=np.zeros_like(dist_to_anti_ideal), where=denom!=0)
    
    return topsis_scores.tolist()

def run_hybrid_selection(data, n_select=5, lambda_param=0.5, max_fidelity_threshold=0.99):
    print(f"🧠 Inicializando Selección Híbrida (Lambda={lambda_param})...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # --- 1. Filtrado de Candidatos (Vacíos / Exceso Fidelidad) ---
    candidates = []
    for c in data:
        text = c.get("generated_data", "").strip()
        fid_score = c["objetivos"][0]
        if text and fid_score <= max_fidelity_threshold:
            candidates.append(c)

    if not candidates:
        print("⚠️ No quedaron candidatos viables.")
        return []

    # --- 2. Calcular Relevancia (TOPSIS) ---
    topsis_vals = calculate_topsis_scores(candidates)
    for i, c in enumerate(candidates):
        c["topsis_score"] = topsis_vals[i]

    # --- 3. Pre-cálculo Embeddings (Redundancia) ---
    texts = [c["generated_data"] for c in candidates]
    cand_embeddings = model.encode(texts, convert_to_tensor=True)
    
    selected_indices = []
    
    # --- 4. Bucle MMR (Greedy) ---
    print("🔄 Ejecutando bucle MMR sobre Scores TOPSIS...")
    
    for _ in range(min(n_select, len(candidates))):
        best_mmr_score = -float('inf')
        best_idx = -1
        
        for i in range(len(candidates)):
            if i in selected_indices: continue
                
            # MMR: Score = (λ * TOPSIS) - ((1-λ) * Similitud_Max)
            relevance = candidates[i]["topsis_score"]
            
            if not selected_indices:
                redundancy = 0.0
            else:
                sims = util.cos_sim(cand_embeddings[i], cand_embeddings[selected_indices])
                redundancy = torch.max(sims).item()
            
            mmr_score = (lambda_param * relevance) - ((1 - lambda_param) * redundancy)
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = i
        
        if best_idx != -1:
            selected_indices.append(best_idx)
            c = candidates[best_idx]
            print(f"   ✅ [MMR: {best_mmr_score:.3f}] TOPSIS: {c['topsis_score']:.3f} (Fid: {c['objetivos'][0]:.2f}, Div: {c['objetivos'][1]:.2f}) -> Role: {c['role']}")

    return [candidates[i] for i in selected_indices]

def main():
    # --- Argumentos CLI ---
    parser = argparse.ArgumentParser(description="Selección final Híbrida (TOPSIS + MMR).")
    parser.add_argument("--folder", type=str, required=True, help="Carpeta en 'exec/'")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--lambda-param", type=float, default=0.6, help="Peso del Score TOPSIS vs Redundancia en MMR")
    parser.add_argument("--max-fidelity", type=float, default=0.99)
    args = parser.parse_args()

    # --- Carga de Datos ---
    base_path = Path("exec") / args.folder
    json_path = base_path / "pareto_front.json"
    print(f"📂 Cargando: {json_path}")
    data = load_json(json_path)
    if not data: return

    # --- Ejecución ---
    final_selection = run_hybrid_selection(
        data, 
        n_select=args.top_k, 
        lambda_param=args.lambda_param,
        max_fidelity_threshold=args.max_fidelity
    )

    # --- Guardado ---
    out_file = base_path / "final_selection_hybrid.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_selection, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Selección guardada en: {out_file}")

if __name__ == "__main__":
    main()