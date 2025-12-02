import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# Configuración de ruta relativa para imports si es necesario
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

def load_json(filepath):
    if not filepath.exists():
        print(f"❌ Error: No se encontró {filepath}")
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def calculate_entropy_weights(matrix):
    """Calcula pesos objetivos (Entropy Method)."""
    col_sums = matrix.sum(axis=0)
    col_sums[col_sums == 0] = 1 
    p_matrix = matrix / col_sums
    
    m = matrix.shape[0]
    if m <= 1: return np.array([0.5, 0.5])
    k = 1.0 / np.log(m)
    
    p_matrix_log = np.where(p_matrix > 0, np.log(p_matrix), 0)
    entropy_vals = -k * np.sum(p_matrix * p_matrix_log, axis=0)
    div = 1.0 - entropy_vals
    weights = div / div.sum()
    
    # Print compacto
    print(f"  Pesos Entropy: Fidelidad={weights[0]:.3f} | Diversidad={weights[1]:.3f}")
    return weights

def calculate_topsis_scores(candidates):
    raw_data = np.array([c["objetivos"] for c in candidates])
    
    # Ajuste para valores negativos antes de entropía
    min_vals = raw_data.min(axis=0)
    shift = np.abs(np.minimum(min_vals, 0)) + 0.0001
    matrix_for_entropy = raw_data + shift

    weights = calculate_entropy_weights(matrix_for_entropy)

    norm = np.linalg.norm(raw_data, axis=0)
    norm = np.where(norm == 0, 1, norm) 
    normalized_matrix = raw_data / norm
    weighted_matrix = normalized_matrix * weights

    ideal_solution = np.max(weighted_matrix, axis=0)
    anti_ideal_solution = np.min(weighted_matrix, axis=0)

    dist_to_ideal = np.sqrt(np.sum((weighted_matrix - ideal_solution)**2, axis=1))
    dist_to_anti_ideal = np.sqrt(np.sum((weighted_matrix - anti_ideal_solution)**2, axis=1))

    denom = dist_to_ideal + dist_to_anti_ideal
    topsis_scores = np.divide(dist_to_anti_ideal, denom, out=np.zeros_like(dist_to_anti_ideal), where=denom!=0)
    
    return topsis_scores.tolist()

def run_hybrid_selection(data, n_select=5, lambda_param=0.5, max_fidelity_threshold=0.99, min_fidelity_threshold=0.25):
    print(f" Análisis MCDM (TOPSIS + MMR)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 1. Filtrado
    candidates = []
    skipped = 0
    
    for c in data:
        text = c.get("generated_data", "").strip()
        fid_score = c["objetivos"][0]
        
        if not text: continue
        if fid_score > max_fidelity_threshold or fid_score < min_fidelity_threshold:
            skipped += 1
            continue
            
        candidates.append(c)

    print(f" Candidatos viables: {len(candidates)} (Filtrados: {skipped})")

    if not candidates:
        print(" No quedaron candidatos viables.")
        return []

    # 2. TOPSIS
    topsis_vals = calculate_topsis_scores(candidates)
    for i, c in enumerate(candidates):
        c["topsis_score"] = topsis_vals[i]

    # 3. MMR Loop
    texts = [c["generated_data"] for c in candidates]
    cand_embeddings = model.encode(texts, convert_to_tensor=True)
    selected_indices = []
    
    print(f"\n TOP {n_select} SELECCIÓN:")
    
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
            
            mmr_score = (lambda_param * relevance) - ((1 - lambda_param) * redundancy)
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = i
        
        if best_idx != -1:
            selected_indices.append(best_idx)
            c = candidates[best_idx]
            # Print compacto de resultados
            print(f" #{rank+1} [Score:{c['topsis_score']:.3f}] Role: {c['role']} | Fid:{c['objetivos'][0]:.2f}/Div:{c['objetivos'][1]:.2f}")
            print(f"     Preview: \"{c['generated_data'][:80]}...\"")

    return [candidates[i] for i in selected_indices]

def main():
    parser = argparse.ArgumentParser(description="Selección final Híbrida (TOPSIS + MMR).")
    parser.add_argument("--folder", type=str, required=True, help="Carpeta en 'exec/'")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--lambda-param", type=float, default=0.6)
    parser.add_argument("--max-fidelity", type=float, default=0.99)
    parser.add_argument("--min-fidelity", type=float, default=0.25)
    args = parser.parse_args()

    base_path = Path("exec") / args.folder
    json_path = base_path / "pareto_front.json"
    
    print(f" Procesando: {args.folder}")
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
    
    print(f"✅ Guardado en: {out_file.name}")

if __name__ == "__main__":
    main()