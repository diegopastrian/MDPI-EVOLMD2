import argparse
import json
import numpy as np
from pathlib import Path
import sys

def load_json(filepath):
    if not filepath.exists():
        print(f"❌ Error: No se encontró el archivo {filepath}")
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def calculate_entropy_weights(matrix):
    """
    Calcula los pesos objetivos usando el método de Entropía de Shannon.
    Referencia: Estado del Arte, Sección 4.2.2 (Ponderación de Criterios).
    """
    # 1. Normalización para obtener probabilidades (P_ij)
    # Dividimos cada valor por la suma de su columna
    sum_cols = matrix.sum(axis=0)
    sum_cols[sum_cols == 0] = 1.0 # Evitar división por cero
    P = matrix / sum_cols

    # 2. Calcular constante k
    m, n = matrix.shape
    if m <= 1: 
        return np.ones(n) / n # Si solo hay 1 individuo, pesos iguales
    
    k = 1.0 / np.log(m)

    # 3. Calcular Entropía (E_j)
    # E = -k * sum(P * log(P))
    # Manejamos log(0) usando una máscara
    P_log_P = np.zeros_like(P)
    mask = P > 0
    P_log_P[mask] = P[mask] * np.log(P[mask])
    
    E = -k * P_log_P.sum(axis=0)

    # 4. Calcular Divergencia (d_j) y Pesos (w_j)
    d = 1 - E
    
    # Si la divergencia es 0 en todo (todos iguales), usamos pesos uniformes
    if d.sum() == 0:
        return np.ones(n) / n
        
    w = d / d.sum()
    return w

def run_topsis(data):
    """
    Ejecuta el algoritmo TOPSIS para rankear las soluciones.
    Referencia: Estado del Arte, Sección 4.2.2 (Métodos MCDM).
    """
    # Extraemos matriz [N_individuos, 2_objetivos]
    # Objetivos: [Fidelidad, Diversidad]
    raw_matrix = np.array([ind["objetivos"] for ind in data])
    
    print(f"📊 Estadísticas de la población ({len(data)} individuos):")
    print(f"   - Fidelidad (Obj 1): Min {raw_matrix[:,0].min():.4f} | Max {raw_matrix[:,0].max():.4f}")
    print(f"   - Diversidad (Obj 2): Min {raw_matrix[:,1].min():.4f} | Max {raw_matrix[:,1].max():.4f}")

    # --- PASO 1: PONDERACIÓN (ENTROPY) ---
    # Usamos Min-Max para el cálculo de entropía para evitar sesgos por escala
    min_vals = raw_matrix.min(axis=0)
    max_vals = raw_matrix.max(axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    
    # Matriz normalizada solo para calcular pesos (0 a 1)
    norm_for_entropy = (raw_matrix - min_vals) / ranges
    # Añadimos un epsilon pequeño para evitar log(0) en ceros puros
    weights = calculate_entropy_weights(norm_for_entropy + 1e-9)
    
    print(f"\n⚖️  Pesos Calculados (Entropy Method):")
    print(f"   - Peso Fidelidad: {weights[0]:.4f}")
    print(f"   - Peso Diversidad: {weights[1]:.4f}")
    if weights[1] > weights[0]:
        print("   (ℹ️ La Diversidad tiene más peso porque varía más en tus resultados)")

    # --- PASO 2: NORMALIZACIÓN VECTORIAL (TOPSIS) ---
    # r_ij = x_ij / sqrt(sum(x^2))
    denominators = np.sqrt((raw_matrix**2).sum(axis=0))
    denominators[denominators == 0] = 1.0
    norm_matrix = raw_matrix / denominators

    # --- PASO 3: MATRIZ PONDERADA ---
    weighted_matrix = norm_matrix * weights

    # --- PASO 4: SOLUCIONES IDEALES ---
    # Asumimos que AMBOS objetivos son de MAXIMIZACIÓN
    # Ideal Best (A+): El valor máximo de cada columna
    A_plus = weighted_matrix.max(axis=0)
    # Ideal Worst (A-): El valor mínimo de cada columna
    A_minus = weighted_matrix.min(axis=0)

    # --- PASO 5: DISTANCIAS EUCLIDIANAS ---
    dist_plus = np.sqrt(((weighted_matrix - A_plus)**2).sum(axis=1))
    dist_minus = np.sqrt(((weighted_matrix - A_minus)**2).sum(axis=1))

    # --- PASO 6: SCORE TOPSIS ---
    # Score = Dist_Worst / (Dist_Best + Dist_Worst)
    # Cercano a 1 es mejor.
    scores = dist_minus / (dist_plus + dist_minus)
    
    return scores, weights

def main():
    parser = argparse.ArgumentParser(description="Selección MCDM del mejor prompt usando TOPSIS y Entropy.")
    parser.add_argument("--folder", type=str, required=True, help="Nombre de la carpeta en 'exec/' (ej: 2025-11-24_14-51-56)")
    args = parser.parse_args()

    # Construir rutas
    base_path = Path("exec") / args.folder
    json_path = base_path / "pareto_front.json"

    # Cargar datos
    print(f"📂 Cargando: {json_path}")
    data = load_json(json_path)

    if not data:
        print("⚠️ El archivo está vacío.")
        return

    # Ejecutar Análisis
    scores, final_weights = run_topsis(data)

    # Asignar scores y ordenar
    for i, ind in enumerate(data):
        ind["topsis_score"] = float(scores[i])
        # Guardamos también los pesos usados para referencia futura
        ind["weights_used"] = {"fidelity": float(final_weights[0]), "diversity": float(final_weights[1])}

    # Ordenar descendente (Mayor score es mejor)
    ranked_data = sorted(data, key=lambda x: x["topsis_score"], reverse=True)

    # --- REPORTE EN CONSOLA ---
    print("\n" + "="*60)
    print("🏆  TOP 3 PROMPTS SELECCIONADOS (TOPSIS + ENTROPY)")
    print("="*60)
    
    for i, ind in enumerate(ranked_data[:3]):
        print(f"\n🥇 RANGO #{i+1} | Score: {ind['topsis_score']:.4f}")
        print(f"   Fidelidad: {ind['objetivos'][0]:.4f} | Diversidad: {ind['objetivos'][1]:.4f}")
        print(f"   Role: {ind['role']}")
        print(f"   Prompt: \"{ind['prompt']}\"")
        print(f"   Generado (preview): \"{ind['generated_data'][:80]}...\"")

    # --- GUARDAR RESULTADO ---
    out_file = base_path / "pareto_ranked.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(ranked_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "-"*60)
    print(f"✅ Ranking completo guardado en: {out_file}")
    print("El primer elemento de este archivo es tu prompt ganador.")

if __name__ == "__main__":
    main()