import argparse
import json
import sys
from pathlib import Path
# Importamos la librería para usar el modelo NLI
# Asegúrate de tener instalado: pip install sentence-transformers
from sentence_transformers import CrossEncoder 

def load_json(filepath):
    """Carga datos desde un JSON."""
    if not filepath.exists():
        print(f"❌ Error: No se encontró el archivo {filepath}")
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def run_greedy_nli_selection(data, n_select=5):
    """
    Algoritmo de Selección Voraz con Veto por NLI.
    
    Lógica:
    1. Ordena todos los candidatos por FIDELIDAD (Calidad SBERT) de mayor a menor.
    2. Recorre la lista y selecciona el candidato SOLO SI no es redundante (Entailment)
       con ninguno de los que ya han sido seleccionados.
    """
    
    # --- PASO 1: ORDENAMIENTO MAESTRO (CALIDAD) ---
    # Ordenamos de Mayor a Menor Fidelidad (Objetivo 0).
    # Ignoramos la "Diversidad" histórica (Objetivo 1) que venía del algoritmo genético.
    candidates = sorted(data, key=lambda x: x["objetivos"][0], reverse=True)
    
    print(f"📊 Total candidatos evaluados: {len(candidates)}")
    print("🔄 Iniciando proceso de selección (Filtro NLI)...")

    # --- PASO 2: CARGA DEL MODELO NLI ---
    # Usamos DeBERTa-v3, un modelo estado del arte para detectar implicancia lógica.
    model_id = 'cross-encoder/nli-deberta-v3-base'
    print(f"🧠 Cargando modelo de validación: {model_id}...")
    try:
        model = CrossEncoder(model_id)
    except Exception as e:
        print(f"❌ Error cargando el modelo NLI. Asegúrate de tener internet o el modelo en caché.\nError: {e}")
        sys.exit(1)
    
    # Mapeo de labels para este modelo específico:
    # 0: Contradiction, 1: Entailment (Implicación/Redundancia), 2: Neutral
    LABEL_ENTAILMENT = 1 
    
    selected_ind = []

    # --- PASO 3: BUCLE VORAZ (GREEDY LOOP) ---
    for i, cand in enumerate(candidates):
        # Condición de parada: Si ya tenemos los k deseados, terminamos.
        if len(selected_ind) >= n_select:
            break
        
        text_cand = cand["generated_data"]
        
        if cand["objetivos"][0] > 0.99:
            continue

        # A. EL PRIMERO SIEMPRE ENTRA
        # Como están ordenados por fidelidad, el primero es el "Mejor Absoluto".
        if not selected_ind:
            selected_ind.append(cand)
            print(f"   ✅ [1/{n_select}] Seleccionado (Mejor Calidad): \"{text_cand[:60]}...\"")
            continue
        
        # B. COMPARACIÓN "TODOS CONTRA TODOS" (VETO)
        # Preparamos los pares para comparar el candidato actual contra TODOS los ya elegidos
        # Formato: [(Texto Nuevo, Texto Elegido 1), (Texto Nuevo, Texto Elegido 2)...]
        pairs_to_check = [(text_cand, sel["generated_data"]) for sel in selected_ind]
        
        # El modelo predice para todos los pares a la vez (Batch processing)
        scores = model.predict(pairs_to_check)
        predicted_labels = scores.argmax(axis=1) # Obtiene el índice de la clase más probable
        
        # C. DECISIÓN DE VETO
        # Si CUALQUIERA de las comparaciones da "Entailment" (1), significa que el candidato
        # es una repetición semántica de algo que ya tenemos.
        if LABEL_ENTAILMENT in predicted_labels:
            # Es redundante. Lo descartamos silenciosamente y pasamos al siguiente.
            continue
        
        # Si sobrevive a todas las comparaciones (es decir, es Neutral o Contradictorio con todos) -> ENTRA
        selected_ind.append(cand)
        print(f"   ✅ [{len(selected_ind)}/{n_select}] Seleccionado (Único): \"{text_cand[:60]}...\"")

    return selected_ind

def main():
    parser = argparse.ArgumentParser(description="Selección final de prompts usando Greedy NLI Veto.")
    parser.add_argument("--folder", type=str, required=True, help="Nombre de la carpeta en 'exec/' (ej: 2025-11-24_14-51-56)")
    # Puedes cambiar el default a 10 si quieres más variedad
    parser.add_argument("--top-k", type=int, default=5, help="Cuántos prompts diversos seleccionar")
    
    args = parser.parse_args()

    # Rutas
    base_path = Path("exec") / args.folder
    json_path = base_path / "pareto_front.json"

    print(f"📂 Cargando Frente de Pareto: {json_path}")
    data = load_json(json_path)

    if not data:
        print("⚠️ El archivo está vacío.")
        return

    # --- EJECUTAR LA NUEVA LÓGICA DE SELECCIÓN ---
    top_prompts = run_greedy_nli_selection(data, n_select=args.top_k)

    # --- REPORTE EN CONSOLA ---
    print("\n" + "="*60)
    print(f"🏆  TOP {len(top_prompts)} PROMPTS FINALES (Calidad + Diversidad NLI)")
    print("="*60)
    
    for i, ind in enumerate(top_prompts):
        print(f"\n🥇 RANGO #{i+1}")
        print(f"   Fidelidad (SBERT): {ind['objetivos'][0]:.4f}") # Mostramos la fidelidad original
        print(f"   Role: {ind['role']}")
        print(f"   Prompt: \"{ind['prompt']}\"")
        print(f"   Generado: \"{ind['generated_data']}\"")

    # --- GUARDAR RESULTADO ---
    # Lo guardamos con un nombre distinto para diferenciarlo del método antiguo
    out_file = base_path / "final_selection_nli.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(top_prompts, f, indent=2, ensure_ascii=False)
    
    print("\n" + "-"*60)
    print(f"✅ Selección guardada en: {out_file}")
    print("Estos son los prompts que deberías usar para generar tu dataset sintético.")

if __name__ == "__main__":
    main()