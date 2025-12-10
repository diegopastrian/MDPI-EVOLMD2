import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# --- CONFIGURACIÓN DE RUTAS ---
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
# Ajustamos la ruta a donde dijiste que dejaste los archivos
exec_path = project_root / "exec" / "TESTS" 

# --- CONFIGURACIÓN DE VISUALIZACIÓN ---
USE_TSNE = False  # False = PCA (Más estable), True = t-SNE (Mejor para clusters)

def get_best_candidates_from_population(pop_data, top_k=5):
    """
    Si no hay archivo de selección (GA Original), extraemos los mejores
    basándonos en su fitness o fidelidad.
    """
    try:
        # Intenta ordenar por 'fitness' (GA Mono-objetivo antiguo)
        # o por el primer objetivo (GA Multi-objetivo)
        if not pop_data:
            return []
            
        first = pop_data[0]
        
        if "fitness" in first:
            # Ordenar descendente por fitness
            sorted_pop = sorted(pop_data, key=lambda x: x.get("fitness", 0), reverse=True)
        elif "objetivos" in first:
            # Ordenar descendente por primer objetivo (Fidelidad)
            sorted_pop = sorted(pop_data, key=lambda x: x["objetivos"][0], reverse=True)
        else:
            return []
            
        return [p["generated_data"] for p in sorted_pop[:top_k]]
    except Exception as e:
        print(f"⚠️ No se pudo inferir el top {top_k}: {e}")
        return []

def load_data(folder_path):
    """Carga datos manejando las diferencias de estructura entre GA Nuevo y Original."""
    ref_text = "REF N/A"
    pop_texts = []
    sel_texts = []
    
    try:
        if not folder_path.exists():
            print(f"⚠️ Carpeta no encontrada: {folder_path}")
            return "", [], []

        # 1. Cargar Referencia
        ref_path = folder_path / "reference.txt"
        if ref_path.exists():
            ref_text = ref_path.read_text(encoding="utf-8").strip()

        # 2. Cargar Población (Estrategia de Búsqueda)
        # Prioridad: pareto_front.json (Nuevo) > data_final_evaluada.json (Original)
        pop_file = None
        if (folder_path / "pareto_front.json").exists():
            pop_file = folder_path / "pareto_front.json"
        elif (folder_path / "data_final_evaluada.json").exists():
            pop_file = folder_path / "data_final_evaluada.json"
        
        pop_data = []
        if pop_file:
            pop_data = json.loads(pop_file.read_text(encoding="utf-8"))
            pop_texts = [p["generated_data"] for p in pop_data if p.get("generated_data")]
        else:
            print(f"⚠️ No se encontró archivo de población en {folder_path.name}")

        # 3. Cargar Seleccionados
        # Estrategia: Buscar archivo explícito -> Si falla, inferir del top de población
        sel_file = None
        if (folder_path / "final_selection_hybrid.json").exists():
            sel_file = folder_path / "final_selection_hybrid.json"
        elif (folder_path / "pareto_ranked.json").exists():
            sel_file = folder_path / "pareto_ranked.json"
            
        if sel_file:
            sel_data = json.loads(sel_file.read_text(encoding="utf-8"))
            sel_texts = [s["generated_data"] for s in sel_data if s.get("generated_data")]
        else:
            # Fallback para GA Original: Tomamos el Top 5 de la población cargada
            if pop_data:
                # print(f"   ℹ️ Infiriendo selección para {folder_path.name} (Top 5 fitness)")
                sel_texts = get_best_candidates_from_population(pop_data, top_k=5)

        return ref_text, pop_texts, sel_texts

    except Exception as e:
        print(f"⚠️ Error cargando {folder_path.name}: {e}")
        return "", [], []

def generate_plot(prueba_name, data_orig, data_new, model):
    print(f"   🎨 Generando gráficos para {prueba_name}...")
    
    ref_orig, pop_orig, sel_orig = data_orig
    ref_new, pop_new, sel_new = data_new

    # Validación básica
    if not pop_orig or not pop_new:
        print(f"      ❌ Falta población en {prueba_name} (Orig: {len(pop_orig)}, New: {len(pop_new)}). Saltando.")
        return

    # --- 1. VECTORIZACIÓN CONJUNTA ---
    # Unimos todo para que el PCA sea consistente (mismo espacio vectorial)
    # Usamos solo UNA referencia para el gráfico (la del nuevo, deberían ser iguales)
    ref_text = ref_new if ref_new != "REF N/A" else ref_orig
    
    all_texts = [ref_text] + pop_orig + pop_new
    embeddings = model.encode(all_texts, convert_to_tensor=False)
    
    # Separamos los índices
    idx_ref = 0
    idx_pop_orig_start = 1
    idx_pop_orig_end = 1 + len(pop_orig)
    idx_pop_new_start = idx_pop_orig_end
    
    # --- 2. REDUCCIÓN DE DIMENSIONALIDAD ---
    if USE_TSNE:
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_texts)-1))
    else:
        reducer = PCA(n_components=2)
        
    coords = reducer.fit_transform(embeddings)
    
    # Coordenadas separadas
    xy_ref = coords[idx_ref]
    xy_pop_orig = coords[idx_pop_orig_start:idx_pop_orig_end]
    xy_pop_new = coords[idx_pop_new_start:]
    
    # Función auxiliar para encontrar coords de seleccionados
    def get_sel_coords(selection_texts, population_texts, population_coords):
        sel_coords = []
        for txt in selection_texts:
            try:
                # Buscamos el índice exacto
                idx = population_texts.index(txt)
                sel_coords.append(population_coords[idx])
            except ValueError:
                continue 
        return np.array(sel_coords)

    xy_sel_orig = get_sel_coords(sel_orig, pop_orig, xy_pop_orig)
    xy_sel_new = get_sel_coords(sel_new, pop_new, xy_pop_new)

    # --- 3. PLOTTING ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Comparación de Diversidad Semántica - {prueba_name}", fontsize=16)

    # Estilos
    ALPHA_CLOUD = 0.4
    SIZE_CLOUD = 40
    SIZE_SEL = 120
    
    # === GRÁFICO 1: GA ORIGINAL ===
    ax = axes[0]
    ax.set_title("GA Original (Baseline)")
    ax.scatter(xy_pop_orig[:,0], xy_pop_orig[:,1], c='red', alpha=ALPHA_CLOUD, s=SIZE_CLOUD, label='Población')
    if len(xy_sel_orig) > 0:
        ax.scatter(xy_sel_orig[:,0], xy_sel_orig[:,1], c='white', edgecolors='darkred', linewidth=2, s=SIZE_SEL, label='Top 5', marker='o')
    ax.scatter(xy_ref[0], xy_ref[1], c='gold', edgecolors='black', s=250, marker='*', label='Referencia')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.2)

    # === GRÁFICO 2: GA NUEVO ===
    ax = axes[1]
    ax.set_title("GA Nuevo (Propuesta)")
    ax.scatter(xy_pop_new[:,0], xy_pop_new[:,1], c='blue', alpha=ALPHA_CLOUD, s=SIZE_CLOUD, label='Población')
    if len(xy_sel_new) > 0:
        ax.scatter(xy_sel_new[:,0], xy_sel_new[:,1], c='white', edgecolors='darkblue', linewidth=2, s=SIZE_SEL, label='Seleccionados', marker='o')
    ax.scatter(xy_ref[0], xy_ref[1], c='gold', edgecolors='black', s=250, marker='*', label='Referencia')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.2)

    # === GRÁFICO 3: SUPERPOSICIÓN ===
    ax = axes[2]
    ax.set_title("Superposición de Espacios de Búsqueda")
    # Dibujamos original abajo
    ax.scatter(xy_pop_orig[:,0], xy_pop_orig[:,1], c='red', alpha=0.2, s=SIZE_CLOUD, label='Original')
    # Dibujamos nuevo arriba
    ax.scatter(xy_pop_new[:,0], xy_pop_new[:,1], c='blue', alpha=0.2, s=SIZE_CLOUD, label='Nuevo')
    
    # Referencia
    ax.scatter(xy_ref[0], xy_ref[1], c='gold', edgecolors='black', s=300, marker='*', label='Referencia')
    
    ax.legend()
    ax.grid(True, alpha=0.2)

    # Guardar
    out_file = project_root / f"comparacion_visual_{prueba_name}.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    print(f"   ✅ Guardado: {out_file.name}")
    plt.close()

def main():
    print("🚀 Iniciando generación de gráficos comparativos...")
    print(f"📂 Buscando datos en: {exec_path}")
    
    if not exec_path.exists():
        print(f"❌ Error: No existe el directorio {exec_path}")
        return

    # Cargar SBERT una sola vez
    print("🧠 Cargando modelo SBERT...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    pruebas = ["prueba1", "prueba2", "prueba3"]
    
    dir_original = exec_path / "GA_original"
    dir_nuevo = exec_path / "GA_nuevo(exp)"

    if not dir_original.exists() or not dir_nuevo.exists():
        print(f"❌ Error: Faltan carpetas 'GA_original' o 'GA_nuevo' dentro de {exec_path}")
        return

    for p_name in pruebas:
        path_orig = dir_original / p_name
        path_new = dir_nuevo / p_name
        
        # Verificar existencia
        if not path_orig.exists():
            print(f"⏩ Saltando {p_name}: No existe en GA_original.")
            continue
        if not path_new.exists():
            print(f"⏩ Saltando {p_name}: No existe en GA_nuevo.")
            continue
            
        print(f"\n📊 Procesando {p_name}...")
        
        # Cargar datos
        data_orig = load_data(path_orig)
        data_new = load_data(path_new)
        
        # Generar gráfico
        generate_plot(p_name, data_orig, data_new, model)

    print("\n✨ Proceso terminado.")

if __name__ == "__main__":
    main()