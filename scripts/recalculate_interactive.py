print("🔄 Iniciando script...") # Confirmación visual inmediata

import sys
import json
import argparse
from pathlib import Path

# --- 1. CONFIGURACIÓN DE RUTAS  ---
script_location = Path(__file__).resolve()
scripts_dir = script_location.parent        
project_root = scripts_dir.parent           

if str(scripts_dir) not in sys.path:
    sys.path.append(str(scripts_dir))

try:
    from analyze_pareto import load_json, run_hybrid_selection
except ImportError as e:
    print(f"\n❌ ERROR CRÍTICO DE IMPORTACIÓN:")
    print(f"No se pudo importar 'analyze_pareto.py'.")
    print(f"Python busca en: {scripts_dir}")
    print(f"Detalle: {e}")
    sys.exit(1)

# --- 2. CONFIGURACIÓN ---
DEFAULT_CONFIG = {
    "top_k": 5,
    "lambda_param": 0.25,
    "max_fidelity": 0.85,
    "min_fidelity": 0.25
}

def get_exec_folders():
    """Busca la carpeta exec en la raíz del proyecto."""
    exec_path = project_root / "exec" / "TESTS" / "GA_nuevo(exp)"
    
    if not exec_path.exists():
        print(f"\n❌ ERROR: No se encuentra la carpeta 'exec'.")
        print(f"   Buscada en: {exec_path}")
        return []
    
    
    folders = sorted([d for d in exec_path.iterdir() if d.is_dir()])
    return folders

def process_folder(folder_path, config):
    json_path = folder_path / "pareto_front.json"
    
    if not json_path.exists():
        return " Incompleto (Falta pareto)"
    
    try:
        data = load_json(json_path)
    except:
        return " JSON corrupto"

    if not data:
        return " JSON vacío"

    try:
        selection = run_hybrid_selection(
            data,
            n_select=config["top_k"],
            lambda_param=config["lambda_param"],
            max_fidelity_threshold=config["max_fidelity"],
            min_fidelity_threshold=config["min_fidelity"]
        )
        
        out_file = folder_path / "final_selection_hybrid.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(selection, f, indent=2, ensure_ascii=False)
            
        return "✅ Recalculado"
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-param", type=float, default=DEFAULT_CONFIG["lambda_param"])
    parser.add_argument("--min-fidelity", type=float, default=DEFAULT_CONFIG["min_fidelity"])
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["lambda_param"] = args.lambda_param
    config["min_fidelity"] = args.min_fidelity

    print(f"📂 Directorio del proyecto: {project_root}")
    folders = get_exec_folders()
    
    if not folders:
        print("⚠️ No hay experimentos en la carpeta 'exec'.")
        return

    print(f"📊 Experimentos encontrados: {len(folders)}")
    print("-" * 50)
    
    print(" [1] 🔄 Recalcular TODOS")
    print(" [2] 👆 Seleccionar uno específico")
    
    try:
        choice = input("\n👉 Opción: ").strip()
    except KeyboardInterrupt:
        print("\n🛑 Cancelado.")
        return
    
    target_folders = []

    if choice == "1":
        target_folders = folders
    elif choice == "2":
        for i, f in enumerate(folders):
            # Check visual rápido
            has_data = "✅" if (f / "pareto_front.json").exists() else "⏳"
            print(f"   [{i+1}] {f.name} {has_data}")
            
        val = input("\nNúmero del experimento: ").strip()
        try:
            # Soporte para lista separada por comas "1, 3"
            indices = [int(x)-1 for x in val.split(",") if x.strip().isdigit()]
            for idx in indices:
                if 0 <= idx < len(folders):
                    target_folders.append(folders[idx])
        except:
            print("❌ Entrada inválida")
            return
    else:
        print("❌ Opción inválida")
        return

    if not target_folders:
        print("⚠️ Ninguna carpeta seleccionada.")
        return

    print("\n🚀 Procesando...\n")
    print(f"{'CARPETA':<35} | {'ESTADO'}")
    print("-" * 60)
    
    for folder in target_folders:
        status = process_folder(folder, config)
        name_short = (folder.name[:32] + '..') if len(folder.name) > 32 else folder.name
        print(f"{name_short:<35} | {status}")

    print("-" * 60)
    print("✨ Listo.")

if __name__ == "__main__":
    main()