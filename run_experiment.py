import subprocess
import sys
import time
import argparse
from pathlib import Path

# --- TEXTOS DE REFERENCIA PARA LA PRUEBA BATCH ---
BATCH_TEXTS = [
    # Caso 1: Italia / Advertencia
    "brother just look italy case first few case came 31st jan now italy top the death list corona viruse cases boom very fast medical facility is also not good in south east asia and you have neighbor worst conditions your government be very careful",
    
    # Caso 2: Salud Mental / Ansiedad
    "this quarantine has kicked my depression up a couple notches thanks to my work and routine being void now and im effectively avoiding my phone now bc everyone is nuts sending corona stuff dont blame them but christ the anxiety is driving me up the Wall",
    
    # Caso 3: Finanzas / Negocios
    "update our action center has been udpated with more information about restaurant shutdowns and disaster financing options for smbs"
]

def get_latest_folder(exec_path: Path):
    """Busca la carpeta más reciente dentro de 'exec/'."""
    if not exec_path.exists():
        return None
    subdirs = [d for d in exec_path.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    return max(subdirs, key=lambda d: d.stat().st_mtime).name

def run_main_script(args):
    """Ejecuta main.py con los argumentos dados."""
    cmd = [sys.executable, "main.py"] + args
    print(f"\n🚀 Ejecutando Fase 1: Algoritmo Genético...")
    print(f"   Comando: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def run_analysis_script(folder_name):
    """Ejecuta analyze_pareto.py con la carpeta dada."""
    cmd = [sys.executable, "analyze_pareto.py", "--folder", folder_name]
    print(f"\n🧠 Ejecutando Fase 2: Análisis MCDM (NLI Selection)...")
    print(f"   Analizando carpeta: {folder_name}")
    subprocess.run(cmd, check=True)

def run_batch_execution():
    """Ejecuta la secuencia de 3 experimentos automáticos."""
    print("\n" + "="*60)
    print("🔄 INICIANDO MODO BATCH (3 Ejecuciones Secuenciales)")
    print("="*60)

    # Argumentos base (Configuración de Producción)
    base_args = [
        "--n", "100",
        "--generaciones", "100",
        "--k", "5",
        "--num-elitismo", "5",
        "--prob-crossover", "0.8",
        "--prob-mutacion", "0.05"
    ]

    for i, text_content in enumerate(BATCH_TEXTS):
        print(f"\n📢 --- INICIANDO EXPERIMENTO {i+1}/3 ---")
        
        # 1. Crear archivo temporal de referencia
        temp_ref_file = Path(f"ref_temp_batch_{i+1}.txt")
        temp_ref_file.write_text(text_content, encoding="utf-8")
        print(f"📄 Archivo de referencia creado: {temp_ref_file}")

        try:
            # 2. Configurar argumentos específicos
            current_args = base_args + ["--texto-referencia", str(temp_ref_file)]
            
            # 3. Ejecutar GA
            run_main_script(current_args)

            # 4. Detectar carpeta output (Esperamos un poco para asegurar timestamp único)
            time.sleep(2) 
            latest_folder = get_latest_folder(Path("exec"))
            
            if latest_folder:
                print(f"✅ Carpeta detectada: {latest_folder}")
                # 5. Ejecutar Análisis
                run_analysis_script(latest_folder)
            else:
                print("❌ Error: No se detectó la carpeta de salida.")

        except Exception as e:
            print(f"❌ Error en el Experimento {i+1}: {e}")
        
        finally:
            # Limpieza: Borrar el archivo temporal
            if temp_ref_file.exists():
                temp_ref_file.unlink()
                print("🧹 Archivo temporal eliminado.")
        
        print(f"✅ FIN EXPERIMENTO {i+1}/3")
        print("-" * 60)

def main():
    parser = argparse.ArgumentParser(description="Runner de experimentos.")
    parser.add_argument("--mode", type=str, choices=["1", "2", "3"], help="Modo de ejecución (1: Rápida, 2: Extensa, 3: Batch 3 Casos)")
    args = parser.parse_args()

    print("="*50)
    print("   AUTOMATIZACIÓN DE EXPERIMENTOS")
    print("="*50)

    # Si pasaron argumento por consola (para nohup), lo usamos. Si no, preguntamos.
    if args.mode:
        choice = args.mode
        print(f"👉 Modo seleccionado por argumento: {choice}")
    else:
        print("1. 🐇 PRUEBA RÁPIDA (Debug) -> (n=10, gen=3)")
        print("2. 🐢 PRUEBA EXTENSIVA (Producción) -> (n=100, gen=100)")
        print("3. 📦 BATCH 3 CASOS (SSH/Nohup) -> (3 ejecuciones secuenciales)")
        choice = input("👉 Ingrese opción (1, 2 o 3): ").strip()

    # --- Lógica de Selección ---
    if choice == "3":
        run_batch_execution()
        return  # El modo batch maneja su propio ciclo

    elif choice == "1":
        ga_args = ["--n", "10", "--generaciones", "3"]
    elif choice == "2":
        ga_args = [
            "--n", "100",
            "--generaciones", "100",
            "--k", "5",
            "--num-elitismo", "5",
            "--prob-crossover", "0.8",
            "--prob-mutacion", "0.05"
        ]
    else:
        print("❌ Opción inválida.")
        sys.exit(1)

    # Ejecución normal (Opción 1 o 2)
    try:
        run_main_script(ga_args)
        
        print("\n🔍 Detectando carpeta de resultados...")
        time.sleep(1)
        latest_folder = get_latest_folder(Path("exec"))
        
        if latest_folder:
            print(f"✅ Carpeta detectada: '{latest_folder}'")
            run_analysis_script(latest_folder)
        else:
            print("⚠️ No se pudo detectar la carpeta.")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error crítico: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Cancelado por usuario.")

if __name__ == "__main__":
    main()