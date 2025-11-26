import subprocess
import sys
import time
from pathlib import Path

def get_latest_folder(exec_path: Path):
    """
    Busca la carpeta más reciente dentro de 'exec/'.
    Retorna el nombre de la carpeta (str) o None si no encuentra nada.
    """
    if not exec_path.exists():
        return None
    
    # Listar subdirectorios
    subdirs = [d for d in exec_path.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    
    # Ordenar por fecha de modificación (el último es el más reciente)
    latest = max(subdirs, key=lambda d: d.stat().st_mtime)
    return latest.name

def run_main_script(args):
    """Ejecuta main.py con los argumentos dados."""
    cmd = [sys.executable, "main.py"] + args
    print(f"\n🚀 Ejecutando Fase 1: Algoritmo Genético...")
    print(f"   Comando: {' '.join(cmd)}")
    
    # check=True lanzará una excepción si main.py falla
    subprocess.run(cmd, check=True) 

def run_analysis_script(folder_name):
    """Ejecuta analyze_pareto.py con la carpeta dada."""
    cmd = [sys.executable, "analyze_pareto.py", "--folder", folder_name]
    print(f"\n🧠 Ejecutando Fase 2: Análisis MCDM (TOPSIS)...")
    print(f"   Analizando carpeta: {folder_name}")
    
    subprocess.run(cmd, check=True)

def main():
    print("="*50)
    print("   AUTOMATIZACIÓN DE EXPERIMENTOS (MODO DESATENDIDO)")
    print("="*50)
    print("Seleccione el tipo de prueba:")
    print("1. 🐇 PRUEBA RÁPIDA (Debug) -> (n=10, gen=3)")
    print("2. 🐢 PRUEBA EXTENSIVA (Producción) -> (n=100, gen=100)")
    print("="*50)

    # Este es el único input que darás al inicio.
    # Después puedes desconectarte sin miedo.
    choice = input("👉 Ingrese opción (1 o 2): ").strip()
    
    if choice == "1":
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
        print("❌ Opción inválida. Saliendo.")
        sys.exit(1)

    try:
        # Referencia a la carpeta exec
        exec_path = Path("exec")

        # 1. Ejecutar main.py (Fase larga)
        run_main_script(ga_args)
        
        # 2. Detectar carpeta de salida automáticamente
        print("\n🔍 Detectando carpeta de resultados...")
        
        # Pequeña pausa de seguridad para asegurar que el FS actualizó el timestamp
        time.sleep(1) 
        
        latest_folder = get_latest_folder(exec_path)
        folder_to_analyze = None
        
        if latest_folder:
            # --- CAMBIO AQUÍ: Aceptación automática ---
            print(f"✅ Carpeta detectada: '{latest_folder}'")
            print("🚀 Procediendo automáticamente con el análisis...")
            folder_to_analyze = latest_folder
        else:
            print("⚠️  No se pudo detectar automáticamente la carpeta.")

        # Fallback: Solo pide input si falló lo anterior (para no crashear en silencio)
        if not folder_to_analyze:
            print("\n🛑 ERROR DE DETECCIÓN AUTOMÁTICA.")
            print("   Como es una sesión desatendida, no podemos pedir input manual.")
            print("   Por favor, ejecuta 'analyze_pareto.py' manualmente cuando vuelvas.")
            sys.exit(1)

        # 3. Ejecutar analyze_pareto.py
        run_analysis_script(folder_to_analyze)

        print("\n" + "="*50)
        print("✅ CICLO COMPLETO FINALIZADO CON ÉXITO")
        print("="*50)

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error crítico durante la ejecución.")
        print(f"   El proceso terminó con código: {e.returncode}")
    except KeyboardInterrupt:
        print("\n🛑 Ejecución cancelada por el usuario.")

if __name__ == "__main__":
    main()