# setup.py

import random
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
import csv

def cargar_texto_unico(archivo: str = "corpus_filtrado.csv") -> str:
    """
    Carga una línea aleatoria del archivo de corpus especificado.
    """
    p = Path(archivo)
    if not p.exists():
        # Si no se encuentra el corpus filtrado, usar el de ejemplo
        p_ejemplo = Path("corpus_ejemplo.txt")
        if p_ejemplo.exists():
            print(f"Advertencia: No se encontró '{archivo}'. Usando 'corpus_ejemplo.txt' como fallback.")
            p = p_ejemplo
            # Lógica original para leer el .txt
            lineas = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if not lineas:
                raise ValueError(f"El archivo {p} está vacío o sin líneas válidas.")
            return random.choice(lineas)
        else:
            raise FileNotFoundError(f"No se encontró el archivo de corpus {archivo}. Usa el corpus de ejemplo o busca el corpus original y ejecuta 'preparar_corpus.py'.")
    
    # Leer CSV filtrado
    lineas = []
    try:
        with open(p, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            # Leemos todas las filas
            for row in reader:
                if row: # Si la fila no está vacía
                    tweet = row[0].strip()
                    if tweet:
                        lineas.append(tweet)
                        
    except Exception as e:
        raise IOError(f"Error al leer el archivo CSV {archivo}: {e}")

    
    if not lineas:
        raise ValueError(f"El archivo {archivo} está vacío o sin líneas válidas.")
    
    return random.choice(lineas)

def guardar_referencia_txt(texto: str, ruta: Path) -> None:
    """
    Guarda el texto de referencia en la ruta especificada.
    """
    # Asegura que el directorio padre exista
    ruta.parent.mkdir(parents=True, exist_ok=True)
    
    # Sobrescribe explícitamente si ya existe
    if ruta.exists():
        ruta.unlink()
    
    ruta.write_text(texto, encoding="utf-8")

def setup_experiment(
    base_dir: Path, 
    texto_referencia_arg: Optional[str] = None
) -> Tuple[Path, str]:
    """
    Prepara el entorno para una única ejecución del algoritmo genético.

    1. Crea un directorio único con timestamp dentro de 'base_dir'.
    2. Carga el texto de referencia (desde un argumento o desde el corpus).
    3. Guarda una copia de ese texto de referencia DENTRO del directorio único.
    """
    
    # 1. Crear directorio único
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = base_dir / ts
    outdir.mkdir(parents=True, exist_ok=True)

    # 2. Cargar texto de referencia
    ref_text = ""
    if texto_referencia_arg:
        # Si el usuario pasó un archivo de referencia específico
        p_ref = Path(texto_referencia_arg)
        if not p_ref.exists():
            raise FileNotFoundError(f"El archivo de referencia especificado no existe: {texto_referencia_arg}")
        ref_text = p_ref.read_text(encoding="utf-8").strip()
    else:
        # Si no, carga uno aleatorio del corpus
        ref_text = cargar_texto_unico() # Usa la función local
    
    # 3. Guardar la referencia DENTRO del directorio de la ejecución
    ref_save_path = outdir / "reference.txt"
    guardar_referencia_txt(ref_text, ref_save_path) # Usa la función local
    
    # 4. Devolver la ruta y el texto
    return outdir, ref_text