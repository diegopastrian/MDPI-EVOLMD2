# Modelo Evolutivo de Prompts

Este proyecto utiliza un Algoritmo Genético (AG) para evolucionar y optimizar automáticamente prompts de LLM. El objetivo es encontrar individuos (compuestos por `role`, `topic`, `prompt` y `keywords`) que generen texto con un alto `fitness` (similitud semántica) respecto a un texto de referencia.

## 1. Configuración del Entorno

### Requisitos Previos

* Python 3.10+
* Un servicio de Ollama corriendo localmente. El `LLMAgent` está configurado para conectarse a `http://127.0.0.1:11434`. Asegúrate de que Ollama esté en ejecución y tenga el modelo que deseas usar (ej. `llama3`).

### Instalación

1.  Clona el repositorio.
2.  (Opcional pero recomendado) Crea un entorno virtual:
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```
3.  Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```
4.  La librería `nltk` descargará los modelos necesarios (`wordnet`, `averaged_perceptron_tagger_eng`) automáticamente la primera vez que se ejecute el script de mutación.

## 2. Preparación del Corpus (Paso Único)

El algoritmo necesita un corpus de textos de referencia. El script `prepare_corpus.py` está diseñado para filtrar un archivo CSV masivo.

1.  Coloca tu archivo de corpus masivo en la raíz del proyecto y asegúrate de que se llame `corpus.csv`.
2.  Ejecuta el script de filtrado **una sola vez**:
    ```bash
    python prepare_corpus.py
    ```
3.  Esto generará un nuevo archivo, `corpus_filtrado.csv`, que será utilizado por el algoritmo.

## 3. Ejecución del Algoritmo Genético

El script principal es `main.py`. Se ejecuta desde la terminal y acepta varios argumentos para configurar la ejecución.

### Ejemplo de Ejecución

```bash
python main.py --n 50 --generaciones 10 --model llama3
```

### Argumentos Clave

Puedes ver todos los argumentos en `main.py`. Los más importantes son:

* `--n` (Requerido): Número de individuos en la población (ej. `50` o `100`).
* `--generaciones`: Número de generaciones que evolucionará el algoritmo (ej. `10`).
* `--model`: El nombre del modelo Ollama a utilizar (ej. `llama3`, `mistral`).
* `--k`: Tamaño del torneo para la selección (ej. `3`).
* `--prob-crossover`: Probabilidad de cruce (ej. `0.8`).
* `--prob-mutacion`: Probabilidad de mutación (ej. `0.1`).
* `--num-elitismo`: Número de individuos de élite que pasan a la siguiente generación (ej. `2`).
* `--texto-referencia`: (Opcional) Ruta a un archivo `.txt` específico si no quieres usar uno aleatorio del corpus filtrado.
* `--bert-model`: El modelo a usar para BERTScore (ej. `bert-base-uncased`).

## 4. Salida y Resultados

Todas las ejecuciones se guardan en el directorio `exec/`.

Cada ejecución crea una carpeta única con un *timestamp* (ej. `exec/2025-11-09_01-08-32/`), que contendrá:

* `reference.txt`: El texto de referencia aleatorio usado para esta ejecución.
* `data_initial_population.json`: Los individuos de la Generación 0 (antes de la evaluación).
* `data_inicial_evaluada.json`: La Generación 0 con su `fitness` calculado.
* `data_final_evaluada.json`: La población final después de todas las generaciones.
* `metrics_gen.csv`: Un CSV con las estadísticas (min, max, media) de fitness de cada generación.
* `runtime.txt`: Tiempos de ejecución detallados.