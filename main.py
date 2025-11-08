# main.py
import argparse
from pathlib import Path
import time
import asyncio

from ga_core import setup
from ga_core import initial_population
from ga_core.utils import guardar_individuos
from metrics.reports import append_metrics
from metrics.bert import bertscore_individuos
from agents.llm_agent import LLMAgent
from ga_core.ga import metaheuristica, generar_data_para_individuo

async def main():
    parser = argparse.ArgumentParser(description="Algoritmo genético para evolución de prompts")

    # --- Argumentos de GA (Evolución) ---
    parser.add_argument("--generaciones", type=int, default=3)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--prob-crossover", type=float, default=0.8)
    parser.add_argument("--prob-mutacion", type=float, default=0.1)
    parser.add_argument("--num-elitismo", type=int, default=2)
    parser.add_argument("--model", default="llama3", help="Modelo LLM a utilizar.")
    parser.add_argument("--bert-model", default="bert-base-uncased", help="Modelo BERT a utilizar para la función de fitness.")

    # --- Argumentos de Población Inicial ---
    parser.add_argument("--n", type=int, required=True, help="Cantidad de individuos.")
    parser.add_argument("--texto-referencia", type=str, default=None, help="Texto de referencia específico a utilizar.")
    
    # --- Argumentos de Salida ---
    parser.add_argument("--outdir-base", type=Path, default=Path("exec"), help="Directorio donde se guardan las ejecuciones.")
    
    args = parser.parse_args()

    # --- 1. Preparar Entorno ---
    print("1/5 Preparando directorio del experimento...")
    outdir, ref_text = setup.setup_experiment(
        base_dir=args.outdir_base,
        texto_referencia_arg=args.texto_referencia
    )
    print(f"   → Todos los archivos se guardarán en: {outdir}")

    t_total0 = time.perf_counter()

    # --- 2. Crear Agente y Población Inicial ---
    print(f"2/5 Generando población inicial de {args.n} individuos...")

    # Se crea una única instancia del agente
    llm_agent = LLMAgent(model=args.model)

    individuos = await initial_population.generar_poblacion_inicial(
        n=args.n,
        llm_agent=llm_agent,
        texto_referencia=ref_text,
        archivo_salida=outdir / "data_initial_population.json"
    )

    # --- 3. Generar Data Inicial ---
    print("3/5 Generando data para la población inicial...")
    t_init_gen0 = time.perf_counter()
    tasks_iniciales = [generar_data_para_individuo(ind, ref_text, llm_agent) for ind in individuos]
    individuos = await asyncio.gather(*tasks_iniciales)
    t_init_gen = time.perf_counter() - t_init_gen0

    # --- 4. Evaluación Inicial ---
    print("4/5 Evaluando fitness de la población inicial...")
    t_init_eval0 = time.perf_counter()
    individuos = bertscore_individuos(individuos, ref_text, model_type=args.bert_model)
    data_inicial_eval_path = outdir / "data_inicial_evaluada.json"
    guardar_individuos(individuos, data_inicial_eval_path) 
    t_init_eval = time.perf_counter() - t_init_eval0
    append_metrics(outdir, 0, individuos, duration_sec=t_init_eval)

    # --- 5. Evolución ---
    print("5/5 Iniciando evolución...")
    poblacion_final = await metaheuristica(
        individuos=individuos,
        ref_text=ref_text,
        llm_agent=llm_agent,
        bert_model=args.bert_model,
        generaciones=args.generaciones,
        k=args.k,
        prob_crossover=args.prob_crossover,
        prob_mutacion=args.prob_mutacion,
        num_elitismo=args.num_elitismo,
        outdir=outdir,
    )

    # --- Guardado final ---
    t_final_eval0 = time.perf_counter()
    poblacion_final_eval = bertscore_individuos(poblacion_final, ref_text, model_type=args.bert_model)
    data_final_eval_path = outdir / "data_final_evaluada.json"
    guardar_individuos(poblacion_final_eval, data_final_eval_path)
    
    t_final_eval = time.perf_counter() - t_final_eval0
    total_sec = time.perf_counter() - t_total0

    # --- Escribir Tiempos ---
    runtime_path = outdir / "runtime.txt"
    evo_sec = 0.0
    tmp_path = outdir / "runtime.tmp"
    if tmp_path.exists():
        try:
            with open(tmp_path, "r", encoding="utf-8") as f:
                for ln in f:
                    if ln.startswith("evolution_sec="):
                        evo_sec = float(ln.split("=", 1)[1].strip())
        finally:
            tmp_path.unlink(missing_ok=True)

    with open(runtime_path, "w", encoding="utf-8") as f:
        f.write(f"initial_gen_sec={t_init_gen:.6f}\n")
        f.write(f"initial_eval_sec={t_init_eval:.6f}\n")
        f.write(f"final_eval_sec={t_final_eval:.6f}\n")
        f.write(f"evolution_sec={evo_sec:.6f}\n")
        f.write(f"total_sec={total_sec:.6f}\n")

    print(f"\n✅ Proceso completado. Resultados guardados en: {outdir}")

if __name__ == "__main__":
    asyncio.run(main())