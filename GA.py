# GA.py
import json
import random
import argparse
from pathlib import Path
from datetime import datetime
from shutil import copy2
import statistics
import csv
import time
import asyncio

from agents.llm_agent import LLMAgent
from agents.generate_data import generar_data_con_ollama
from agents.keyword_prompts import obtener_prompt_por_keywords
from metrics.bert import bertscore_individuos
from operadores.crossover_pablo import crossover
from operadores.mutacion_nico import mutacion

DATA_PATH = Path("data.json")
REF_PATH = Path("data/reference.txt")

# -------- utils --------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def cargar_individuos(archivo=DATA_PATH):
    with open(archivo, "r", encoding="utf-8") as f:
        return json.load(f)

def guardar_individuos(individuos, archivo: Path):
    ensure_dir(archivo.parent)
    with open(archivo, "w", encoding="utf-8") as f:
        json.dump(individuos, f, indent=2, ensure_ascii=False)
    # print(f"✅ {len(individuos)} individuos guardados en {archivo}")

# -------- métricas --------
def _fitness_list(poblacion):
    return [ind.get("fitness", 0.0) for ind in poblacion]

def compute_stats(poblacion):
    fits = _fitness_list(poblacion)
    if not fits:
        return {"count":0,"mean":0.0,"std":0.0,"min":0.0,"p25":0.0,"median":0.0,"p75":0.0,"max":0.0}
    srt = sorted(fits)
    n = len(srt) - 1
    return {
        "count": len(fits),
        "mean": float(statistics.mean(fits)),
        "std": float(statistics.pstdev(fits)) if len(fits) > 1 else 0.0,
        "min": float(srt[0]),
        "median": float(statistics.median(srt)),
        "max": float(srt[-1]),
        "p25": float(srt[int(0.25*n)]),
        "p75": float(srt[int(0.75*n)]),
    }

def init_metrics_files(outdir: Path):
    csv_path = outdir / "metrics_gen.csv"
    txt_path = outdir / "metrics_gen.txt"
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "generation","count","mean","std","min","median","max","p25","p75","duration_sec"
            ])
    if not txt_path.exists():
        txt_path.write_text("# Métricas por generación\n", encoding="utf-8")
    return csv_path, txt_path

def append_metrics(outdir: Path, generation: int, poblacion, duration_sec: float):
    stats = compute_stats(poblacion)
    csv_path, txt_path = init_metrics_files(outdir)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            generation, stats["count"],
            f"{stats['mean']:.6f}", f"{stats['std']:.6f}",
            f"{stats['min']:.6f}", f"{stats['median']:.6f}", f"{stats['max']:.6f}",
            f"{stats['p25']:.6f}", f"{stats['p75']:.6f}",
            f"{duration_sec:.6f}"
        ])
    line = (f"Gen {generation:02d} | n={stats['count']} "
            f"| mean={stats['mean']:.6f} | min={stats['min']:.6f} | max={stats['max']:.6f} "
            f"| duration_sec={duration_sec:.3f}\n")
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(line)
    gen_dir = outdir / "metrics"
    ensure_dir(gen_dir)
    with open(gen_dir / f"gen_{generation:02d}.txt", "w", encoding="utf-8") as f:
        f.write(
            f"mean={stats['mean']:.6f}\n"
            f"max={stats['max']:.6f}\n"
            f"min={stats['min']:.6f}\n"
            f"duration_sec={duration_sec:.6f}\n"
        )

# -------- GA --------
async def generar_data_para_individuo(individuo, ref_text, llm_agent: 'LLMAgent', temperatura=0.7):
    out = await generar_data_con_ollama(
        individuo, texto_referencia=ref_text, llm_agent=llm_agent, temperatura=temperatura
    )
    individuo["generated_data"] = out.data.strip() if out else ""
    return individuo

async def regenerar_prompt(individuo, ref_text, llm_agent: 'LLMAgent', temperatura=0.9):
    if individuo.get("keywords"):
        prompts = await obtener_prompt_por_keywords(
            texto_referencia=ref_text, rol=individuo["rol"], topic=individuo["topic"],
            keywords=individuo["keywords"], n=1, llm_agent=llm_agent, temperatura=temperatura
        )
        individuo["prompt"] = prompts[0] if prompts else ""
    else:
        individuo["prompt"] = ""
    return individuo

def torneo(individuos, k=3):
    candidatos = random.sample(individuos, k)
    return max(candidatos, key=lambda x: x["fitness"])

# Función asíncrona para procesar un solo hijo
async def procesar_hijo(hijo: dict, ref_text: str, llm_agent: 'LLMAgent', prob_mutacion: float):
    """
    Encapsula toda la lógica de procesamiento (mutación, prompt, data) para un individuo.
    """
    if random.random() < prob_mutacion:
        hijo = await mutacion(hijo, llm_agent) # La mutación ahora es asíncrona
    
    hijo = await regenerar_prompt(hijo, ref_text, llm_agent)
    hijo = await generar_data_para_individuo(hijo, ref_text, llm_agent)
    
    return hijo


async def metaheuristica(individuos, ref_text, llm_agent: 'LLMAgent', bert_model: str,
                   generaciones=5, k=3, prob_crossover=0.8, prob_mutacion=0.1, num_elitismo=2,
                   outdir: Path = Path(".")):
    poblacion = individuos[:]
    evo_t0 = time.perf_counter()
    for g in range(generaciones):
        gen_t0 = time.perf_counter()
        print(f"\n🌀 Generación {g+1}/{generaciones}")
        poblacion.sort(key=lambda x: x["fitness"], reverse=True)

        nueva_poblacion = poblacion[:num_elitismo]  # elitismo
        
        # Creamos una lista de hijos sin procesar
        hijos_a_procesar = []

        while len(nueva_poblacion) + len(hijos_a_procesar) < len(poblacion):
            if random.random() < prob_crossover:
                p1, p2 = torneo(poblacion, k=k), torneo(poblacion, k=k)
                hijo = crossover(p1, p2)
            else:
                hijo = torneo(poblacion, k=k).copy()
            hijos_a_procesar.append(hijo)

        # Creamos una lista de tareas asíncronas
        tasks = [
            procesar_hijo(h, ref_text, llm_agent, prob_mutacion) for h in hijos_a_procesar
        ]
        
        # Ejecutamos todas las tareas en paralelo y esperamos los resultados
        poblacion_a_evaluar = await asyncio.gather(*tasks)

        # Evaluamos a TODOS los nuevos individuos en un solo lote
        if poblacion_a_evaluar:
            poblacion_recien_evaluada = bertscore_individuos(poblacion_a_evaluar, ref_text, model_type=bert_model)
            # Añadimos los individuos ya evaluados a la nueva población.
            nueva_poblacion.extend(poblacion_recien_evaluada)

        poblacion = nueva_poblacion
        best = max(poblacion, key=lambda x: x["fitness"])
        gen_dt = time.perf_counter() - gen_t0
        print(f"   → Mejor fitness: {best['fitness']:.4f} (Duración: {gen_dt:.2f}s)")
        append_metrics(outdir, g+1, poblacion, duration_sec=gen_dt)
        
    evo_dt = time.perf_counter() - evo_t0
    with open(outdir / "runtime.tmp", "a", encoding="utf-8") as f:
        f.write(f"evolution_sec={evo_dt:.6f}\n")
    return poblacion

# -------- main --------
async def main():
    parser = argparse.ArgumentParser(description="Algoritmo genético para evolución de prompts")
    parser.add_argument("--generaciones", type=int, default=3)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--prob-crossover", type=float, default=0.8)
    parser.add_argument("--prob-mutacion", type=float, default=0.1)
    parser.add_argument("--num-elitismo", type=int, default=2)
    parser.add_argument("--outdir", type=Path, default=Path("pruebas/_tmp"))
    parser.add_argument("--data-in", type=Path, default=DATA_PATH)
    parser.add_argument("--model", default="llama3", help="Modelo LLM a utilizar.")
    parser.add_argument("--bert-model", default="microsoft/deberta-xlarge-mnli", help="Modelo BERT a utilizar para la función de fitness.")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = args.outdir
    ensure_dir(outdir)

    # print(f"📦 Outdir: {outdir}")

    t_total0 = time.perf_counter()

    if not REF_PATH.exists(): raise FileNotFoundError(f"No se encontró referencia: {REF_PATH}")
    ref_text = REF_PATH.read_text(encoding="utf-8").strip()

    # Se crea una única instancia del agente
    llm_agent = LLMAgent(model=args.model)

    individuos = cargar_individuos(args.data_in)

    # Generación de data inicial
    t_init_gen0 = time.perf_counter()
    # print(f"⚙️ Generando data inicial para {len(individuos)} individuos...")
    # La generación de data inicial también se hace en paralelo
    tasks_iniciales = [generar_data_para_individuo(ind, ref_text, llm_agent) for ind in individuos]
    individuos = await asyncio.gather(*tasks_iniciales)
    t_init_gen = time.perf_counter() - t_init_gen0

    # Evaluación inicial
    t_init_eval0 = time.perf_counter()
    # print("\n📏 Fitness inicial...")
    individuos = bertscore_individuos(individuos, ref_text, model_type=args.bert_model)
    data_inicial_eval_path = outdir / "data_inicial_evaluada.json"
    guardar_individuos(individuos, data_inicial_eval_path)
    t_init_eval = time.perf_counter() - t_init_eval0
    append_metrics(outdir, 0, individuos, duration_sec=t_init_eval)

    # Evolución
    # print("\n🚀 Evolución...")
    poblacion_final = await metaheuristica(
        individuos=individuos,
        ref_text=ref_text,
        llm_agent=llm_agent,
        bert_model=args.bert_model,
        generaciones=args.generaciones,
        k=args.k,
        prob_crossover=args.prob_crossover,  # manter prob_crossover
        prob_mutacion=args.prob_mutacion,
        num_elitismo=args.num_elitismo,
        outdir=outdir,
    )

    # Guardado final: SOLO evaluada
    t_final_eval0 = time.perf_counter()
    poblacion_final_eval = bertscore_individuos(poblacion_final, ref_text, model_type=args.bert_model)
    data_final_eval_path = outdir / "data_final_evaluada.json"
    guardar_individuos(poblacion_final_eval, data_final_eval_path)
    best_individual_path = outdir / "best_individual.json"
    best_final = max(poblacion_final_eval, key=lambda x: x["fitness"], default=None)
    if best_final:
        with open(best_individual_path, "w", encoding="utf-8") as f:
            json.dump(best_final, f, ensure_ascii=False, indent=2)
        # print(f"🏁 Mejor individuo final -> {best_individual_path}")
    t_final_eval = time.perf_counter() - t_final_eval0

    total_sec = time.perf_counter() - t_total0

    # Escribir tiempos
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

if __name__ == "__main__":
    asyncio.run(main())