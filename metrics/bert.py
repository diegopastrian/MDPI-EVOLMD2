# metrics/bert.py
import json
from pathlib import Path
from bert_score import score

DATA_PATH = Path("../data.json")
REF_PATH = Path("../data/reference.txt")


def cargar_individuos(archivo=DATA_PATH):
    with open(archivo, "r", encoding="utf-8") as f:
        return json.load(f)


def guardar_individuos(individuos, archivo=DATA_PATH):
    with open(archivo, "w", encoding="utf-8") as f:
        json.dump(individuos, f, indent=2, ensure_ascii=False)
    # print(f"✅ {len(individuos)} individuos con fitness guardados en {archivo}")


def bertscore_individuos(
    individuos,
    ref_text: str,
    model_type: str,
    lang="en"
):
    """
    Recibe lista de individuos y el texto de referencia.
    Calcula BERTScore (F1) entre generated_data y ref_text.
    Devuelve la lista de individuos con 'fitness' actualizado.
    """
    candidates = [ind.get("generated_data", "") for ind in individuos]
    references = [ref_text] * len(individuos)

    P, R, F1 = score(
        cands=candidates,
        refs=references,
        model_type=model_type,
        lang=lang,
        verbose=False
    )

    f1_scores = F1.tolist()

    for ind, fit in zip(individuos, f1_scores):
        ind["fitness"] = fit

    return individuos


def main():
    if not REF_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo de referencia: {REF_PATH}")
    ref_text = REF_PATH.read_text(encoding="utf-8").strip()

    individuos = cargar_individuos()
    print(f"⚙️ Calculando BERTScore para {len(individuos)} individuos...")

    individuos = bertscore_individuos(individuos, ref_text)

    guardar_individuos(individuos)


if __name__ == "__main__":
    main()
