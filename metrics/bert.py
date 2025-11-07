# metrics/bert.py
from bert_score import score

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
