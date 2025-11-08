# agents/keywords_extractor.py
from typing import List, Iterable, Tuple, Any
from pydantic import BaseModel, Field
import re

# ---------------- Utilidades de normalización ----------------

_PUNCT_RE = re.compile(r"[^\w\s']+", flags=re.UNICODE)  # deja letras/dígitos/espacios/apóstrofo
_MULTI_WS_RE = re.compile(r"\s+")

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "for", "to", "of", "in", "on", "at", "by", "with",
    "from", "as", "is", "are", "was", "were", "be", "been", "being", "that", "this", "these", "those",
    "it", "its", "into", "over", "under", "about", "up", "down", "out", "off", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "can", "will", "just", "should", "could", "would"
}

def normalize_text(s: str) -> str:
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _MULTI_WS_RE.sub(" ", s).strip()
    return s

def ngrams(words: List[str], n_min: int = 1, n_max: int = 3) -> Iterable[Tuple[str, ...]]:
    L = len(words)
    for n in range(n_min, n_max + 1):
        for i in range(L - n + 1):
            yield tuple(words[i:i + n])

def build_prompt_phrase_set(prompt: str, n_min: int = 1, n_max: int = 3) -> List[str]:
    """
    Construye una lista ordenada de n-grams (1–3 palabras) que aparecen literalmente en el prompt normalizado.
    Filtra stopwords puras y n-grams de solo stopwords.
    """
    norm = normalize_text(prompt)
    words = [w for w in norm.split() if w]
    phrases = []
    seen = set()
    for ng in ngrams(words, n_min=n_min, n_max=n_max):
        # excluir n-grams de solo stopwords
        if all(w in STOPWORDS for w in ng):
            continue
        phrase = " ".join(ng)
        if phrase not in seen:
            seen.add(phrase)
            phrases.append(phrase)
    return phrases

def contained_in_prompt(keyword: str, prompt_norm: str) -> bool:
    kw = normalize_text(keyword)
    # Contención como subcadena; al normalizar, coincide aunque el original tenga puntuación/caps.
    return kw in prompt_norm if kw else False

# ---------------- Pydantic ----------------

class KeywordsOutput(BaseModel):
    keywords: List[str] = Field(
        ...,
        description="A list of 5–12 keywords/phrases, each appearing verbatim (case-insensitive) inside the given prompt."
    )

# ---------------- Heurística de schema ----------------

def _looks_like_schema(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    if "keywords" in obj:
        return False
    return (
        "$schema" in obj
        or ("type" in obj and obj.get("type") == "object" and "properties" in obj)
        or ("description" in obj and "properties" in obj and "type" in obj)
    )

# ---------------- Normalización de keywords devueltas ----------------

def _normalize_and_filter_keywords(
    proposed: List[str],
    prompt_text: str,
    max_items: int = 12
) -> List[str]:
    """
    Normaliza (lower/strip), quita duplicados y, MUY IMPORTANTE,
    solo deja keywords que estén contenidas literalmente en el prompt (normalizado).
    """
    prompt_norm = normalize_text(prompt_text)
    out: List[str] = []
    seen = set()
    for kw in proposed:
        if not isinstance(kw, str):
            continue
        clean = normalize_text(kw)
        if not clean:
            continue
        if clean in seen:
            continue
        if not contained_in_prompt(clean, prompt_norm):
            continue
        seen.add(clean)
        out.append(clean)
        if len(out) >= max_items:
            break
    return out

def _fallback_extract_from_prompt(prompt: str, k_min: int = 5, k_max: int = 12) -> List[str]:
    """
    Fallback sin LLM: extrae n-grams 1–3 palabras del prompt normalizado.
    Prioriza 2–3 palabras; si faltan, completa con unigramas informativos (no stopwords).
    """
    phrases = build_prompt_phrase_set(prompt, 1, 3)
    # Heurística: preferir bigramas y trigramas; luego unigramas.
    tri = [p for p in phrases if len(p.split()) == 3]
    bi  = [p for p in phrases if len(p.split()) == 2]
    uni = [p for p in phrases if len(p.split()) == 1 and p not in STOPWORDS]

    selected: List[str] = []
    for group in (tri, bi, uni):
        for ph in group:
            if ph not in selected:
                selected.append(ph)
            if len(selected) >= k_max:
                break
        if len(selected) >= k_max:
            break

    if len(selected) < k_min:
        # Completar con unigramas restantes si hay
        for ph in uni:
            if ph not in selected:
                selected.append(ph)
            if len(selected) >= k_min:
                break

    return selected[:k_max]

# ---------------- Agente: extracción de keywords ----------------

async def extraer_keywords_con_ollama(
    prompt: str,
    topic: str,
    role: str,
    llm_agent: 'LLMAgent',
    temperatura: float = 0.3,
    max_reintentos: int = 1,
    k_min: int = 5,
    k_max: int = 12,
) -> List[str]:
    """
    Extrae keywords delegando la lógica de la llamada al LLMAgent.
    """
    return await llm_agent.extraer_keywords(
        prompt=prompt,
        topic=topic,
        role=role,
        temperatura=temperatura,
        max_reintentos=max_reintentos,
        k_min=k_min,
        k_max=k_max
    )
