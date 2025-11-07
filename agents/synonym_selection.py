# agents/synonym_selection_agent.py
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class SynonymOutput(BaseModel):
    """
    Modelo para validar la salida del LLM que selecciona sinónimos
    """
    selected_synonym: str = Field(
        ...,
        description="The selected synonym selected for the given word, considering context and semantic relevance."
    )


def _looks_like_schema(obj: Any) -> bool:
    """Heurística: detecta si 'obj' parece un JSON Schema en vez de una instancia."""
    if not isinstance(obj, dict):
        return False
    if "selected_synonym" in obj:
        return False
    return (
        "$schema" in obj
        or ("type" in obj and obj.get("type") == "object" and "properties" in obj)
        or ("description" in obj and "properties" in obj and "type" in obj)
    )


async def seleccionar_sinonimo(
    individuo: Dict[str, Any],
    parametro_a_mutar: str,
    palabra_a_sustituir: str,
    sinonimos_disponibles: List[str],
    llm_agent: 'LLMAgent', # Recibe el agente central
    temperatura: float = 0.3
) -> Optional[SynonymOutput]:
    """
    Delega la selección inteligente del sinónimo al LLMAgent.
    """
    return await llm_agent.seleccionar_sinonimo(
        individuo=individuo,
        parametro_a_mutar=parametro_a_mutar,
        palabra_a_sustituir=palabra_a_sustituir,
        sinonimos_disponibles=sinonimos_disponibles,
        temperatura=temperatura
    )