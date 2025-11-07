# agents/generate_data.py
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class DataOutput(BaseModel):
    """
    Modelo para validar la salida del LLM que genera data a partir del prompt.
    """
    data: str = Field(
        ...,
        description="Generated text based on the reference text, adapted to the given prompt, role, and topic."
    )


def _looks_like_schema(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    if "data" in obj:
        return False
    return (
        "$schema" in obj
        or ("type" in obj and obj.get("type") == "object" and "properties" in obj)
        or ("description" in obj and "properties" in obj and "type" in obj)
    )


async def generar_data_con_ollama(
    individuo: Dict[str, Any],
    texto_referencia: str,
    llm_agent: 'LLMAgent',
    temperatura: float = 0.7
) -> Optional[DataOutput]:

    return await llm_agent.generar_data(
        individuo=individuo,
        texto_referencia=texto_referencia,
        temperatura=temperatura
    )
