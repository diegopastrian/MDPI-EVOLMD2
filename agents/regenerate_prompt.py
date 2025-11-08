# agents/keyword_prompts.py
from typing import List, Optional, Any
from pydantic import BaseModel, Field
import time


# --- Modelo de validación Pydantic ---
class RegeneratePromptOutput(BaseModel):
    """
    Modelo para validar la salida del LLM que genera prompts en base a individuo.
    """
    prompt: str = Field(
        ...,
        description="A single high-quality prompt string generated using the given keywords, role, topic, and reference text."
    )


def _looks_like_schema(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    if "prompt" in obj: 
        return False
    return (
        "$schema" in obj
        or ("type" in obj and obj.get("type") == "object" and "properties" in obj)
        or ("description" in obj and "properties" in obj and "type" in obj)
    )


async def obtener_prompt_regenerado(
    texto_referencia: str,
    role: str,
    topic: str,
    keywords: List[str],
    llm_agent: 'LLMAgent',
    n: int = 1,
    temperatura: float = 0.9,
    max_reintentos: int = 2,
    sleep_seg: float = 0.0,
) -> List[str]:
    """
    Devuelve hasta n prompts generados a partir de un individuo, delegando al LLMAgent.
    """
    prompts: List[str] = []
    vistos = set()

    while len(prompts) < n:
        intentos = 0
        prompt_text: Optional[str] = None

        while intentos <= max_reintentos and prompt_text is None:
            obj = await llm_agent.regenerar_prompt(
                texto_referencia=texto_referencia,
                role=role,
                topic=topic,
                keywords=keywords,
                temperatura=temperatura,
            )

            if obj and getattr(obj, "prompt", None):
                p = obj.prompt.strip()
                if p and p.lower() not in vistos:
                    vistos.add(p.lower())
                    prompt_text = p
                    break

            intentos += 1
            if intentos <= max_reintentos and sleep_seg > 0: time.sleep(sleep_seg)

        if prompt_text is None: break
        prompts.append(prompt_text)

    return prompts
