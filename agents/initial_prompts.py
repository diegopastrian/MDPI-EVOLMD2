# obtener_prompts_agente.py
from typing import List, Optional, Any
from pydantic import BaseModel, Field
import time
import asyncio


# --- Modelo de validación pydantic ---
class PromptOutput(BaseModel):
    """
    Modelo para validar la salida del LLM que genera prompts.
    Debe devolver una instancia con la clave 'prompt'.
    """
    prompt: str = Field(
        ...,
        description="A single high-quality prompt string generated for the given role, topic, and reference text."
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


async def obtener_prompts_agente(
    texto_referencia: str,
    role: str,
    topic: str,
    n: int,
    llm_agent: 'LLMAgent',
    temperatura: float = 0.9,
    deduplicar: bool = True,
    max_reintentos: int = 2,
    sleep_seg: float = 0.0,
) -> List[str]:
    """
    Devuelve hasta n prompts delegando la llamada al LLMAgent.
    """
    prompts: List[str] = []
    vistos = set()

    while len(prompts) < n:
        intentos = 0
        prompt_text: Optional[str] = None

        while intentos <= max_reintentos and prompt_text is None:
            # La llamada a Ollama se delega al agente centralizado
            obj = await llm_agent.generar_prompt_inicial(
                texto_referencia=texto_referencia,
                role=role,
                topic=topic,
                temperatura=temperatura,
            )

            if obj and getattr(obj, "prompt", None):
                p = obj.prompt.strip()
                if p:
                    if deduplicar:
                        clave = p.lower()
                        if clave in vistos:
                            intentos += 1
                            if intentos <= max_reintentos and sleep_seg > 0: await asyncio.sleep(sleep_seg)
                            continue
                        vistos.add(clave)
                    prompt_text = p
                    break

            intentos += 1
            if intentos <= max_reintentos and sleep_seg > 0: await asyncio.sleep(sleep_seg)

        if prompt_text is None:
            break

        prompts.append(prompt_text)

    return prompts
