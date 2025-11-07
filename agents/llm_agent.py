# agents/llm_agent.py

import ollama
import json
import time
from typing import List, Dict, Any, Optional

# Importamos los modelos Pydantic de los otros agentes
from .generate_data import DataOutput, _looks_like_schema as looks_like_data_schema
from .initial_prompts import PromptOutput, _looks_like_schema as looks_like_prompt_schema
from .keyword_prompts import KeywordPromptOutput, _looks_like_schema as looks_like_kw_prompt_schema
from .keywords import KeywordsOutput, _looks_like_schema as looks_like_keywords_schema, _normalize_and_filter_keywords, _fallback_extract_from_prompt
from .synonym_selection import SynonymOutput, _looks_like_schema as looks_like_synonym_schema


class LLMAgent:
    """
    Clase centralizada asíncrona para manejar todas las interacciones con el modelo LLM.
    """
    def __init__(self, model: str = "llama3"):
        # print(f"Inicializando LLMAgent")
        self.model = model
        self.client = ollama.AsyncClient(host='http://127.0.0.1:11434')

    async def generar_prompt_inicial(
        self, texto_referencia: str, rol: str, task: str, temperatura: float = 0.9
    ) -> Optional[PromptOutput]:
        """
        Genera un prompt inicial. Lógica extraída de 'initial_prompts.py'.
        """
        system_prompt = f"""
You are an AI prompt engineer.
Your job: create ONE high-quality prompt that:
- Will be given directly to another LLM to generate text.
- Must clearly instruct that LLM on what to produce, how, and from what perspective.
- Fulfills the given task
- Is written from the specified role's perspective
- Is thematically aligned with the reference text
- Must be concise and fit within 2 lines of text (about 1–2 sentences).

STRICT OUTPUT RULES:
- Return a JSON object that is a VALID INSTANCE of this schema (NOT the schema itself).
- The prompt must be a ready-to-use instruction for the LLM — do not describe the prompt or add meta-comments.
- Do NOT include explanations, code fences, or any extra text.
- Do NOT include the words "Role" or "Task" in the output.

JSON Schema:
{PromptOutput.model_json_schema()}
""".strip()
        user_prompt = f"""
Reference text: "{texto_referencia}"
Role: {rol}
Task: {task}
""".strip()
        try:
            response = await self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                format="json",
                options={"temperature": temperatura}
            )
            content = response["message"]["content"]
            if isinstance(content, dict):
                if looks_like_prompt_schema(content): raise ValueError("Model returned schema")
                return PromptOutput.model_validate(content)
            if isinstance(content, str):
                if (("\"properties\"" in content or "\"type\": \"object\"" in content) and "\"prompt\"" not in content): raise ValueError("Model returned schema")
                return PromptOutput.model_validate_json(content)
            raise ValueError("Unexpected response content type")
        except Exception as e:
            print(f"❌ Error en LLMAgent.generar_prompt_inicial: {e}")
            return None

    async def extraer_keywords(
        self, prompt: str, topic: str, rol: str, temperatura: float = 0.3, max_reintentos: int = 1, k_min: int = 5, k_max: int = 12
    ) -> List[str]:
        """
        Extrae keywords de un prompt. Lógica extraída de 'keywords.py'.
        """
        system_prompt = f"""
You are an expert at keyword extraction. Given a role, a topic, and a short prompt, extract {k_min}–{k_max} concise keywords or key phrases.

CRITICAL RULES:
- Every keyword MUST appear verbatim (case-insensitive) inside the given prompt text. Do NOT invent words or synonyms.
- Prefer 1–3 word phrases that will be useful to guide an LLM later (e.g., reuse exact phrases present in the prompt).
- Lowercase all keywords. No hashtags. No trailing punctuation.
- Return ONLY a JSON object that is a VALID INSTANCE of the schema below (NOT the schema itself).
- No explanations, no code fences, no extra text.

JSON Schema:
{KeywordsOutput.model_json_schema()}
""".strip()
        user_prompt = f"role: {rol}\ntopic: {topic}\nprompt: \"{prompt}\"".strip()
        
        intentos = 0
        while intentos <= max_reintentos:
            try:
                resp = await self.client.chat(
                    model=self.model,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    format="json",
                    options={"temperature": temperatura},
                )
                content = resp["message"]["content"]
                parsed: Optional[KeywordsOutput] = None
                if isinstance(content, dict):
                    if looks_like_keywords_schema(content): raise ValueError("Model returned schema")
                    parsed = KeywordsOutput.model_validate(content)
                else:
                    text = str(content)
                    if (("\"properties\"" in text or "\"type\": \"object\"" in text) and "\"keywords\"" not in text): raise ValueError("Model returned schema")
                    parsed = KeywordsOutput.model_validate_json(text)
                
                proposed = parsed.keywords if parsed else []
                filtered = _normalize_and_filter_keywords(proposed, prompt_text=prompt, max_items=k_max)
                if len(filtered) < k_min:
                    extra = _fallback_extract_from_prompt(prompt, k_min=k_min, k_max=k_max)
                    merged = list(dict.fromkeys(filtered + extra)) # Simple merge and dedupe
                    filtered = merged[:k_max]
                return filtered
            except Exception as e:
                intentos += 1
                if intentos > max_reintentos:
                    print(f"❌ Error en LLMAgent.extraer_keywords tras {intentos} intentos: {e}")
                    break
        
        return _fallback_extract_from_prompt(prompt, k_min=k_min, k_max=k_max)

    async def generar_data(
        self, individuo: Dict[str, Any], texto_referencia: str, temperatura: float = 0.7
    ) -> Optional[DataOutput]:
        """
        Genera datos a partir de un individuo. Lógica extraída de 'generate_data.py'.
        """
        system_prompt = f"""
You are an AI text generator. Your job: produce ONE short output text that resembles the reference text in style and tone, but adapted to the given role and topic.

STRICT RULES:
- Output MUST be a JSON object conforming to the schema.
- Do not output the schema itself.
- No explanations, no meta-comments, no code fences.
- The 'data' string must be MAXIMUM two lines (≈1–2 sentences).
- The text must be concise, ready-to-use, and thematically aligned with the reference.

JSON Schema:
{DataOutput.model_json_schema()}
""".strip()
        user_prompt = f"""
Reference text:
\"\"\"{texto_referencia}\"\"\"

Role: {individuo.get('rol')}
Topic: {individuo.get('topic')}
Prompt: {individuo.get('prompt')}
""".strip()
        try:
            response = await self.client.chat(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                format="json",
                options={"temperature": temperatura}
            )
            content = response["message"]["content"]
            if isinstance(content, dict):
                if looks_like_data_schema(content): raise ValueError("Model returned schema")
                return DataOutput.model_validate(content)
            if isinstance(content, str):
                if (("\"properties\"" in content or "\"type\": \"object\"" in content) and "\"data\"" not in content): raise ValueError("Model returned schema")
                return DataOutput.model_validate_json(content)
            raise ValueError("Unexpected response content type")
        except Exception as e:
            print(f"❌ Error en LLMAgent.generar_data: {e}")
            return None

    async def regenerar_prompt(
        self, texto_referencia: str, rol: str, topic: str, keywords: List[str], temperatura: float = 0.9
    ) -> Optional[KeywordPromptOutput]:
        """
        Regenera un prompt a partir de keywords. Lógica de 'keyword_prompts.py'.
        """
        system_prompt = f"""
You are an AI prompt engineer. Your task is to create ONE high-quality instruction prompt that:
- Will be given directly to another LLM to generate text.
- Must be clearly written from the perspective of the specified role.
- Must align with the given topic.
- Must explicitly incorporate and focus on the provided keywords.
- Must be thematically aligned with the reference text.
- The prompt must be concise (1–2 sentences, max 2 lines).

STRICT RULES:
- Output MUST be a JSON object valid against the schema.
- Do NOT return the schema itself.
- Do NOT include explanations, comments, or code fences.

JSON Schema:
{KeywordPromptOutput.model_json_schema()}
""".strip()
        user_prompt = f"""
Reference text:
\"\"\"{texto_referencia}\"\"\"

Role: {rol}
Topic: {topic}
Keywords: {", ".join(keywords)}
""".strip()
        try:
            response = await self.client.chat(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                format="json",
                options={"temperature": temperatura}
            )
            content = response["message"]["content"]
            if isinstance(content, dict):
                if looks_like_kw_prompt_schema(content): raise ValueError("Model returned schema")
                return KeywordPromptOutput.model_validate(content)
            if isinstance(content, str):
                if (("\"properties\"" in content or "\"type\": \"object\"" in content) and "\"prompt\"" not in content): raise ValueError("Model returned schema")
                return KeywordPromptOutput.model_validate_json(content)
            raise ValueError("Unexpected response content type")
        except Exception as e:
            print(f"❌ Error en LLMAgent.regenerar_prompt: {e}")
            return None

    async def seleccionar_sinonimo(
        self, individuo: Dict[str, Any], parametro_a_mutar: str, palabra_a_sustituir: str, sinonimos_disponibles: List[str], temperatura: float = 0.3
    ) -> Optional[SynonymOutput]:
        """
        Selecciona un sinónimo. Lógica de 'synonym_selection.py'.
        """
        system_prompt = f"""
You are a creative AI assistant specializing in genetic algorithms.
Your goal is to perform a MUTATION by selecting a creative and semantically valid synonym for a given word.

CRITICAL RULES:
1.  Your primary goal is to introduce variation. AVOID selecting the original word.
2.  Choose ONE synonym from the provided list that best fits the context.
3.  If and ONLY IF all other options are completely nonsensical, you may return the original word.
4.  Output ONLY a JSON object with the selected synonym, nothing else.

JSON Schema:
{SynonymOutput.model_json_schema()}
""".strip()
        user_prompt = f"""
MUTATION CONTEXT:
- Role: "{individuo.get('rol', '')}"
- Topic: "{individuo.get('topic', '')}"
- Keywords: {individuo.get('keywords', [])}"

TASK:
- Parameter to mutate: "{parametro_a_mutar}"
- Original word: "{palabra_a_sustituir}"
- Available synonyms: {json.dumps(sinonimos_disponibles)}

Select the best synonym from the list to cause a creative mutation.
""".strip()
        try:
            response = await self.client.chat(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                format="json",
                options={"temperature": temperatura}
            )
            content = response["message"]["content"]
            if isinstance(content, dict):
                if looks_like_synonym_schema(content): raise ValueError("Model returned schema")
                return SynonymOutput.model_validate(content)
            if isinstance(content, str):
                if (("\"properties\"" in content or "\"type\": \"object\"" in content) and "\"selected_synonym\"" not in content): raise ValueError("Model returned schema")
                return SynonymOutput.model_validate_json(content)
            raise ValueError("Unexpected response content type")
        except Exception as e:
            print(f"❌ Error en LLMAgent.seleccionar_sinonimo: {e}")
            return None