# llm_model.py
import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, AsyncGenerator

# LangChain Core Imports
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq # NEW: Import for Llama models

logger = logging.getLogger(__name__)

# --- NEW: Updated Model Alias Dictionary to match your screenshot ---
_MODEL_ALIASES = {
    # OpenAI
    "gpt4o": "gpt-4o",
    "gpt4omini": "gpt-4o-mini",
    # Google (mapping your UI names to the latest valid models)
    "geminiflash25": "gemini-1.5-flash-latest",
    "geminipro25": "gemini-1.5-pro-latest",
    # Anthropic (mapping your UI name to the valid Haiku model)
    "claude35haiku": "claude-3-haiku-20240307",
    "claude35sonnet": "claude-3-5-sonnet-20240620",
    # Groq for Llama
    "llama38b": "llama3-8b-8192",
    # Note: "Llama 4 Scout" is not on Groq and is not a standard open model.
    # It would require a different provider like OpenRouter to be added.
}

def _canonicalize_model_name(name: str, default: str) -> str:
    if not name:
        return default
    normalized_key = name.lower().replace(" ", "").replace("_", "").replace("-", "").replace(".", "")
    return _MODEL_ALIASES.get(normalized_key, name)

@dataclass
class LLMConfig:
    """Configuration for the LLM Manager, with added Groq API key."""
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", "")) # NEW

    default_model: str = field(default_factory=lambda: os.getenv("DEFAULT_MODEL", "gpt-4o"))
    temperature: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.2"))
    max_tokens: int = int(os.getenv("DEFAULT_MAX_TOKENS", "2000"))
    default_system_prompt: Optional[str] = None

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls()

class LLMManager:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        if not any([cfg.openai_api_key, cfg.anthropic_api_key, cfg.google_api_key, cfg.groq_api_key]):
            raise ValueError("At least one LLM API key must be provided.")

    def get_llm(self, model_name: Optional[str] = None, temperature: Optional[float] = None) -> BaseChatModel:
        model_id = _canonicalize_model_name(name=model_name, default=self.cfg.default_model)
        temp = temperature if temperature is not None else self.cfg.temperature

        # NEW: Groq for Llama models
        if "llama" in model_id:
            if not self.cfg.groq_api_key:
                raise ValueError("GROQ_API_KEY is required to use Llama models.")
            logger.info(f"Instantiating Groq Llama model (resolved from '{model_name}' to '{model_id}')")
            return ChatGroq(
                model_name=model_id,
                groq_api_key=self.cfg.groq_api_key,
                temperature=temp,
                max_tokens=self.cfg.max_tokens,
            )

        # Google Gemini Models
        if "gemini" in model_id:
            if not self.cfg.google_api_key:
                raise ValueError("GOOGLE_API_KEY is required to use Gemini models.")
            logger.info(f"Instantiating Google Gemini model (resolved from '{model_name}' to '{model_id}')")
            return ChatGoogleGenerativeAI(
                model=model_id,
                google_api_key=self.cfg.google_api_key,
                temperature=temp,
                max_output_tokens=self.cfg.max_tokens,
                convert_system_message_to_human=True,
            )

        # Anthropic Claude Models
        if "claude" in model_id:
            if not self.cfg.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY is required to use Claude models.")
            logger.info(f"Instantiating Anthropic Claude model (resolved from '{model_name}' to '{model_id}')")
            return ChatAnthropic(
                model=model_id,
                anthropic_api_key=self.cfg.anthropic_api_key,
                temperature=temp,
                max_tokens=self.cfg.max_tokens,
            )

        # Default to OpenAI GPT Models
        if not self.cfg.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required to use OpenAI models.")
        logger.info(f"Instantiating OpenAI GPT model (resolved from '{model_name}' to '{model_id}')")
        return ChatOpenAI(
            model=model_id,
            api_key=self.cfg.openai_api_key,
            temperature=temp,
            max_tokens=self.cfg.max_tokens,
            streaming=True
        )

    # ... generate_response and other methods remain the same ...
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        stream: bool = False,
        system_prompt: Optional[str] = None,
    ) -> str | AsyncGenerator[str, None]:
        _messages = []
        if system_prompt:
            _messages.append({"role": "system", "content": system_prompt})
        _messages.extend(messages)
        llm = self.get_llm(model_name=model)
        if stream:
            async def iterator():
                async for chunk in llm.astream(_messages):
                    yield chunk.content
            return iterator()
        else:
            resp = await llm.ainvoke(_messages)
            return resp.content

    async def analyze_query(self, query: str, mode: str = "web_search", model: Optional[str] = None) -> Dict[str, Any]:
        sys = "You are a classifier. Output JSON with a single boolean field 'result'."
        prompt = f"Mode={mode}. For the user query: {query!r}, should we do this mode?"
        try:
            msg = [{"role": "system", "content": sys}, {"role": "user", "content": prompt}]
            txt = await self.generate_response(messages=msg, model=model or self.cfg.default_model, stream=False)
            txt_str = str(txt) if txt is not None else ""
            out = {"result": "true" in txt_str.lower() or "yes" in txt_str.lower()}
            return out
        except Exception as e:
            logger.warning(f"analyze_query failed: {e}")
            return {"result": True}

    async def cleanup(self):
        return