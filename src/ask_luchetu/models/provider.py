from __future__ import annotations

from ask_luchetu.config import settings
from ask_luchetu.models.deepseek import DeepSeekClient
from ask_luchetu.models.interface import Model, ModelSettings
from ask_luchetu.models.ollama import OllamaClient


def load_model(model_settings: ModelSettings = ModelSettings()) -> Model:
    provider = settings.llm_provider

    if provider == "deepseek":
        return DeepSeekClient(
            model_name=settings.llm_model,
            api_key=settings.deepseek_api_key,
            settings=model_settings,
        )

    if provider == "ollama":
        return OllamaClient(
            model_name=settings.llm_model,
            settings=model_settings,
            base_url=settings.ollama_base_url,
        )

    raise ValueError(
        f"Unknown provider '{provider}'. "
        "Set LLM_PROVIDER in your .env to one of: deepseek, ollama"
    )
