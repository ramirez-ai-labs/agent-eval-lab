"""The only provider-specific code: which model object the agent runs on.

  workers-ai  gpt-oss-120b via Chat Completions (Responses rejects tools there),
              through the compatibility client in workers_ai.py
  openai      the Responses API on api.openai.com
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from agents import OpenAIChatCompletionsModel, OpenAIResponsesModel
from openai import AsyncOpenAI

from tracks.openai.workers_ai import make_client

ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    return yaml.safe_load(open(ROOT / "config" / "models.yaml", encoding="utf-8"))


def make_model(provider: str, cfg: dict) -> tuple[object, str]:
    """Return (model object, model id) for a provider."""
    agent_cfg = cfg["agent"]
    if provider == "workers-ai":
        model_id = agent_cfg["workers_ai_model"]
        client = make_client(os.environ["CF_ACCOUNT_ID"], os.environ["CF_API_TOKEN"])
        return OpenAIChatCompletionsModel(model=model_id, openai_client=client), model_id
    if provider == "openai":
        model_id = agent_cfg["openai_model"]
        if not model_id or model_id == "TODO":
            raise ValueError("Set agent.openai_model in config/models.yaml first.")
        return OpenAIResponsesModel(model=model_id, openai_client=AsyncOpenAI()), model_id
    raise ValueError(f"unknown provider: {provider}")
