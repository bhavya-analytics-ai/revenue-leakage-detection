from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .prompt_builder import build_template_explanation


@dataclass
class LLMConfig:
    mode: str = "template"  # "template" | "openai"
    model: str = "gpt-4o-mini"  # only used if mode="openai"
    max_tokens: int = 180


def generate_explanations(df: pd.DataFrame, cfg: Optional[LLMConfig] = None) -> pd.Series:
    """
    Returns a Series of explanation text.
    Default is template mode (no external calls).

    Note: OpenAI mode is optional and only runs if:
      - cfg.mode == "openai"
      - OPENAI_API_KEY exists
      - openai package installed
    """
    cfg = cfg or LLMConfig()

    if cfg.mode != "openai":
        return df.apply(build_template_explanation, axis=1)

    # Optional OpenAI mode (safe fallback to template on failure)
    try:
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return df.apply(build_template_explanation, axis=1)

        from openai import OpenAI  # requires `pip install openai`
        client = OpenAI(api_key=api_key)

        def _one(row: pd.Series) -> str:
            payload = row.to_dict()

            system = (
                "You are a billing analytics assistant. "
                "Write a concise, human analyst-style explanation of a revenue leakage alert. "
                "Do not invent facts. Use only provided fields. "
                "Keep it 2-4 sentences plus 1 action sentence."
            )

            user = (
                "Explain this invoice leakage alert using these fields:\n"
                f"{payload}\n\n"
                "Output format:\n"
                "- Explanation: ...\n"
                "- Action: ..."
            )

            resp = client.chat.completions.create(
                model=cfg.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=cfg.max_tokens,
                temperature=0.2,
            )
            text = resp.choices[0].message.content.strip()
            return text

        return df.apply(_one, axis=1)

    except Exception:
        return df.apply(build_template_explanation, axis=1)
