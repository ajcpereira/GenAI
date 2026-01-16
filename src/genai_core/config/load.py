from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from .schema import AppConfig


def load_raw_config(config_path: str) -> Dict[str, Any]:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def load_and_validate_config(config_path: str) -> Tuple[Dict[str, Any], AppConfig]:
    """Load YAML and validate it against the project schema.

    Returns a pair: (raw_dict, parsed_model).
    """
    raw = load_raw_config(config_path)
    model = AppConfig.model_validate(raw)
    return raw, model
