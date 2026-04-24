from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml_config(path: Optional[str | Path]) -> Dict[str, Any]:
    if path is None:
        return {}

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    return payload or {}
