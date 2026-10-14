"""Load the YAML configuration into a plain nested dict with resolved paths."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Relative paths in the config are relative to the project root.
    cfg["data"]["cache_dir"] = PROJECT_ROOT / cfg["data"]["cache_dir"]
    return cfg
