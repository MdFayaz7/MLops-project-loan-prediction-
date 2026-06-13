import yaml
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(ROOT_DIR / config_path) as f:
        return yaml.safe_load(f)