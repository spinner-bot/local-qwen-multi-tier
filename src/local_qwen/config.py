from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_REGISTRY = REPO_ROOT / "models.tsv"
DEFAULT_ROOT = Path("/data/Users/zrh")


@dataclass(frozen=True)
class Tier:
    name: str
    model: str
    port: int


def load_tiers(path: Path = MODEL_REGISTRY) -> dict[str, Tier]:
    """Read the tier names and ports from the repository's model registry."""
    tiers: dict[str, Tier] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            name = row.get("tier", "").strip().lower()
            if not name:
                continue
            try:
                port = int(row["port"])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"invalid port for tier {name!r}") from exc
            tiers[name] = Tier(name=name, model=name, port=port)
    if not tiers:
        raise ValueError(f"no tiers found in {path}")
    return tiers


def get_tier(name: str, registry: Path = MODEL_REGISTRY) -> Tier:
    tiers = load_tiers(registry)
    normalized = name.strip().lower()
    try:
        return tiers[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown model tier {name!r}; choose from {', '.join(tiers)}") from exc


def run_dir() -> Path:
    return Path(os.environ.get("LLM_RUN_DIR", os.environ.get("LLM_RUN", DEFAULT_ROOT / "run/llm")))


def memory_db_path() -> Path:
    return Path(os.environ.get("LLM_MEMORY_DB", run_dir() / "chat.sqlite3"))


def api_key() -> str:
    inline = os.environ.get("LLM_API_KEY", "").strip()
    if inline:
        return inline
    key_file = Path(os.environ.get("LLM_API_KEY_FILE", DEFAULT_ROOT / ".secrets/llm_api_key"))
    try:
        return key_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""
    except OSError as exc:
        raise RuntimeError(f"API key file not readable: {key_file}") from exc


def api_base_url(tier: Tier) -> str:
    return os.environ.get("LLM_API_BASE_URL", f"http://127.0.0.1:{tier.port}/v1").rstrip("/")
