"""Layered configuration.

The packaged `default-config.json` holds every setting with a working value.
A site config only needs the keys it changes: it is deep-merged on top, so a
three-line override file stays three lines.

Lookup order for the site config (first hit wins):
    --config PATH
    $SERVICE_TRIMMER_CONFIG
    ./service-trimmer.json in the current directory
"""

from __future__ import annotations

import copy
import json
import os
import platform
from importlib import resources
from pathlib import Path

from .audio import die

ENV_VAR = "SERVICE_TRIMMER_CONFIG"
LOCAL_NAME = "service-trimmer.json"


def deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge `over` into a copy of `base`. Lists are replaced, not joined."""
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def default_config() -> dict:
    text = resources.files("service_trimmer").joinpath("default-config.json").read_text("utf-8")
    return json.loads(text)


def site_config_path(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    if os.environ.get(ENV_VAR):
        return Path(os.environ[ENV_VAR])
    local = Path.cwd() / LOCAL_NAME
    return local if local.exists() else None


def load_config(path: str | None = None) -> dict:
    cfg = default_config()
    site = site_config_path(path)
    if site is None:
        return cfg
    if not site.exists():
        die(f"config file not found: {site}")
    try:
        with open(site, encoding="utf-8") as fh:
            override = json.load(fh)
    except json.JSONDecodeError as exc:
        die(f"{site}: invalid JSON ({exc})")
    cfg = deep_merge(cfg, override)
    cfg["_source"] = str(site)
    return cfg


def asr_model(cfg: dict) -> str:
    """Speech model name. "auto": small.en on ARM (a Pi), medium.en elsewhere."""
    name = cfg.get("transcribe", {}).get("model") or "auto"
    if name != "auto":
        return name
    return "small.en" if platform.machine().lower() in ("aarch64", "arm64") else "medium.en"


def asr_threads(cfg: dict) -> int:
    """CPU threads per transcription. "auto": every core, up to 16."""
    n = cfg.get("transcribe", {}).get("threads", "auto")
    if n in (None, "auto"):
        return min(os.cpu_count() or 4, 16)
    return int(n)


def model_dir(cfg: dict) -> str:
    """Where whisper weights are cached. Defaults to ./models beside the repo checkout."""
    explicit = os.environ.get("SERVICE_TRIMMER_MODEL_DIR") or cfg.get("transcribe", {}).get("model_dir")
    if explicit:
        return str(Path(explicit).expanduser())
    repo = Path(__file__).resolve().parents[2]
    return str(repo / "models")
