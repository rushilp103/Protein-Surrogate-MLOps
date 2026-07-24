"""YAML configuration loader and validator for the offline pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_TOP_LEVEL = ("protein", "drug", "mutations", "paths")
REQUIRED_PROTEIN_FIELDS = ("name", "uniprot_id", "alphafold_id")
REQUIRED_DRUG_FIELDS = ("name",)
REQUIRED_PATH_KEYS = (
    "raw",
    "mutants",
    "docking",
    "processed",
    "models",
    "outputs",
)


class ConfigError(ValueError):
    """Raised when a pipeline config is missing required fields or is invalid."""


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"'{field}' must be a mapping, got {type(value).__name__}")
    return value


def _require_keys(mapping: dict[str, Any], keys: tuple[str, ...], parent: str) -> None:
    missing = [key for key in keys if key not in mapping or mapping[key] in (None, "")]
    if missing:
        raise ConfigError(f"Missing required field(s) under '{parent}': {', '.join(missing)}")


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate a loaded config dict and return it unchanged if valid."""
    if not isinstance(config, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(config).__name__}")

    missing_top = [key for key in REQUIRED_TOP_LEVEL if key not in config]
    if missing_top:
        raise ConfigError(f"Missing required field(s): {', '.join(missing_top)}")

    protein = _require_mapping(config["protein"], "protein")
    _require_keys(protein, REQUIRED_PROTEIN_FIELDS, "protein")

    drug = _require_mapping(config["drug"], "drug")
    _require_keys(drug, REQUIRED_DRUG_FIELDS, "drug")

    mutations = config["mutations"]
    if not isinstance(mutations, list) or not mutations:
        raise ConfigError("'mutations' must be a non-empty list")
    if not all(isinstance(m, str) and m.strip() for m in mutations):
        raise ConfigError("Each mutation must be a non-empty string (e.g. 'L858R')")

    paths = _require_mapping(config["paths"], "paths")
    _require_keys(paths, REQUIRED_PATH_KEYS, "paths")

    return config


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML pipeline config from disk."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if raw is None:
        raise ConfigError(f"Config file is empty: {config_path}")

    return validate_config(raw)


def resolve_paths(config: dict[str, Any], root: str | Path | None = None) -> dict[str, Path]:
    """Resolve configured relative paths against the project root."""
    base = Path(root) if root is not None else Path.cwd()
    return {name: (base / rel).resolve() for name, rel in config["paths"].items()}
