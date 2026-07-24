"""Unit tests for the YAML configuration parser."""

from pathlib import Path

import pytest
import yaml

from src.config import ConfigError, load_config, resolve_paths, validate_config

REPO_ROOT = Path(__file__).resolve().parents[1]
EGFR_CONFIG = REPO_ROOT / "configs" / "egfr.yaml"


def test_egfr_yaml_loads_successfully():
    config = load_config(EGFR_CONFIG)

    assert config["protein"]["name"] == "EGFR"
    assert config["drug"]["name"] == "Gefitinib"
    assert config["mutations"] == ["L858R", "T790M", "G719S"]
    assert set(config["paths"]) >= {
        "raw",
        "mutants",
        "docking",
        "processed",
        "models",
        "outputs",
    }


@pytest.mark.parametrize("missing_field", ["protein", "drug"])
def test_missing_required_top_level_fields_raise(missing_field, tmp_path):
    with EGFR_CONFIG.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    del data[missing_field]

    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(yaml.dump(data), encoding="utf-8")

    with pytest.raises(ConfigError, match=missing_field):
        load_config(bad_path)


def test_missing_nested_protein_field_raises():
    with EGFR_CONFIG.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    del data["protein"]["name"]

    with pytest.raises(ConfigError, match="protein"):
        validate_config(data)


def test_empty_mutations_raise():
    with EGFR_CONFIG.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    data["mutations"] = []

    with pytest.raises(ConfigError, match="mutations"):
        validate_config(data)


def test_missing_config_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config(REPO_ROOT / "configs" / "does_not_exist.yaml")


def test_resolve_paths_against_repo_root():
    config = load_config(EGFR_CONFIG)
    paths = resolve_paths(config, root=REPO_ROOT)

    assert paths["raw"] == (REPO_ROOT / "data" / "raw").resolve()
    assert paths["processed"] == (REPO_ROOT / "data" / "processed").resolve()
