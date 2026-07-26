"""Unit tests for OpenMM mutator helpers and Vina log parsing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.docking import DockingError, parse_vina_affinity, write_labels_csv
from src.openmm_mutator import MutatorError, mutation_to_pdbfixer, parse_mutation

REPO_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_VINA_LOG = """\
#################################################################
# If you used AutoDock Vina in your work, please cite:          #
#################################################################

Scoring function : vina
Rigid receptor: receptor.pdbqt
Ligand: ligand.pdbqt
Center: 0 0 0
Size: 20 20 20
Exhaustiveness: 8

mode |   affinity | dist from best mode
     | (kcal/mol) | rmsd l.b.| rmsd u.b.
-----+------------+----------+----------
   1       -8.4      0.000      0.000
   2       -7.9      1.812      2.451
   3       -7.1      3.204      5.110
"""


def test_parse_mutation_l858r():
    wt, pos, mut = parse_mutation("L858R")
    assert wt == "LEU"
    assert pos == 858
    assert mut == "ARG"
    assert mutation_to_pdbfixer("l858r") == "LEU-858-ARG"


def test_parse_mutation_invalid():
    with pytest.raises(MutatorError):
        parse_mutation("858R")
    with pytest.raises(MutatorError):
        parse_mutation("LX58R")


def test_parse_vina_affinity_top1_most_negative():
    score = parse_vina_affinity(SAMPLE_VINA_LOG)
    assert score == pytest.approx(-8.4)


def test_parse_vina_affinity_legacy_header():
    legacy = """\
-----+------------+----------+----------+
   | affinity | dist.from | best mode
   | (kcal/mol)| rmsd l.b.| rmsd u.b.
-----+------------+----------+----------+
   1       -9.15     0.000      0.000
   2       -8.00     2.100      3.400
"""
    assert parse_vina_affinity(legacy) == pytest.approx(-9.15)


def test_parse_vina_affinity_empty_raises():
    with pytest.raises(DockingError, match="Could not parse"):
        parse_vina_affinity("no scores here")


def test_write_labels_csv(tmp_path):
    path = write_labels_csv(
        [
            {"mutation": "L858R", "vina_score": -8.4},
            {"mutation": "T790M", "vina_score": -7.25},
        ],
        tmp_path / "labels.csv",
    )
    text = path.read_text(encoding="utf-8")
    assert "mutation,vina_score" in text
    assert "L858R,-8.4000" in text
    assert "T790M,-7.2500" in text


def test_find_vina_binary_uses_env(tmp_path, monkeypatch):
    from src.docking import find_vina_binary

    fake = tmp_path / "vina.exe"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setenv("VINA_BINARY", str(fake))
    with patch("src.docking.shutil.which", return_value=None):
        assert find_vina_binary() == fake.resolve()


def test_egfr_config_has_docking_box():
    from src.config import load_config

    config = load_config(REPO_ROOT / "configs" / "egfr.yaml")
    assert "center" in config["docking"]
    assert "size" in config["docking"]
    for axis in ("x", "y", "z"):
        assert axis in config["docking"]["center"]
        assert axis in config["docking"]["size"]
