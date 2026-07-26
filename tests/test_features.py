"""Unit and integration tests for Step 4 feature extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_COLUMNS,
    FeatureError,
    build_feature_store,
    delta_charge,
    delta_volume,
    extract_mutation_features,
    ligand_centroid,
    parse_mutation,
    residue_ca_coord,
    residue_sasa,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EGFR_CONFIG = REPO_ROOT / "configs" / "egfr.yaml"
MUTANT_L858R = REPO_ROOT / "data" / "mutants" / "L858R.pdb"
DOCKED_L858R = REPO_ROOT / "data" / "docking" / "L858R_docked.pdbqt"
LABELS = REPO_ROOT / "data" / "processed" / "labels.csv"


def test_parse_mutation_l858r():
    wt, pos, mut = parse_mutation("L858R")
    assert wt == "L"
    assert pos == 858
    assert mut == "R"


def test_parse_mutation_invalid():
    with pytest.raises(FeatureError):
        parse_mutation("858R")
    with pytest.raises(FeatureError):
        parse_mutation("LX58R")


def test_delta_charge_leu_to_arg():
    # L (0) → R (+1)
    assert delta_charge("L", "R") == pytest.approx(1.0)
    assert delta_charge("l", "r") == pytest.approx(1.0)


def test_delta_volume_leu_to_arg():
    # Zamyatnin volumes: L=166.7, R=173.4 → Δ=+6.7
    assert delta_volume("L", "R") == pytest.approx(6.7)


def test_delta_charge_and_volume_other_pairs():
    assert delta_charge("D", "K") == pytest.approx(2.0)  # -1 → +1
    assert delta_charge("K", "D") == pytest.approx(-2.0)
    assert delta_volume("G", "W") == pytest.approx(227.8 - 60.1)


@pytest.mark.skipif(not MUTANT_L858R.is_file(), reason="L858R mutant PDB missing")
def test_residue_ca_coord_l858r():
    ca = residue_ca_coord(MUTANT_L858R, 858)
    assert ca.shape == (3,)
    assert np.isfinite(ca).all()


@pytest.mark.skipif(not DOCKED_L858R.is_file(), reason="L858R docked PDBQT missing")
def test_ligand_centroid_l858r():
    centroid = ligand_centroid(DOCKED_L858R)
    assert centroid.shape == (3,)
    assert np.isfinite(centroid).all()


@pytest.mark.skipif(not MUTANT_L858R.is_file(), reason="L858R mutant PDB missing")
def test_residue_sasa_l858r():
    sasa = residue_sasa(MUTANT_L858R, 858)
    assert sasa > 0.0
    assert np.isfinite(sasa)


@pytest.mark.skipif(
    not (MUTANT_L858R.is_file() and DOCKED_L858R.is_file()),
    reason="L858R mutant/docked artifacts missing",
)
def test_extract_mutation_features_schema_no_nan():
    feats = extract_mutation_features(
        "L858R",
        mutant_pdb=MUTANT_L858R,
        docked_pdbqt=DOCKED_L858R,
        pocket_center={"x": -50.0, "y": 0.0, "z": -20.0},
    )
    expected = {
        "mutation",
        "residue_position",
        "wt_aa",
        "mut_aa",
        "dist_to_ligand",
        "dist_to_pocket",
        "sasa",
        "delta_charge",
        "delta_volume",
    }
    assert set(feats) == expected
    assert feats["mutation"] == "L858R"
    assert feats["residue_position"] == 858
    assert feats["wt_aa"] == "L"
    assert feats["mut_aa"] == "R"
    assert feats["delta_charge"] == pytest.approx(1.0)
    assert feats["delta_volume"] == pytest.approx(6.7)
    for key in (
        "dist_to_ligand",
        "dist_to_pocket",
        "sasa",
        "delta_charge",
        "delta_volume",
    ):
        assert feats[key] is not None
        assert np.isfinite(feats[key])


@pytest.mark.skipif(
    not (
        MUTANT_L858R.is_file()
        and DOCKED_L858R.is_file()
        and LABELS.is_file()
        and (REPO_ROOT / "data" / "mutants" / "T790M.pdb").is_file()
        and (REPO_ROOT / "data" / "mutants" / "G719S.pdb").is_file()
        and (REPO_ROOT / "data" / "docking" / "T790M_docked.pdbqt").is_file()
        and (REPO_ROOT / "data" / "docking" / "G719S_docked.pdbqt").is_file()
    ),
    reason="Full EGFR micro-dataset artifacts missing",
)
def test_build_feature_store_integration(tmp_path):
    from src.config import load_config

    config = load_config(EGFR_CONFIG)
    # Restrict to the micro-dataset artifacts present under data/, even when
    # configs/egfr.yaml lists the Step 7 macro mutation set.
    config = {
        **config,
        "mutations": ["L858R", "T790M", "G719S"],
        "paths": {
            **config["paths"],
            "processed": str(tmp_path / "processed"),
        },
    }
    (tmp_path / "processed").mkdir(parents=True)
    # Copy labels so merge works against the temp processed path.
    labels_text = LABELS.read_text(encoding="utf-8")
    (tmp_path / "processed" / "labels.csv").write_text(labels_text, encoding="utf-8")

    out = build_feature_store(config, root=REPO_ROOT)
    assert out.is_file()

    df = pd.read_csv(out)
    assert list(df.columns) == FEATURE_COLUMNS
    assert len(df) == 3
    assert not df.isna().any().any()
    assert set(df["mutation"]) == {"L858R", "T790M", "G719S"}
    assert df.loc[df["mutation"] == "L858R", "delta_charge"].iloc[0] == pytest.approx(1.0)
