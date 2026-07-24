"""Unit tests for wild-type download / ligand preparation helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from src.download import (
    DownloadError,
    clean_pdb,
    download_alphafold_pdb,
    download_ligand_sdf,
    prepare_wildtype,
    resolve_alphafold_pdb_url,
    sdf_to_pdbqt,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_minimal_pdb(path: Path) -> Path:
    path.write_text(
        "ATOM      1  N   ALA A   1      11.104  13.311  10.000  1.00 50.00           N\n"
        "ATOM      2  CA  ALA A   1      12.560  13.300  10.000  1.00 50.00           C\n"
        "ATOM      3  C   ALA A   1      13.086  14.700  10.000  1.00 50.00           C\n"
        "ATOM      4  O   ALA A   1      12.300  15.700  10.000  1.00 50.00           O\n"
        "ATOM      5  CB  ALA A   1      13.086  12.500   8.800  1.00 50.00           C\n"
        "ATOM      6  N   SER A   2      14.400  14.900  10.000  1.00 50.00           N\n"
        "ATOM      7  CA  SER A   2      15.000  16.200  10.000  1.00 50.00           C\n"
        "ATOM      8  C   SER A   2      16.500  16.100  10.000  1.00 50.00           C\n"
        "ATOM      9  O   SER A   2      17.100  15.000  10.000  1.00 50.00           O\n"
        "ATOM     10  CB  SER A   2      14.500  17.000   8.800  1.00 50.00           C\n"
        "ATOM     11  OG  SER A   2      13.100  17.100   8.800  1.00 50.00           O\n"
        "HETATM   12  O   HOH A 101      20.000  20.000  20.000  1.00 30.00           O\n"
        "END\n",
        encoding="utf-8",
    )
    return path


def test_clean_pdb_drops_heteroatoms(tmp_path):
    raw = _write_minimal_pdb(tmp_path / "raw.pdb")
    cleaned = clean_pdb(raw, tmp_path / "clean.pdb")
    text = cleaned.read_text(encoding="utf-8")
    assert "HOH" not in text
    assert "HETATM" not in text
    assert "ALA" in text
    assert "SER" in text


def test_sdf_to_pdbqt_writes_vina_fields(tmp_path):
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None
    mol = Chem.AddHs(mol)
    assert AllChem.EmbedMolecule(mol, randomSeed=1) == 0
    sdf = tmp_path / "ethanol.sdf"
    writer = Chem.SDWriter(str(sdf))
    writer.write(mol)
    writer.close()

    pdbqt = sdf_to_pdbqt(sdf, tmp_path / "ethanol.pdbqt")
    text = pdbqt.read_text(encoding="utf-8")
    assert "ATOM" in text
    # OpenBabel and Meeko both emit usable PDBQT; field names differ slightly.
    assert ("TORSDOF" in text) or ("ROOT" in text) or ("REMARK" in text)
    zs = [float(line.split()[7]) for line in text.splitlines() if line.startswith("ATOM")]
    assert max(zs) - min(zs) > 1e-3 or len(zs) <= 1


def test_sdf_to_pdbqt_prefers_openbabel_python(tmp_path, monkeypatch):
    pytest.importorskip("openbabel")

    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None
    mol = Chem.AddHs(mol)
    assert AllChem.EmbedMolecule(mol, randomSeed=1) == 0
    sdf = tmp_path / "ethanol.sdf"
    writer = Chem.SDWriter(str(sdf))
    writer.write(mol)
    writer.close()

    called = {"meeko": False}

    def boom_meeko(*_args, **_kwargs):
        called["meeko"] = True
        raise AssertionError("Meeko fallback should not run when OpenBabel works")

    monkeypatch.setattr("src.download._sdf_to_pdbqt_meeko", boom_meeko)
    monkeypatch.setattr("src.download._sdf_to_pdbqt_openbabel_cli", lambda *_a, **_k: None)

    pdbqt = sdf_to_pdbqt(sdf, tmp_path / "ethanol.pdbqt", force=True)
    assert pdbqt.is_file()
    assert not called["meeko"]
    assert "ATOM" in pdbqt.read_text(encoding="utf-8")


def test_download_ligand_sdf_from_smiles(tmp_path):
    path = download_ligand_sdf(
        name="toy",
        pubchem_cid=None,
        smiles="CCO",
        output_dir=tmp_path,
        force=True,
    )
    assert path.is_file()
    assert path.suffix == ".sdf"
    mols = [m for m in Chem.SDMolSupplier(str(path)) if m is not None]
    assert len(mols) == 1
    conf = mols[0].GetConformer()
    zs = [conf.GetAtomPosition(i).z for i in range(mols[0].GetNumAtoms())]
    assert max(zs) - min(zs) > 1e-3


def test_resolve_alphafold_pdb_url_uses_api():
    fake_payload = b'[{"entryId":"AF-P00533-F1","pdbUrl":"https://example.test/egfr.pdb"}]'

    with patch("src.download._http_get", return_value=fake_payload):
        url = resolve_alphafold_pdb_url("P00533", "AF-P00533-F1")
    assert url == "https://example.test/egfr.pdb"


def test_download_alphafold_pdb_writes_file(tmp_path):
    pdb_bytes = (
        b"ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 50.00           N\n"
        b"END\n"
    )

    with (
        patch(
            "src.download.resolve_alphafold_pdb_url",
            return_value="https://example.test/x.pdb",
        ),
        patch(
            "src.download._http_get",
            side_effect=lambda url, dest=None, timeout=120.0: (
                dest.write_bytes(pdb_bytes) or pdb_bytes
            ),
        ),
    ):
        path = download_alphafold_pdb("P00533", "AF-P00533-F1", tmp_path, force=True)

    assert path == tmp_path / "AF-P00533-F1.pdb"
    assert path.read_bytes() == pdb_bytes


def test_prepare_wildtype_end_to_end_mocked(tmp_path):
    from src.config import load_config

    config = load_config(REPO_ROOT / "configs" / "egfr.yaml")
    config = {
        **config,
        "paths": {
            "raw": "raw",
            "mutants": "mutants",
            "docking": "docking",
            "processed": "processed",
            "models": "models",
            "outputs": "outputs",
        },
    }

    def fake_download_alphafold(*_args, **_kwargs):
        raw = tmp_path / "raw"
        raw.mkdir(exist_ok=True)
        return _write_minimal_pdb(raw / "AF-P00533-F1.pdb")

    with (
        patch("src.download.download_alphafold_pdb", side_effect=fake_download_alphafold),
        patch(
            "src.download.download_ligand_sdf",
            side_effect=lambda **kwargs: download_ligand_sdf(
                name=kwargs["name"],
                pubchem_cid=None,
                smiles="CCO",
                output_dir=kwargs["output_dir"],
                force=True,
            ),
        ),
    ):
        artifacts = prepare_wildtype(config, root=tmp_path, force=True)

    assert artifacts["pdb_clean"].is_file()
    assert "HOH" not in artifacts["pdb_clean"].read_text(encoding="utf-8")
    assert artifacts["ligand_pdbqt"].is_file()
    assert "ATOM" in artifacts["ligand_pdbqt"].read_text(encoding="utf-8")


def test_download_error_on_network_failure():
    with patch(
        "src.download.urllib.request.urlopen",
        side_effect=__import__("urllib.error").error.URLError("offline"),
    ):
        with pytest.raises(DownloadError):
            from src.download import _http_get

            _http_get("https://example.test/missing")
