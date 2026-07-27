"""Wild-type preparation: AlphaFold PDB download, cleaning, and ligand PDBQT."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from Bio.PDB import PDBIO, PDBParser, Select
from meeko import MoleculePreparation, PDBQTWriterLegacy
from rdkit import Chem
from rdkit.Chem import AllChem

from src.config import load_config, resolve_paths

ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
ALPHAFOLD_FILE = "https://alphafold.ebi.ac.uk/files/{alphafold_id}-model_{version}.pdb"
ALPHAFOLD_VERSIONS = ("v6", "v4", "v3", "v2", "v1")
PUBCHEM_SDF = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF"
USER_AGENT = "Protein-Surrogate-MLOps/0.1 (research; WT preparation)"

STANDARD_AMINO_ACIDS = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
)


class DownloadError(RuntimeError):
    """Raised when a remote download or file conversion fails."""


class _PolymerResidues(Select):
    """Keep polymer ATOM records for standard amino acids only (no HETATM/water)."""

    def accept_residue(self, residue) -> bool:  # noqa: N802 (Biopython API)
        hetero_flag, _resseq, _icode = residue.id
        return hetero_flag == " " and residue.get_resname() in STANDARD_AMINO_ACIDS


def _http_get(url: str, dest: Path | None = None, timeout: float = 120.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        raise DownloadError(f"HTTP {exc.code} downloading {url}") from exc
    except urllib.error.URLError as exc:
        raise DownloadError(f"Network error downloading {url}: {exc.reason}") from exc

    if dest is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
    return payload


def resolve_alphafold_pdb_url(uniprot_id: str, alphafold_id: str) -> str:
    """Resolve the best available AlphaFold PDB URL for a UniProt accession."""
    api_url = ALPHAFOLD_API.format(uniprot_id=uniprot_id)
    try:
        payload = json.loads(_http_get(api_url).decode("utf-8"))
        entries = payload if isinstance(payload, list) else [payload]
        for entry in entries:
            if entry.get("entryId") == alphafold_id and entry.get("pdbUrl"):
                return str(entry["pdbUrl"])
        for entry in entries:
            if entry.get("pdbUrl"):
                return str(entry["pdbUrl"])
    except DownloadError:
        pass

    for version in ALPHAFOLD_VERSIONS:
        candidate = ALPHAFOLD_FILE.format(alphafold_id=alphafold_id, version=version)
        request = urllib.request.Request(
            candidate,
            method="HEAD",
            headers={"User-Agent": USER_AGENT},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if 200 <= response.status < 300:
                    return candidate
        except (urllib.error.HTTPError, urllib.error.URLError):
            continue

    raise DownloadError(
        f"Could not resolve an AlphaFold PDB URL for {alphafold_id} ({uniprot_id})"
    )


def download_alphafold_pdb(
    uniprot_id: str,
    alphafold_id: str,
    output_dir: Path,
    *,
    force: bool = False,
) -> Path:
    """Download the AlphaFold PDB for a protein into ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f"{alphafold_id}.pdb"

    if dest.is_file() and dest.stat().st_size > 0 and not force:
        return dest

    url = resolve_alphafold_pdb_url(uniprot_id, alphafold_id)
    _http_get(url, dest=dest)
    if dest.stat().st_size == 0:
        raise DownloadError(f"Downloaded empty PDB from {url}")
    return dest


def clean_pdb(pdb_path: Path, output_path: Path | None = None) -> Path:
    """Write a mutation-ready PDB with standard amino-acid ATOM records only."""
    pdb_path = Path(pdb_path)
    dest = (
        Path(output_path)
        if output_path is not None
        else pdb_path.with_name(f"{pdb_path.stem}_clean{pdb_path.suffix}")
    )

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    io = PDBIO()
    io.set_structure(structure)
    dest.parent.mkdir(parents=True, exist_ok=True)
    io.save(str(dest), _PolymerResidues())
    return dest


def _is_flat_conformer(mol: Chem.Mol, z_eps: float = 1e-3) -> bool:
    """Return True when all atoms lie in a plane (typical PubChem 2D SDF)."""
    if mol.GetNumConformers() == 0:
        return True
    conf = mol.GetConformer()
    zs = [conf.GetAtomPosition(i).z for i in range(mol.GetNumAtoms())]
    return (max(zs) - min(zs)) < z_eps


def _embed_3d(mol: Chem.Mol) -> Chem.Mol:
    """Add hydrogens and generate an energy-minimized 3D conformer."""
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    if AllChem.EmbedMolecule(mol, params) != 0:
        if AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=0xC0FFEE) != 0:
            raise DownloadError("3D embedding failed for ligand")
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol)
    return mol


def download_ligand_sdf(
    *,
    name: str,
    pubchem_cid: int | None = None,
    smiles: str | None = None,
    output_dir: Path,
    force: bool = False,
) -> Path:
    """Download (or build from SMILES) a 3D ligand SDF into ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    dest = output_dir / f"{slug}.sdf"

    if dest.is_file() and dest.stat().st_size > 0 and not force:
        return dest

    mol: Chem.Mol | None = None
    if pubchem_cid is not None:
        url = PUBCHEM_SDF.format(cid=int(pubchem_cid))
        _http_get(url, dest=dest)
        supplier = Chem.SDMolSupplier(str(dest), removeHs=False)
        mol = next((m for m in supplier if m is not None), None)

    if mol is None:
        if not smiles:
            raise DownloadError(
                f"Failed to obtain SDF for '{name}' and no SMILES fallback was provided"
            )
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise DownloadError(f"Could not parse SMILES for '{name}'")

    if _is_flat_conformer(mol):
        try:
            mol = _embed_3d(mol)
        except DownloadError as exc:
            raise DownloadError(f"3D embedding failed for ligand '{name}'") from exc
    elif mol.GetNumAtoms() > 0 and mol.GetAtomWithIdx(0).GetTotalNumHs() == 0:
        # Keep existing 3D coords; ensure explicit hydrogens for PDBQT prep.
        mol = Chem.AddHs(mol, addCoords=True)

    writer = Chem.SDWriter(str(dest))
    writer.write(mol)
    writer.close()
    return dest


def _sdf_to_pdbqt_openbabel_python(sdf_path: Path, dest: Path) -> Path | None:
    """Convert SDF→PDBQT with the OpenBabel Python bindings. Returns None if unavailable."""
    try:
        from openbabel import openbabel as ob
    except ImportError:
        return None

    conv = ob.OBConversion()
    if not conv.SetInFormat("sdf") or not conv.SetOutFormat("pdbqt"):
        return None

    obmol = ob.OBMol()
    if not conv.ReadFile(obmol, str(sdf_path)):
        raise DownloadError(f"OpenBabel could not read SDF: {sdf_path}")

    obmol.AddHydrogens()
    pdbqt_string = conv.WriteString(obmol)
    if not pdbqt_string or not pdbqt_string.strip():
        raise DownloadError(f"OpenBabel produced empty PDBQT for {sdf_path}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(pdbqt_string, encoding="utf-8")
    return dest


def _sdf_to_pdbqt_openbabel_cli(sdf_path: Path, dest: Path) -> Path | None:
    """Convert SDF→PDBQT with the ``obabel`` CLI. Returns None if unavailable."""
    obabel = shutil.which("obabel") or shutil.which("obabel.exe")
    if not obabel:
        return None

    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [obabel, str(sdf_path), "-O", str(dest), "-h"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0 and dest.is_file() and dest.stat().st_size > 0:
        return dest
    return None


def _sdf_to_pdbqt_meeko(mol: Chem.Mol, dest: Path) -> Path:
    preparator = MoleculePreparation()
    setups = preparator.prepare(mol)
    if not setups:
        raise DownloadError("Meeko produced no MoleculeSetup for ligand")
    pdbqt_string, success, error_msg = PDBQTWriterLegacy.write_string(setups[0])
    if not success:
        raise DownloadError(f"Meeko PDBQT write failed: {error_msg}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(pdbqt_string, encoding="utf-8")
    return dest


def sdf_to_pdbqt(
    sdf_path: Path,
    pdbqt_path: Path | None = None,
    *,
    force: bool = False,
) -> Path:
    """
    Convert a ligand SDF to PDBQT.

    Preference order:
      1. OpenBabel Python API (``openbabel`` / ``openbabel-wheel``)
      2. OpenBabel CLI (``obabel``)
      3. Meeko / RDKit fallback
    """
    sdf_path = Path(sdf_path)
    dest = Path(pdbqt_path) if pdbqt_path is not None else sdf_path.with_suffix(".pdbqt")

    if dest.is_file() and dest.stat().st_size > 0 and not force:
        return dest

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = next((m for m in supplier if m is not None), None)
    if mol is None:
        raise DownloadError(f"Could not read molecule from {sdf_path}")

    if _is_flat_conformer(mol):
        mol = _embed_3d(mol)
        three_d_sdf = sdf_path.with_name(f"{sdf_path.stem}_3d.sdf")
        writer = Chem.SDWriter(str(three_d_sdf))
        writer.write(mol)
        writer.close()
        sdf_for_obabel = three_d_sdf
    else:
        mol = Chem.AddHs(mol, addCoords=True)
        sdf_for_obabel = sdf_path

    for converter in (
        _sdf_to_pdbqt_openbabel_python,
        _sdf_to_pdbqt_openbabel_cli,
    ):
        result = converter(sdf_for_obabel, dest)
        if result is not None:
            return result

    return _sdf_to_pdbqt_meeko(mol, dest)


def prepare_wildtype(
    config: dict[str, Any],
    *,
    root: Path | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """
    Run Step 2 WT preparation.

    1. Download AlphaFold PDB + ligand SDF into ``paths.raw``.
    2. Clean the PDB (drop heteroatoms / water).
    3. Convert the ligand SDF to PDBQT under ``paths.docking``.
    """
    base = Path(root) if root is not None else Path.cwd()
    paths = resolve_paths(config, root=base)
    raw_dir = paths["raw"]
    docking_dir = paths["docking"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    docking_dir.mkdir(parents=True, exist_ok=True)

    protein = config["protein"]
    drug = config["drug"]

    pdb_raw = download_alphafold_pdb(
        protein["uniprot_id"],
        protein["alphafold_id"],
        raw_dir,
        force=force,
    )
    pdb_clean = clean_pdb(pdb_raw, raw_dir / f"{protein['alphafold_id']}_clean.pdb")

    sdf_path = download_ligand_sdf(
        name=drug["name"],
        pubchem_cid=drug.get("pubchem_cid"),
        smiles=drug.get("smiles"),
        output_dir=raw_dir,
        force=force,
    )
    ligand_slug = "".join(
        ch if ch.isalnum() or ch in "-_" else "_" for ch in drug["name"]
    )
    pdbqt_path = sdf_to_pdbqt(
        sdf_path,
        docking_dir / f"{ligand_slug}.pdbqt",
        force=force,
    )

    return {
        "pdb_raw": pdb_raw,
        "pdb_clean": pdb_clean,
        "ligand_sdf": sdf_path,
        "ligand_pdbqt": pdbqt_path,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 2: download WT AlphaFold structure and prepare ligand PDBQT.",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/egfr.yaml",
        help="Path to pipeline YAML config (default: configs/egfr.yaml)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root for resolving relative paths (default: cwd)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download / re-convert even if outputs already exist",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = args.root if args.root is not None else Path.cwd()
    config = load_config(args.config)
    try:
        artifacts = prepare_wildtype(config, root=root, force=args.force)
    except DownloadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for key, path in artifacts.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
