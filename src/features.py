"""Structural and biochemical feature extraction (Step 4)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

from src.config import load_config, resolve_paths

# Formal net charge at physiological pH (~7).
AA_CHARGE: dict[str, float] = {
    "A": 0.0,
    "R": 1.0,
    "N": 0.0,
    "D": -1.0,
    "C": 0.0,
    "Q": 0.0,
    "E": -1.0,
    "G": 0.0,
    "H": 0.0,
    "I": 0.0,
    "L": 0.0,
    "K": 1.0,
    "M": 0.0,
    "F": 0.0,
    "P": 0.0,
    "S": 0.0,
    "T": 0.0,
    "W": 0.0,
    "Y": 0.0,
    "V": 0.0,
}

# Residue volumes (Å³) from Zamyatnin (1972) / common biophysical tables.
AA_VOLUME: dict[str, float] = {
    "A": 88.6,
    "R": 173.4,
    "N": 114.1,
    "D": 111.1,
    "C": 108.5,
    "Q": 143.8,
    "E": 138.4,
    "G": 60.1,
    "H": 153.2,
    "I": 166.7,
    "L": 166.7,
    "K": 168.6,
    "M": 162.9,
    "F": 189.9,
    "P": 112.7,
    "S": 89.0,
    "T": 116.1,
    "W": 227.8,
    "Y": 193.6,
    "V": 140.0,
}

MUTATION_RE = re.compile(r"^(?P<wt>[A-Za-z])(?P<pos>\d+)(?P<mut>[A-Za-z])$")

_ATOM_RECORD_RE = re.compile(r"^(ATOM  |HETATM)")

FEATURE_COLUMNS = [
    "mutation",
    "residue_position",
    "wt_aa",
    "mut_aa",
    "dist_to_ligand",
    "dist_to_pocket",
    "sasa",
    "delta_charge",
    "delta_volume",
    "vina_score",
]


class FeatureError(RuntimeError):
    """Raised when feature extraction or feature-store assembly fails."""


def parse_mutation(mutation: str) -> tuple[str, int, str]:
    """
    Parse a mutation string such as ``L858R``.

    Returns ``(wt_1letter, residue_number, mut_1letter)``.
    """
    text = mutation.strip().upper()
    match = MUTATION_RE.match(text)
    if not match:
        raise FeatureError(
            f"Invalid mutation '{mutation}'. Expected format like 'L858R'."
        )

    wt = match.group("wt")
    mut = match.group("mut")
    if wt not in AA_CHARGE or mut not in AA_CHARGE:
        raise FeatureError(f"Unknown amino-acid code in mutation '{mutation}'")

    return wt, int(match.group("pos")), mut


def delta_charge(wt_aa: str, mut_aa: str) -> float:
    """Return formal charge change (mut − wt) at pH ~7."""
    wt = wt_aa.strip().upper()
    mut = mut_aa.strip().upper()
    if wt not in AA_CHARGE or mut not in AA_CHARGE:
        raise FeatureError(f"Unknown amino-acid code(s): {wt_aa}, {mut_aa}")
    return AA_CHARGE[mut] - AA_CHARGE[wt]


def delta_volume(wt_aa: str, mut_aa: str) -> float:
    """Return residue volume change in Å³ (mut − wt)."""
    wt = wt_aa.strip().upper()
    mut = mut_aa.strip().upper()
    if wt not in AA_VOLUME or mut not in AA_VOLUME:
        raise FeatureError(f"Unknown amino-acid code(s): {wt_aa}, {mut_aa}")
    return AA_VOLUME[mut] - AA_VOLUME[wt]


def residue_ca_coord(
    pdb_path: Path,
    residue_number: int,
    *,
    chain_id: str | None = None,
) -> np.ndarray:
    """Return the Cα coordinates (Å) of ``residue_number`` from a mutant PDB."""
    pdb_path = Path(pdb_path)
    if not pdb_path.is_file():
        raise FeatureError(f"Mutant PDB not found: {pdb_path}")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    model = structure[0]

    chains = list(model)
    if not chains:
        raise FeatureError(f"No chains found in {pdb_path}")

    if chain_id is None:
        chain = chains[0]
    else:
        try:
            chain = model[chain_id]
        except KeyError as exc:
            raise FeatureError(
                f"Chain '{chain_id}' not found in {pdb_path}"
            ) from exc

    try:
        residue = chain[residue_number]
    except KeyError as exc:
        raise FeatureError(
            f"Residue {residue_number} not found in chain {chain.id} of {pdb_path}"
        ) from exc

    if "CA" not in residue:
        raise FeatureError(
            f"Residue {residue_number} ({residue.get_resname()}) has no CA atom"
        )
    return np.asarray(residue["CA"].coord, dtype=float)


def ligand_centroid(pdbqt_path: Path) -> np.ndarray:
    """
    Compute the centroid of ligand heavy atoms from the Top-1 Vina pose.

    Only ``MODEL 1`` is read when present; hydrogens (element H / type HD) are skipped.
    """
    pdbqt_path = Path(pdbqt_path)
    if not pdbqt_path.is_file():
        raise FeatureError(f"Docked ligand PDBQT not found: {pdbqt_path}")

    coords: list[list[float]] = []
    in_model = False
    saw_model = False

    text = pdbqt_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if line.startswith("MODEL"):
            saw_model = True
            # MODEL 1 is the Top-1 pose; ignore subsequent models.
            try:
                model_num = int(line.split()[1])
            except (IndexError, ValueError):
                model_num = 1
            in_model = model_num == 1
            if model_num > 1:
                break
            continue
        if line.startswith("ENDMDL"):
            if in_model:
                break
            continue
        if saw_model and not in_model:
            continue
        if not _ATOM_RECORD_RE.match(line):
            continue

        # Skip hydrogens: PDB element column or AutoDock atom type.
        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        atom_type = line[77:79].strip().upper() if len(line) >= 79 else ""
        name = line[12:16].strip().upper() if len(line) >= 16 else ""
        if element == "H" or atom_type in {"H", "HD"} or name.startswith("H"):
            continue

        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except (ValueError, IndexError) as exc:
            raise FeatureError(
                f"Could not parse coordinates in {pdbqt_path}: {line[:54]!r}"
            ) from exc
        coords.append([x, y, z])

    if not coords:
        raise FeatureError(f"No ligand atoms found in {pdbqt_path}")
    return np.mean(np.asarray(coords, dtype=float), axis=0)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance (Å) between two 3-vectors."""
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def residue_sasa(
    pdb_path: Path,
    residue_number: int,
    *,
    chain_id: str | None = None,
) -> float:
    """Return total SASA (Å²) for ``residue_number`` via FreeSASA."""
    try:
        import freesasa
    except ImportError as exc:
        raise FeatureError(
            "freesasa is required for SASA features (pip install freesasa)"
        ) from exc

    pdb_path = Path(pdb_path)
    if not pdb_path.is_file():
        raise FeatureError(f"Mutant PDB not found: {pdb_path}")

    structure = freesasa.Structure(str(pdb_path))
    result = freesasa.calc(structure)
    areas = result.residueAreas()

    if chain_id is None:
        if len(areas) == 1:
            chain_areas = next(iter(areas.values()))
        else:
            # Prefer chain A when present; otherwise first chain.
            chain_areas = areas.get("A") or next(iter(areas.values()))
    else:
        if chain_id not in areas:
            raise FeatureError(
                f"Chain '{chain_id}' not found in FreeSASA results for {pdb_path}"
            )
        chain_areas = areas[chain_id]

    key = str(residue_number)
    if key not in chain_areas:
        raise FeatureError(
            f"Residue {residue_number} not found in FreeSASA results for {pdb_path}"
        )
    return float(chain_areas[key].total)


def extract_mutation_features(
    mutation: str,
    *,
    mutant_pdb: Path,
    docked_pdbqt: Path,
    pocket_center: dict[str, float],
    chain_id: str | None = None,
) -> dict[str, Any]:
    """Compute structural and biochemical features for one mutation."""
    wt, pos, mut = parse_mutation(mutation)
    ca = residue_ca_coord(mutant_pdb, pos, chain_id=chain_id)
    lig = ligand_centroid(docked_pdbqt)
    pocket = np.asarray(
        [pocket_center["x"], pocket_center["y"], pocket_center["z"]],
        dtype=float,
    )

    return {
        "mutation": mutation.strip().upper(),
        "residue_position": pos,
        "wt_aa": wt,
        "mut_aa": mut,
        "dist_to_ligand": distance(ca, lig),
        "dist_to_pocket": distance(ca, pocket),
        "sasa": residue_sasa(mutant_pdb, pos, chain_id=chain_id),
        "delta_charge": delta_charge(wt, mut),
        "delta_volume": delta_volume(wt, mut),
    }


def load_labels(labels_path: Path) -> pd.DataFrame:
    """Load ``labels.csv`` produced by Step 3 docking."""
    labels_path = Path(labels_path)
    if not labels_path.is_file():
        raise FeatureError(
            f"Labels file not found: {labels_path}. Run src.docking first."
        )
    df = pd.read_csv(labels_path)
    if "mutation" not in df.columns or "vina_score" not in df.columns:
        raise FeatureError(
            f"labels.csv must contain 'mutation' and 'vina_score' columns: {labels_path}"
        )
    df["mutation"] = df["mutation"].astype(str).str.strip().str.upper()
    return df


def build_feature_store(
    config: dict[str, Any],
    *,
    root: Path | None = None,
    chain_id: str | None = None,
) -> Path:
    """
    Extract features for every configured mutation, merge with Vina labels,
    and write ``features.csv`` under ``paths.processed``.
    """
    base = Path(root) if root is not None else Path.cwd()
    paths = resolve_paths(config, root=base)
    mutants_dir = paths["mutants"]
    docking_dir = paths["docking"]
    processed_dir = paths["processed"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    docking_cfg = config.get("docking") or {}
    if "center" not in docking_cfg:
        raise FeatureError("Config missing docking.center (pocket coordinates)")
    pocket_center = docking_cfg["center"]

    labels = load_labels(processed_dir / "labels.csv")
    label_map = dict(zip(labels["mutation"], labels["vina_score"], strict=False))

    rows: list[dict[str, Any]] = []
    for mutation in config["mutations"]:
        mut_key = mutation.strip().upper()
        mutant_pdb = mutants_dir / f"{mut_key}.pdb"
        docked_pdbqt = docking_dir / f"{mut_key}_docked.pdbqt"

        if mut_key not in label_map:
            raise FeatureError(
                f"No vina_score for {mut_key} in labels.csv. Re-run src.docking."
            )
        if not mutant_pdb.is_file():
            raise FeatureError(
                f"Mutant PDB not found: {mutant_pdb}. Run src.openmm_mutator first."
            )
        if not docked_pdbqt.is_file():
            raise FeatureError(
                f"Docked PDBQT not found: {docked_pdbqt}. Run src.docking first."
            )

        print(f"features {mut_key} ...", flush=True)
        feats = extract_mutation_features(
            mut_key,
            mutant_pdb=mutant_pdb,
            docked_pdbqt=docked_pdbqt,
            pocket_center=pocket_center,
            chain_id=chain_id,
        )
        feats["vina_score"] = float(label_map[mut_key])
        rows.append(feats)
        print(
            f"  {mut_key}: dist_lig={feats['dist_to_ligand']:.2f} "
            f"dist_pocket={feats['dist_to_pocket']:.2f} "
            f"sasa={feats['sasa']:.2f} "
            f"dQ={feats['delta_charge']:+.1f} "
            f"dV={feats['delta_volume']:+.1f}",
            flush=True,
        )

    frame = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    if frame.isna().any().any():
        bad = frame.columns[frame.isna().any()].tolist()
        raise FeatureError(f"Feature store contains NaN values in columns: {bad}")

    out_path = processed_dir / "features.csv"
    frame.to_csv(out_path, index=False, float_format="%.6f")
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Step 4: extract structural/biochemical features and write features.csv."
        ),
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
        "--chain",
        default=None,
        help="PDB chain ID for the mutated residue (default: auto / first chain)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = args.root if args.root is not None else Path.cwd()
    config = load_config(args.config)
    try:
        features_path = build_feature_store(
            config,
            root=root,
            chain_id=args.chain,
        )
    except FeatureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"features: {features_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
