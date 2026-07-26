"""AutoDock Vina docking automation and binding-affinity label extraction."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.config import load_config, resolve_paths

# Vina mode table rows look like:
#   1       -7.2      0.000      0.000
# Optionally preceded by whitespace; affinity is column 2 (kcal/mol).
VINA_MODE_ROW_RE = re.compile(
    r"^\s*(\d+)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$",
    re.MULTILINE,
)


class DockingError(RuntimeError):
    """Raised when receptor prep, Vina execution, or log parsing fails."""


def find_vina_binary(vina_binary: str | Path | None = None) -> Path:
    """
    Locate the AutoDock Vina executable.

    Search order:
      1. Explicit ``vina_binary`` argument
      2. ``VINA_BINARY`` environment variable
      3. ``vina`` / ``vina.exe`` on ``PATH``
      4. ``tools/vina.exe`` or ``tools/vina`` under the current working directory
    """
    candidates: list[Path] = []

    if vina_binary:
        candidates.append(Path(vina_binary))

    env_binary = os.environ.get("VINA_BINARY")
    if env_binary:
        candidates.append(Path(env_binary))

    for name in ("vina", "vina.exe"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))

    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / "tools" / "vina.exe",
            cwd / "tools" / "vina",
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise DockingError(
        "AutoDock Vina executable not found. Install the CLI binary and either "
        "add it to PATH, set VINA_BINARY, pass --vina, or place it at tools/vina.exe. "
        "Windows builds: https://github.com/ccsb-scripps/AutoDock-Vina/releases"
    )


def parse_vina_affinity(log_text: str) -> float:
    """
    Extract the Top-1 (most negative) binding affinity from a Vina stdout/log.

    Returns the affinity in kcal/mol as a float.
    """
    scores = [float(match.group(2)) for match in VINA_MODE_ROW_RE.finditer(log_text)]
    if not scores:
        raise DockingError("Could not parse any Vina affinity scores from log output")
    return min(scores)


def pdb_to_pdbqt(pdb_path: Path, pdbqt_path: Path, *, force: bool = False) -> Path:
    """Convert a receptor PDB to PDBQT via the OpenBabel Python API."""
    pdb_path = Path(pdb_path)
    pdbqt_path = Path(pdbqt_path)

    if pdbqt_path.is_file() and pdbqt_path.stat().st_size > 0 and not force:
        return pdbqt_path

    if not pdb_path.is_file():
        raise DockingError(f"Receptor PDB not found: {pdb_path}")

    try:
        from openbabel import openbabel as ob
    except ImportError as exc:
        raise DockingError(
            "OpenBabel Python bindings are required for receptor PDB→PDBQT conversion "
            "(pip install openbabel-wheel)"
        ) from exc

    conv = ob.OBConversion()
    if not conv.SetInFormat("pdb") or not conv.SetOutFormat("pdbqt"):
        raise DockingError("OpenBabel could not set PDB→PDBQT formats")

    # Receptor-style PDBQT: rigid, no torsions / partial charges from ligand prep.
    conv.AddOption("r", ob.OBConversion.OUTOPTIONS)  # rigid molecule
    conv.AddOption("c", ob.OBConversion.OUTOPTIONS)  # combine multi-model into one

    mol = ob.OBMol()
    if not conv.ReadFile(mol, str(pdb_path)):
        raise DockingError(f"OpenBabel could not read receptor PDB: {pdb_path}")

    mol.AddHydrogens()
    pdbqt_path.parent.mkdir(parents=True, exist_ok=True)
    if not conv.WriteFile(mol, str(pdbqt_path)):
        # Some OpenBabel builds write via WriteString more reliably for PDBQT.
        text = conv.WriteString(mol)
        if not text or not text.strip():
            raise DockingError(f"OpenBabel produced empty PDBQT for {pdb_path}")
        pdbqt_path.write_text(text, encoding="utf-8")

    if not pdbqt_path.is_file() or pdbqt_path.stat().st_size == 0:
        raise DockingError(f"Failed to write receptor PDBQT: {pdbqt_path}")
    return pdbqt_path


def run_vina(
    *,
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    out_pdbqt: Path,
    log_path: Path,
    center: dict[str, float],
    size: dict[str, float],
    exhaustiveness: int = 8,
    num_modes: int = 9,
    vina_binary: Path | None = None,
    force: bool = False,
) -> tuple[Path, Path, float]:
    """
    Invoke the Vina CLI for one receptor/ligand pair.

    Returns ``(out_pdbqt, log_path, top1_affinity)``.
    """
    receptor_pdbqt = Path(receptor_pdbqt)
    ligand_pdbqt = Path(ligand_pdbqt)
    out_pdbqt = Path(out_pdbqt)
    log_path = Path(log_path)

    if (
        not force
        and out_pdbqt.is_file()
        and out_pdbqt.stat().st_size > 0
        and log_path.is_file()
        and log_path.stat().st_size > 0
    ):
        affinity = parse_vina_affinity(log_path.read_text(encoding="utf-8", errors="replace"))
        return out_pdbqt, log_path, affinity

    binary = find_vina_binary(vina_binary)
    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(binary),
        "--receptor",
        str(receptor_pdbqt),
        "--ligand",
        str(ligand_pdbqt),
        "--out",
        str(out_pdbqt),
        "--center_x",
        str(center["x"]),
        "--center_y",
        str(center["y"]),
        "--center_z",
        str(center["z"]),
        "--size_x",
        str(size["x"]),
        "--size_y",
        str(size["y"]),
        "--size_z",
        str(size["z"]),
        "--exhaustiveness",
        str(int(exhaustiveness)),
        "--num_modes",
        str(int(num_modes)),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    log_path.write_text(combined, encoding="utf-8")

    if result.returncode != 0:
        raise DockingError(
            f"Vina failed (exit {result.returncode}) for receptor {receptor_pdbqt.name}. "
            f"See log: {log_path}"
        )

    affinity = parse_vina_affinity(combined)
    return out_pdbqt, log_path, affinity


def write_labels_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    """Write docking labels to CSV with columns mutation, vina_score."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["mutation", "vina_score"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "mutation": row["mutation"],
                    "vina_score": f"{float(row['vina_score']):.4f}",
                }
            )
    return output_path


def dock_mutants(
    config: dict[str, Any],
    *,
    root: Path | None = None,
    vina_binary: str | Path | None = None,
    force: bool = False,
) -> Path:
    """
    Dock the configured ligand against each mutant PDB and write ``labels.csv``
    under ``paths.processed``.
    """
    base = Path(root) if root is not None else Path.cwd()
    paths = resolve_paths(config, root=base)
    mutants_dir = paths["mutants"]
    docking_dir = paths["docking"]
    processed_dir = paths["processed"]
    docking_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    docking_cfg = config.get("docking") or {}
    for key in ("center", "size"):
        if key not in docking_cfg:
            raise DockingError(f"Config missing docking.{key}")
    center = docking_cfg["center"]
    size = docking_cfg["size"]
    exhaustiveness = int(docking_cfg.get("exhaustiveness", 8))
    num_modes = int(docking_cfg.get("num_modes", 9))

    drug_name = config["drug"]["name"]
    ligand_slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in drug_name)
    ligand_pdbqt = docking_dir / f"{ligand_slug}.pdbqt"
    if not ligand_pdbqt.is_file():
        raise DockingError(
            f"Ligand PDBQT not found: {ligand_pdbqt}. Run Step 2 (src.download) first."
        )

    binary = find_vina_binary(vina_binary)
    rows: list[dict[str, Any]] = []

    for mutation in config["mutations"]:
        mut_key = mutation.strip().upper()
        mutant_pdb = mutants_dir / f"{mut_key}.pdb"
        if not mutant_pdb.is_file():
            raise DockingError(
                f"Mutant PDB not found: {mutant_pdb}. Run src.openmm_mutator first."
            )

        receptor_pdbqt = docking_dir / f"{mut_key}_receptor.pdbqt"
        out_pdbqt = docking_dir / f"{mut_key}_docked.pdbqt"
        log_path = docking_dir / f"{mut_key}_vina.log"

        print(f"docking {mut_key} ...", flush=True)
        pdb_to_pdbqt(mutant_pdb, receptor_pdbqt, force=force)
        _out, _log, affinity = run_vina(
            receptor_pdbqt=receptor_pdbqt,
            ligand_pdbqt=ligand_pdbqt,
            out_pdbqt=out_pdbqt,
            log_path=log_path,
            center=center,
            size=size,
            exhaustiveness=exhaustiveness,
            num_modes=num_modes,
            vina_binary=binary,
            force=force,
        )
        print(f"  {mut_key}: vina_score={affinity:.3f} kcal/mol", flush=True)
        rows.append({"mutation": mut_key, "vina_score": affinity})

    labels_path = processed_dir / "labels.csv"
    write_labels_csv(rows, labels_path)
    return labels_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 3b: dock mutants with AutoDock Vina and write labels.csv.",
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
        "--vina",
        type=Path,
        default=None,
        help="Path to vina executable (overrides PATH / VINA_BINARY)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run receptor prep and docking even if outputs exist",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = args.root if args.root is not None else Path.cwd()
    config = load_config(args.config)
    try:
        labels_path = dock_mutants(
            config,
            root=root,
            vina_binary=args.vina,
            force=args.force,
        )
    except DockingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"labels: {labels_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
