"""OpenMM / PDBFixer mutant generation and local AMBER14 minimization."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

# Ensure conda env Library/bin is findable for MKL OpenMP (libiomp5md.dll) when
# numpy.linalg.svd runs inside PDBFixer.addMissingAtoms. Prefer `conda activate
# protein` (puts this on PATH); this is a safety net if activation was skipped.
if sys.platform == "win32":
    _dll_dir = os.path.join(sys.prefix, "Library", "bin")
    if os.path.isdir(_dll_dir):
        _path = os.environ.get("PATH", "")
        if _dll_dir.lower() not in _path.lower().split(os.pathsep):
            os.environ["PATH"] = _dll_dir + os.pathsep + _path

from openmm import LocalEnergyMinimizer, Platform, unit
from openmm.app import ForceField, HBonds, NoCutoff, PDBFile, Simulation
from openmm.openmm import LangevinMiddleIntegrator
from pdbfixer import PDBFixer

from src.config import load_config, resolve_paths

AA1_TO_3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

MUTATION_RE = re.compile(
    r"^(?P<wt>[A-Za-z])(?P<pos>\d+)(?P<mut>[A-Za-z])$"
)


class MutatorError(RuntimeError):
    """Raised when mutation parsing or OpenMM preparation fails."""


def parse_mutation(mutation: str) -> tuple[str, int, str]:
    """
    Parse a mutation string such as ``L858R``.

    Returns ``(wt_3letter, residue_number, mut_3letter)``.
    """
    text = mutation.strip().upper()
    match = MUTATION_RE.match(text)
    if not match:
        raise MutatorError(
            f"Invalid mutation '{mutation}'. Expected format like 'L858R'."
        )

    wt1 = match.group("wt")
    mut1 = match.group("mut")
    if wt1 not in AA1_TO_3 or mut1 not in AA1_TO_3:
        raise MutatorError(f"Unknown amino-acid code in mutation '{mutation}'")

    return AA1_TO_3[wt1], int(match.group("pos")), AA1_TO_3[mut1]


def mutation_to_pdbfixer(mutation: str) -> str:
    """Convert ``L858R`` → PDBFixer ``LEU-858-ARG``."""
    wt3, pos, mut3 = parse_mutation(mutation)
    return f"{wt3}-{pos}-{mut3}"


def _pick_platform() -> Platform:
    """Prefer CUDA / OpenCL when available; otherwise fall back to CPU/Reference."""
    preferred = ("CUDA", "OpenCL", "CPU", "Reference")
    for name in preferred:
        try:
            return Platform.getPlatformByName(name)
        except Exception:
            continue
    return Platform.getPlatform(0)


def mutate_and_minimize(
    pdb_path: Path,
    mutation: str,
    output_pdb: Path,
    *,
    chain_id: str = "A",
    ph: float = 7.0,
    max_iterations: int = 500,
    force: bool = False,
) -> Path:
    """
    Apply one point mutation with PDBFixer, add hydrogens at ``ph``, then run
    an AMBER14 local energy minimization with OBC2 implicit solvent
    (``amber14-all.xml`` + ``implicit/obc2.xml``) and write ``output_pdb``.
    """
    pdb_path = Path(pdb_path)
    output_pdb = Path(output_pdb)
    if output_pdb.is_file() and output_pdb.stat().st_size > 0 and not force:
        return output_pdb

    if not pdb_path.is_file():
        raise MutatorError(f"Input PDB not found: {pdb_path}")

    fixer_mutation = mutation_to_pdbfixer(mutation)
    fixer = PDBFixer(filename=str(pdb_path))
    fixer.applyMutations([fixer_mutation], chain_id)
    fixer.findMissingResidues()
    # AlphaFold models are contiguous; skip gap-filling, only rebuild mutated side chains.
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)

    forcefield = ForceField("amber14-all.xml", "implicit/obc2.xml")
    system = forcefield.createSystem(
        fixer.topology,
        nonbondedMethod=NoCutoff,
        constraints=HBonds,
        rigidWater=True,
    )
    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picoseconds,
    )
    platform = _pick_platform()
    simulation = Simulation(fixer.topology, system, integrator, platform)
    simulation.context.setPositions(fixer.positions)
    LocalEnergyMinimizer.minimize(simulation.context, maxIterations=max_iterations)

    state = simulation.context.getState(getPositions=True)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    with output_pdb.open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(fixer.topology, state.getPositions(), handle, keepIds=True)
    return output_pdb


def generate_mutants(
    config: dict[str, Any],
    *,
    root: Path | None = None,
    input_pdb: Path | None = None,
    chain_id: str = "A",
    ph: float = 7.0,
    max_iterations: int = 500,
    force: bool = False,
) -> dict[str, Path]:
    """
    Loop over ``config['mutations']``, mutate + minimize each, and write PDBs
    under ``paths.mutants`` named ``{mutation}.pdb``.
    """
    base = Path(root) if root is not None else Path.cwd()
    paths = resolve_paths(config, root=base)
    mutants_dir = paths["mutants"]
    mutants_dir.mkdir(parents=True, exist_ok=True)

    if input_pdb is None:
        alphafold_id = config["protein"]["alphafold_id"]
        input_pdb = paths["raw"] / f"{alphafold_id}_clean.pdb"
    input_pdb = Path(input_pdb)

    outputs: dict[str, Path] = {}
    for mutation in config["mutations"]:
        dest = mutants_dir / f"{mutation.strip().upper()}.pdb"
        print(f"mutating {mutation} -> {dest}", flush=True)
        outputs[mutation] = mutate_and_minimize(
            input_pdb,
            mutation,
            dest,
            chain_id=chain_id,
            ph=ph,
            max_iterations=max_iterations,
            force=force,
        )
    return outputs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 3a: mutate WT PDB with PDBFixer and minimize with OpenMM AMBER14.",
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
        "--input-pdb",
        type=Path,
        default=None,
        help="Override WT PDB (default: data/raw/{alphafold_id}_clean.pdb)",
    )
    parser.add_argument(
        "--chain-id",
        default="A",
        help="PDB chain to mutate (default: A)",
    )
    parser.add_argument(
        "--ph",
        type=float,
        default=7.0,
        help="pH for adding missing hydrogens (default: 7.0)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=500,
        help="OpenMM LocalEnergyMinimizer max iterations (default: 500)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-generate mutants even if output PDBs already exist",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = args.root if args.root is not None else Path.cwd()
    config = load_config(args.config)
    try:
        outputs = generate_mutants(
            config,
            root=root,
            input_pdb=args.input_pdb,
            chain_id=args.chain_id,
            ph=args.ph,
            max_iterations=args.max_iterations,
            force=args.force,
        )
    except MutatorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for mutation, path in outputs.items():
        print(f"{mutation}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
