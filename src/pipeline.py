"""Run the full offline data-generation + training pipeline (Step 7)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import load_config
from src.docking import dock_mutants
from src.download import prepare_wildtype
from src.features import build_feature_store
from src.openmm_mutator import generate_mutants
from src.train import run_training


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline pipeline: download → mutate/minimize → dock → features → train. "
            "Intended for overnight macro-dataset runs."
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
        "--skip-download",
        action="store_true",
        help="Skip WT / ligand download if raw artifacts already exist",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute mutants / docking even when outputs already exist",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = args.root if args.root is not None else Path.cwd()
    config = load_config(args.config)
    n_mut = len(config["mutations"])
    print(f"offline pipeline: {n_mut} mutations from {args.config}", flush=True)

    if not args.skip_download:
        print("=== Step 2: download / prepare WT + ligand ===", flush=True)
        prepare_wildtype(config, root=root)
    else:
        print("=== Step 2: skipped (--skip-download) ===", flush=True)

    print("=== Step 3a: OpenMM mutate + minimize ===", flush=True)
    generate_mutants(config, root=root, force=args.force)

    print("=== Step 3b: AutoDock Vina docking ===", flush=True)
    dock_mutants(config, root=root, force=args.force)

    print("=== Step 4: feature extraction ===", flush=True)
    build_feature_store(config, root=root)

    print("=== Step 5/7: train surrogate ===", flush=True)
    artifacts = run_training(config, root=root)
    for key, path in artifacts.items():
        print(f"{key}: {path}", flush=True)

    print("offline pipeline complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
