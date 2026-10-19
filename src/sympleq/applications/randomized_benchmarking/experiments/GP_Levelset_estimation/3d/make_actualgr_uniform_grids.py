"""Generate uniform 3D GP grids for JSONs in per-seed ActualGR folders.

For each

    .../seed_XXXX/ActualGR/*.json

this writes

    .../seed_XXXX/ActualGR_grid/*_gp_grid_3d.npz

Original JSONs are not modified.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
GP_DIR = SCRIPT_DIR.parent
if str(GP_DIR) not in sys.path:
    sys.path.insert(0, str(GP_DIR))

from make_h2_1_uniform_10_5000_grids import (  # noqa: E402
    GATES_BOUNDS,
    N_GATES_GRID,
    N_QUBITS_GRID,
    N_RATIO_GRID,
    QUBITS_BOUNDS,
    RATIO_BOUNDS,
    output_grid_path,
    save_uniform_grid,
)


DEFAULT_ROOT = Path("Personal") / "Data"
ACTUALGR = "ActualGR"
ACTUALGR_GRID = "ActualGR_grid"


def debug(message: str) -> None:
    print(f"[actualgr-grid] {message}", flush=True)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def parse_targets(values: list[str]) -> tuple[set[str], set[str]]:
    if "all" in values:
        return {"H2_1", "H2_2"}, {"H2-1", "H2-2"}

    fle_devices: set[str] = set()
    ca_devices: set[str] = set()
    for value in values:
        if value == "fle-h2-1":
            fle_devices.add("H2_1")
        elif value == "fle-h2-2":
            fle_devices.add("H2_2")
        elif value == "ca-h2-1":
            ca_devices.add("H2-1")
        elif value == "ca-h2-2":
            ca_devices.add("H2-2")
        else:
            raise ValueError(f"Unknown target: {value}")
    return fle_devices, ca_devices


def actualgr_folders(
    root: Path,
    *,
    fle_devices: set[str],
    ca_devices: set[str],
    q_folders: set[str] | None,
) -> list[Path]:
    folders: list[Path] = []

    fle_root = root / "FLE"
    for device in sorted(fle_devices):
        device_root = fle_root / device
        if not device_root.exists():
            debug(f"skip missing FLE device folder: {device_root}")
            continue
        for q_folder in sorted(path for path in device_root.iterdir() if path.is_dir()):
            if q_folders is not None and q_folder.name not in q_folders:
                continue
            folders.extend(
                path
                for path in sorted(q_folder.glob(f"seed_*/{ACTUALGR}"))
                if path.is_dir()
            )

    ca_root = root / "Cost_aware"
    for device in sorted(ca_devices):
        device_root = ca_root / device
        if not device_root.exists():
            debug(f"skip missing cost-aware device folder: {device_root}")
            continue
        folders.extend(
            path
            for path in sorted(device_root.glob(f"seed_*/{ACTUALGR}"))
            if path.is_dir()
        )

    return folders


def actualgr_jsons(folders: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for folder in folders:
        paths.extend(
            path
            for path in sorted(folder.glob("*.json"))
            if "checkpoint" not in path.stem.lower()
        )
    return paths


def grid_output_for_json(json_path: Path) -> Path:
    actualgr_folder = json_path.parent
    output_folder = actualgr_folder.parent / ACTUALGR_GRID
    return output_grid_path(
        json_path,
        source_root=actualgr_folder,
        output_root=output_folder,
    )


def print_status(json_paths: list[Path], *, write: bool, overwrite: bool) -> None:
    mode = "WRITE" if write else "DRY-RUN"
    debug("=" * 88)
    debug("ActualGR uniform-grid generation")
    debug(f"mode={mode} overwrite={overwrite}")
    debug(
        "grid: "
        f"gates={GATES_BOUNDS} ratio={RATIO_BOUNDS} qubits={QUBITS_BOUNDS} "
        f"shape=({N_QUBITS_GRID}, {N_RATIO_GRID}, {N_GATES_GRID})"
    )
    debug(f"jsons={len(json_paths)}")
    debug("-" * 88)
    for json_path in json_paths:
        output_path = grid_output_for_json(json_path)
        state = "exists" if output_path.exists() else "new"
        debug(f"folder: {display_path(json_path.parent.parent)}")
        debug(f"  source: {display_path(json_path)}")
        debug(f"  output: {display_path(output_path)} [{state}]")
    debug("=" * 88)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["all"],
        choices=["all", "fle-h2-1", "fle-h2-2", "ca-h2-1", "ca-h2-2"],
    )
    parser.add_argument(
        "--q-folders",
        nargs="+",
        default=None,
        help="Optional FLE q folders to include, for example: --q-folders qband_4.",
    )
    parser.add_argument(
        "--json-path",
        action="append",
        type=Path,
        default=None,
        help="Explicit ActualGR JSON path. Can be passed more than once.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.json_path is not None:
        json_paths = args.json_path
    else:
        fle_devices, ca_devices = parse_targets(args.targets)
        q_folders = None if args.q_folders is None else set(args.q_folders)
        folders = actualgr_folders(
            args.root,
            fle_devices=fle_devices,
            ca_devices=ca_devices,
            q_folders=q_folders,
        )
        json_paths = actualgr_jsons(folders)

    if args.limit is not None:
        json_paths = json_paths[: args.limit]

    print_status(json_paths, write=args.write, overwrite=args.overwrite)
    if not args.write:
        debug("status-only dry-run; add --write to generate grids")
        return

    written = skipped = 0
    for index, json_path in enumerate(json_paths, start=1):
        output_path = grid_output_for_json(json_path)
        debug(f"[{index}/{len(json_paths)}] {display_path(json_path)}")
        if output_path.exists() and not args.overwrite:
            debug(f"  skip existing output: {display_path(output_path)}")
            skipped += 1
            continue

        save_uniform_grid(
            json_path,
            source_root=json_path.parent,
            output_root=json_path.parent.parent / ACTUALGR_GRID,
        )
        written += 1

    debug(
        f"done written={written} skipped_existing={skipped} total={len(json_paths)}"
    )


if __name__ == "__main__":
    main()
