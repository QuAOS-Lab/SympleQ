"""Build every H2-2 checkpoint input needed by Plot 1."""

from pathlib import Path

from make_actualgr_uniform_grids import ACTUALGR_GRID, save_uniform_grid
from make_costaware_native_actualgr_grids import make_grid as make_pulse_grid
from rewrite_actualgr_actual_gates import main as rewrite_actual_gr


DATA_ROOT = Path("Personal/Data")
FLARE_ROOT = DATA_ROOT / "FLE/H2_2/qband_4"
PULSE_ROOT = DATA_ROOT / "Cost_aware/H2-2"


def actual_gr_files(root: Path) -> list[Path]:
    return sorted(root.glob("seed_*/ActualGR/*_actual_gates.json"))


def main() -> None:
    # Keep existing rewrites, but create every missing checkpoint ActualGR JSON.
    status = rewrite_actual_gr(
        [
            "--root",
            str(DATA_ROOT),
            "--targets",
            "fle-h2-2",
            "ca-h2-2",
            "--q-folders",
            "qband_4",
            "--all-fle-checkpoints",
            "--all-ca-checkpoints",
            "--allow-ambiguous-fle",
            "--skip-existing",
            "--write",
        ]
    )
    if status:
        raise RuntimeError("ActualGR generation did not complete")

    flare_jsons = actual_gr_files(FLARE_ROOT)
    pulse_jsons = actual_gr_files(PULSE_ROOT)
    if not flare_jsons or not pulse_jsons:
        raise FileNotFoundError("Missing H2-2 FLARE or PULSE ActualGR JSONs")

    # FLARE's native GP, evaluated on the shared uniform 10--5000 grid.
    for index, json_path in enumerate(flare_jsons, start=1):
        print(f"[FLARE grid {index}/{len(flare_jsons)}] {json_path}", flush=True)
        save_uniform_grid(
            json_path,
            source_root=json_path.parent,
            output_root=json_path.parent.parent / ACTUALGR_GRID,
        )

    # PULSE's cost-aware posterior on the same coordinate bounds.
    for index, json_path in enumerate(pulse_jsons, start=1):
        print(f"[PULSE grid {index}/{len(pulse_jsons)}] {json_path}", flush=True)
        make_pulse_grid(json_path)

    print(
        f"[done] rebuilt {len(flare_jsons)} FLARE and "
        f"{len(pulse_jsons)} PULSE H2-2 grids",
        flush=True,
    )


if __name__ == "__main__":
    main()
