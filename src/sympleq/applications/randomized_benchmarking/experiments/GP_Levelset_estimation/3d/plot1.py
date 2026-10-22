"""Create the standalone H2-1 building blocks used to assemble Plot 1.

The script produces two volume-versus-HQC figures and four static 3-D level-set
figures. PULSE volumes use the native monotonic Cost-aware boundary; FLARE
volumes use the GP latent mean and variance. The curved uncertainty bands belong
to individual representative runs, whereas the horizontal final-volume lines
and bounds come from the corresponding accumulated grids.

Representative runs are selected reproducibly with ``RNG_SEED``. The selected
run is not intended to be the best, worst, or median run. The intermediate 3-D
figures use the first saved checkpoint at or above the configured HQC threshold.
All figures are saved before one final ``plt.show()`` call.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtri


PULSE_ROOT = Path("Personal/Data/Cost_aware/H2-1")
FLARE_ROOT = Path("Personal/Data/FLE/H2_1/qband_4")
ACCUMULATED_ROOT = Path(
    "Personal/Data/accumulated/uniform_10_5000_grids/H2-1"
)
OUTPUT_DIR = Path("Personal/Data/Paper_plots/plot_1")

RNG_SEED = 42
SOBOL_HQC_LIMIT = 60.0
PULSE_INTERMEDIATE_HQC = 100.0
FLARE_INTERMEDIATE_HQC = 200.0

PULSE_COLOR = "#d85454"
PULSE_BAND_COLOR = "#ef6a6a"
FLARE_COLOR = "navy"
FLARE_BAND_COLOR = "#6baed6"

SRC_ROOT = Path("src").resolve()
PLOTTER_DIR = (
    SRC_ROOT
    / "sympleq/applications/randomized_benchmarking/experiments/"
    "GP_Levelset_estimation/3d"
)
for module_path in (SRC_ROOT, PLOTTER_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import plot3dFLE as plot3d  # noqa: E402


def axis_spacing(values: np.ndarray, *, log: bool = False) -> float:
    """Return the mean spacing of one unique grid axis."""
    axis = np.unique(np.asarray(values, dtype=float))
    if log:
        axis = np.log10(axis)
    if len(axis) < 2:
        raise ValueError("A volume axis must contain at least two values")
    return float(np.mean(np.diff(np.sort(axis))))


def success_volumes(
    grid_path: Path,
    *,
    native: bool = False,
) -> tuple[float, float, float]:
    """Return mean, lower-2-sigma, and upper-2-sigma success volumes.

    Native PULSE grids store their monotonic gate-count boundaries explicitly.
    FLARE grids instead define success by thresholding latent mean and its
    uncertainty at the probit-transformed target probability.
    """
    with np.load(grid_path) as grid:
        gates_grid = np.asarray(grid["gates_grid"], dtype=float)

        if native:
            boundaries = (
                np.asarray(grid["native_boundary_mean"], dtype=float),
                np.asarray(grid["native_boundary_lower_2sigma"], dtype=float),
                np.asarray(grid["native_boundary_upper_2sigma"], dtype=float),
            )
            success_masks = [
                np.isfinite(boundary)[..., np.newaxis]
                & (gates_grid <= boundary[..., np.newaxis])
                for boundary in boundaries
            ]
        else:
            latent_mean = np.asarray(grid["latent_mean"], dtype=float)
            latent_std = np.sqrt(
                np.maximum(
                    np.asarray(grid["latent_variance"], dtype=float),
                    0.0,
                )
            )
            latent_target = ndtri(float(np.asarray(grid["target"]).item()))
            finite = np.isfinite(latent_mean) & np.isfinite(latent_std)
            success_masks = [
                finite & (field >= latent_target)
                for field in (
                    latent_mean,
                    latent_mean - 2.0 * latent_std,
                    latent_mean + 2.0 * latent_std,
                )
            ]

        voxel_volume = (
            axis_spacing(gates_grid, log=True)
            * axis_spacing(grid["ratio_grid"])
            * axis_spacing(grid["qubits_grid"])
        )

    return tuple(
        float(np.sum(mask) * voxel_volume)
        for mask in success_masks
    )


def hqc_for_json(json_path: Path) -> float:
    """Read cumulative HQC, including legacy final PULSE JSONs.

    Intermediate rewrites contain HQC directly. Older final PULSE rewrites do
    not, so their adjacent restartable checkpoint is used as the fallback.
    """
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    experiment_hqc = payload.get("experiment", {}).get("spent_hqc")
    if experiment_hqc is not None:
        return float(experiment_hqc)

    rewrite = payload.get("actual_gates_rewrite", {})
    rewrite_hqc = rewrite.get("spent_hqc")
    if rewrite_hqc is not None:
        return float(rewrite_hqc)

    saved_checkpoint = rewrite.get("checkpoint_json")
    checkpoint_path = Path(saved_checkpoint) if saved_checkpoint else Path()
    if not checkpoint_path.is_file():
        checkpoint_path = (
            json_path.parent.parent / "CostAware_restartable_checkpoint.json"
        )
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    return float(checkpoint["spent_hqc"])


def grid_path_for_json(json_path: Path, *, native: bool) -> Path:
    """Resolve a per-seed uniform grid from its rewritten actual-GR JSON."""
    seed_dir = json_path.parent.parent
    if native:
        return (
            seed_dir
            / "ActualGR_native_grid"
            / f"{json_path.stem}_native_gp_grid_3d.npz"
        )
    return seed_dir / "ActualGR_grid" / f"{json_path.stem}_gp_grid_3d.npz"


def load_seed_results(root: Path, *, native: bool) -> dict[str, np.ndarray]:
    """Load each seed's ``(HQC, mean, lower, upper)`` volume trajectory."""
    results: dict[str, np.ndarray] = {}
    for seed_dir in sorted(root.glob("seed_*")):
        rows = []
        for json_path in sorted(
            seed_dir.joinpath("ActualGR").glob("*_actual_gates.json")
        ):
            grid_path = grid_path_for_json(json_path, native=native)
            if not grid_path.exists():
                continue
            mean, lower, upper = success_volumes(grid_path, native=native)
            rows.append((hqc_for_json(json_path), mean, lower, upper))

        if rows:
            results[seed_dir.name] = np.asarray(
                sorted(rows, key=lambda row: row[0]),
                dtype=float,
            )
    if not results:
        raise FileNotFoundError(f"No usable seed grids found under {root}")
    return results


def latest_accumulated_grid(*, native: bool) -> Path:
    """Return the newest method-only accumulated H2-1 grid."""
    if native:
        candidates = list(
            ACCUMULATED_ROOT.glob(
                "accumulated_actualgr_H2-1_costaware_*"
                "_native_gp_grid_3d.npz"
            )
        )
    else:
        candidates = [
            path
            for path in ACCUMULATED_ROOT.glob(
                "accumulated_actualgr_H2-1_fle_*_gp_grid_3d.npz"
            )
            if "_fle_costaware_" not in path.name
        ]
    if not candidates:
        method = "PULSE" if native else "FLARE"
        raise FileNotFoundError(f"No accumulated {method} grid in {ACCUMULATED_ROOT}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def add_final_reference(
    ax,
    volumes: tuple[float, float, float],
    *,
    color: str,
    method: str,
) -> None:
    """Add the accumulated mean and unlabeled two-sigma volume limits."""
    mean, lower, upper = volumes
    ax.axhline(
        mean,
        color=color,
        linestyle="--",
        linewidth=2.3,
        label=f"Final volume ({method})",
    )
    for bound in (lower, upper):
        ax.axhline(
            bound,
            color=color,
            linestyle="-.",
            linewidth=1.5,
        )


def save_figure(fig, stem: str) -> None:
    """Save one building block in vector and high-resolution raster formats."""
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(
        OUTPUT_DIR / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
    )


def plot_representative_volumes(
    pulse_rows: np.ndarray,
    flare_rows: np.ndarray,
    pulse_final: tuple[float, float, float],
    flare_final: tuple[float, float, float],
):
    """Plot one reproducibly selected trajectory for each algorithm."""
    fig, ax = plt.subplots(figsize=(8.2, 5.8))

    ax.plot(
        pulse_rows[:, 0],
        pulse_rows[:, 1],
        color=PULSE_COLOR,
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        label="PULSE Representative",
    )
    ax.fill_between(
        pulse_rows[:, 0],
        pulse_rows[:, 2],
        pulse_rows[:, 3],
        color=PULSE_BAND_COLOR,
        alpha=0.18,
        linewidth=0,
    )

    ax.plot(
        flare_rows[:, 0],
        flare_rows[:, 1],
        color=FLARE_COLOR,
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        label="FLARE Representative",
    )
    ax.fill_between(
        flare_rows[:, 0],
        flare_rows[:, 2],
        flare_rows[:, 3],
        color=FLARE_BAND_COLOR,
        alpha=0.18,
        linewidth=0,
    )

    add_final_reference(ax, pulse_final, color=PULSE_COLOR, method="PULSE")
    add_final_reference(ax, flare_final, color=FLARE_COLOR, method="FLARE")
    ax.axvline(
        SOBOL_HQC_LIMIT,
        color=FLARE_COLOR,
        linestyle=":",
        linewidth=1.5,
        label="Sobol limit",
    )

    ax.set_xlim(10, 510)
    ax.set_ylim(0, 87)
    ax.set_xlabel("HQC spent")
    ax.set_ylabel("QARB-Volume")
    ax.set_title("H2-1")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    save_figure(fig, "plot1_h2_1_pulse_flare_comparison")
    return fig


def plot_all_seed_volumes(
    pulse_results: dict[str, np.ndarray],
    flare_results: dict[str, np.ndarray],
    pulse_uncertainty_seed: str,
    flare_uncertainty_seed: str,
    pulse_final: tuple[float, float, float],
    flare_final: tuple[float, float, float],
):
    """Plot all trajectories and one representative uncertainty band per method."""
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    pulse_colors = plt.cm.Reds(np.linspace(0.85, 0.35, len(pulse_results)))
    flare_colors = plt.cm.Blues(np.linspace(0.85, 0.35, len(flare_results)))

    for color, (seed, rows) in zip(pulse_colors, sorted(pulse_results.items())):
        ax.plot(
            rows[:, 0],
            rows[:, 1],
            color=color,
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            label=f"PULSE {seed.replace('_', ' ')}",
        )
    for color, (seed, rows) in zip(flare_colors, sorted(flare_results.items())):
        ax.plot(
            rows[:, 0],
            rows[:, 1],
            color=color,
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            label=f"FLARE {seed.replace('_', ' ')}",
        )

    pulse_rows = pulse_results[pulse_uncertainty_seed]
    flare_rows = flare_results[flare_uncertainty_seed]
    ax.fill_between(
        pulse_rows[:, 0],
        pulse_rows[:, 2],
        pulse_rows[:, 3],
        color="#cb4d4d",
        alpha=0.18,
        linewidth=0,
    )
    ax.fill_between(
        flare_rows[:, 0],
        flare_rows[:, 2],
        flare_rows[:, 3],
        color=FLARE_BAND_COLOR,
        alpha=0.18,
        linewidth=0,
    )

    add_final_reference(ax, pulse_final, color=PULSE_COLOR, method="PULSE")
    add_final_reference(ax, flare_final, color=FLARE_COLOR, method="FLARE")
    ax.axvline(
        SOBOL_HQC_LIMIT,
        color=FLARE_COLOR,
        linestyle=":",
        linewidth=1.5,
    )

    ax.set_xlim(60, 503)
    ax.set_ylim(40, 65)
    ax.set_xlabel("HQC spent")
    ax.set_ylabel("QARB-Volume")
    ax.set_title("H2-1")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    save_figure(fig, "plot1_h2_1_pulse_flare_seed_comparison")
    return fig


def first_checkpoint_at_or_above(
    json_paths: list[Path],
    threshold: float,
) -> Path:
    """Return the earliest saved checkpoint whose cumulative HQC crosses a limit."""
    eligible = [path for path in json_paths if hqc_for_json(path) >= threshold]
    if not eligible:
        raise FileNotFoundError(f"No checkpoint reaches {threshold:g} HQC")
    return min(eligible, key=hqc_for_json)


def selected_grid_inputs(
    pulse_seed: str,
    flare_seed: str,
) -> dict[str, tuple[Path, Path]]:
    """Resolve final and intermediate JSON/grid pairs for both methods."""
    pulse_dir = PULSE_ROOT / pulse_seed
    flare_dir = FLARE_ROOT / flare_seed
    pulse_jsons = sorted(
        pulse_dir.joinpath("ActualGR").glob("*_actual_gates.json")
    )
    flare_jsons = sorted(
        flare_dir.joinpath("ActualGR").glob("*_actual_gates.json")
    )
    if not pulse_jsons or not flare_jsons:
        raise FileNotFoundError("Missing actual-GR JSONs for a representative seed")

    selected_jsons = {
        "pulse_final": max(pulse_jsons, key=hqc_for_json),
        "flare_final": max(flare_jsons, key=hqc_for_json),
        "pulse_intermediate": first_checkpoint_at_or_above(
            pulse_jsons,
            PULSE_INTERMEDIATE_HQC,
        ),
        "flare_intermediate": first_checkpoint_at_or_above(
            flare_jsons,
            FLARE_INTERMEDIATE_HQC,
        ),
    }
    inputs = {
        name: (
            json_path,
            grid_path_for_json(json_path, native=name.startswith("pulse")),
        )
        for name, json_path in selected_jsons.items()
    }
    for _, grid_path in inputs.values():
        if not grid_path.exists():
            raise FileNotFoundError(grid_path)
    return inputs


def plot_3d_building_blocks(
    pulse_seed: str,
    flare_seed: str,
) -> None:
    """Create final and intermediate static 3-D surfaces without showing yet."""
    inputs = selected_grid_inputs(pulse_seed, flare_seed)
    pulse_final_json, pulse_final_grid = inputs["pulse_final"]
    flare_final_json, flare_final_grid = inputs["flare_final"]
    pulse_mid_json, pulse_mid_grid = inputs["pulse_intermediate"]
    flare_mid_json, flare_mid_grid = inputs["flare_intermediate"]

    print(f"PULSE seed: {pulse_seed}")
    print(
        f"  first >={PULSE_INTERMEDIATE_HQC:g} HQC: "
        f"{hqc_for_json(pulse_mid_json):.3f}"
    )
    print(f"  final:          {hqc_for_json(pulse_final_json):.3f}")
    print(f"FLARE seed: {flare_seed}")
    print(
        f"  first >={FLARE_INTERMEDIATE_HQC:g} HQC: "
        f"{hqc_for_json(flare_mid_json):.3f}"
    )
    print(f"  final:          {hqc_for_json(flare_final_json):.3f}")

    plot3d.SHOW_ONE_SIGMA_SURFACES = False
    plot3d.SHOW_TWO_SIGMA_SURFACES = True
    plot3d.SHOW_MEASURED_POINTS = True

    pulse_labels = {
        "Success-side volume": "Success volume",
        "PULSE posterior mean boundary": "Levelset",
        "PULSE - 2 sigma": None,
        "PULSE + 2 sigma": None,
    }
    flare_labels = {
        "Success-side volume": "Success volume",
        "FLARE GP P(success) = 0.5": "Levelset",
        "FLARE - 2 sigma": None,
        "FLARE + 2 sigma": None,
    }

    plot3d.plot_fle_isosurface_3d_matplotlib(
        json_path=pulse_final_json,
        grid_path=pulse_final_grid,
        png_path=OUTPUT_DIR / "pulse_final.png",
        xlim=(0.1, 0.97),
        ylim=(20, 56),
        legend_labels=pulse_labels,
        show=False,
    )
    plot3d.plot_fle_isosurface_3d_matplotlib(
        json_path=flare_final_json,
        grid_path=flare_final_grid,
        png_path=OUTPUT_DIR / "flare_final.png",
        xlim=(0.1, 1.0),
        ylim=(20, 56),
        legend_labels=flare_labels,
        show=False,
    )
    plot3d.plot_fle_isosurface_3d_matplotlib(
        json_path=pulse_mid_json,
        grid_path=pulse_mid_grid,
        png_path=(
            OUTPUT_DIR
            / f"{pulse_seed}_pulse_first_above_100_hqc_3d.png"
        ),
        xlim=(0.1, 0.97),
        ylim=(20, 56),
        legend_labels=pulse_labels,
        show=False,
    )
    plot3d.plot_fle_isosurface_3d_matplotlib(
        json_path=flare_mid_json,
        grid_path=flare_mid_grid,
        png_path=(
            OUTPUT_DIR
            / f"{flare_seed}_flare_first_above_200_hqc_3d.png"
        ),
        xlim=(0.1, 0.97),
        ylim=(20, 56),
        zlim=(1.5, 3.45),
        legend_labels=flare_labels,
        show=False,
    )


def main() -> None:
    """Generate and save every Plot 1 building block, then display them together."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pulse_results = load_seed_results(PULSE_ROOT, native=True)
    flare_results = load_seed_results(FLARE_ROOT, native=False)

    pulse_grid = latest_accumulated_grid(native=True)
    flare_grid = latest_accumulated_grid(native=False)
    pulse_final = success_volumes(pulse_grid, native=True)
    flare_final = success_volumes(flare_grid)

    # Sequential draws from one fixed generator preserve the existing choices.
    rng = np.random.default_rng(RNG_SEED)
    pulse_seed = str(rng.choice(sorted(pulse_results)))
    flare_seed = str(rng.choice(sorted(flare_results)))
    print(f"PULSE representative seed: {pulse_seed}")
    print(f"FLARE representative seed: {flare_seed}")

    plot_representative_volumes(
        pulse_results[pulse_seed],
        flare_results[flare_seed],
        pulse_final,
        flare_final,
    )

    # Preserve the separate uncertainty-band choices used by the detail panel.
    pulse_uncertainty_seed = str(
        np.random.default_rng(RNG_SEED).choice(sorted(pulse_results))
    )
    flare_cluster = [
        seed
        for seed in ("seed_2026", "seed_2028", "seed_2029")
        if seed in flare_results
    ]
    if not flare_cluster:
        flare_cluster = sorted(flare_results)
    flare_uncertainty_seed = str(
        np.random.default_rng(RNG_SEED).choice(flare_cluster)
    )
    plot_all_seed_volumes(
        pulse_results,
        flare_results,
        pulse_uncertainty_seed,
        flare_uncertainty_seed,
        pulse_final,
        flare_final,
    )

    plot_3d_building_blocks(pulse_seed, flare_seed)

    print(f"PULSE uncertainty seed: {pulse_uncertainty_seed}")
    print(f"FLARE uncertainty seed: {flare_uncertainty_seed}")
    print(f"PULSE accumulated grid: {pulse_grid}")
    print(f"FLARE accumulated grid: {flare_grid}")
    print(f"Saved outputs in {OUTPUT_DIR}")

    plt.show()


if __name__ == "__main__":
    main()
