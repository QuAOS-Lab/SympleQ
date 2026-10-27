"""Create the standalone building blocks used to assemble Plot 1.

The script produces the same two volume-versus-HQC figures and four static 3-D
level-set figures for H2-1 and H2-2. PULSE volumes use the native monotonic
Cost-aware boundary; FLARE volumes use the GP latent mean and variance. The
curved uncertainty bands belong to individual representative runs, whereas the
horizontal final-volume lines and bounds come from the corresponding
accumulated grids. Volume plots show exact checkpoint measurement-shot counts
for PULSE and FLARE on one shared top-axis spine.

Representative runs are selected reproducibly with ``REPRESENTATIVE_RNG_SEEDS``.
The selected run is not intended to be the best, worst, or median run. The
intermediate 3-D figures use the first saved checkpoint at or above the
configured HQC threshold. All figures are saved before one final ``plt.show()``
call.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtri


SHOW_VOLUME_LEGENDS = False

PLOT_SWITCHES = {
    "H2-1": {
        "representative_volumes": True,
        "all_seed_volumes": True,
        "pulse_final_3d": True,
        "flare_final_3d": True,
        "pulse_intermediate_3d": True,
        "flare_intermediate_3d": True,
    },
    "H2-2": {
        "representative_volumes": True,
        "all_seed_volumes": True,
        "pulse_final_3d": True,
        "flare_final_3d": True,
        "pulse_intermediate_3d": True,
        "flare_intermediate_3d": True,
    },
}

VOLUME_ROOTS = {
    "H2-1": (
        Path("Personal/Data/Cost_aware/H2-1"),
        Path("Personal/Data/FLE/H2_1/qband_4"),
        Path("Personal/Data/accumulated/uniform_10_5000_grids/H2-1"),
    ),
    "H2-2": (
        Path("Personal/Data/Cost_aware/H2-2"),
        Path("Personal/Data/FLE/H2_2/qband_4"),
        Path("Personal/Data/accumulated/uniform_10_5000_grids/H2-2"),
    ),
}
OUTPUT_DIR = Path("Personal/Data/Paper_plots/plot_1")

RNG_SEED = 42
REPRESENTATIVE_RNG_SEEDS = {"H2-1": 42, "H2-2": 43}
H2_2_FLARE_REPRESENTATIVE = "seed_2028"
SOBOL_HQC_LIMIT = 60.0
PULSE_INTERMEDIATE_HQC = 200.0
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


def measurement_shots(json_path: Path) -> int:
    """Count the cumulative measurement outcomes used by one saved grid."""
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    return sum(
        int(count)
        for record in payload.get("data", [])
        for _, count in record.get("results", [])
    )


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
    """Load each seed's ``(HQC, mean, lower, upper, shots)`` trajectory."""
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
            rows.append(
                (
                    hqc_for_json(json_path),
                    mean,
                    lower,
                    upper,
                    measurement_shots(json_path),
                )
            )

        if rows:
            results[seed_dir.name] = np.asarray(
                sorted(rows, key=lambda row: row[0]),
                dtype=float,
            )
    if not results:
        raise FileNotFoundError(f"No usable seed grids found under {root}")
    return results


def latest_accumulated_grid(
    accumulated_root: Path,
    device: str,
    *,
    native: bool,
) -> Path:
    """Return the newest method-only accumulated grid for one device."""
    if native:
        candidates = list(
            accumulated_root.glob(
                f"accumulated_actualgr_{device}_costaware_*"
                "_native_gp_grid_3d.npz"
            )
        )
    else:
        candidates = [
            path
            for path in accumulated_root.glob(
                f"accumulated_actualgr_{device}_fle_*_gp_grid_3d.npz"
            )
            if "_fle_costaware_" not in path.name
        ]
    if not candidates:
        method = "PULSE" if native else "FLARE"
        raise FileNotFoundError(
            f"No accumulated {device} {method} grid in {accumulated_root}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def add_final_reference(
    ax,
    volumes: tuple[float, float, float],
    *,
    color: str,
    method: str,
    shade_bounds: bool = False,
    bounds_alpha: float = 0.18,
) -> None:
    """Add the accumulated mean and its two-sigma volume limits."""
    mean, lower, upper = volumes
    ax.axhline(
        mean,
        color=color,
        linestyle="--",
        linewidth=2.3,
        label=f"Final volume ({method})",
    )
    if shade_bounds:
        ax.axhspan(
            lower,
            upper,
            color=color,
            alpha=bounds_alpha,
            linewidth=0,
            zorder=0,
        )
    else:
        for bound in (lower, upper):
            ax.axhline(
                bound,
                color=color,
                linestyle="-.",
                linewidth=1.5,
            )


def device_output_dir(device: str) -> Path:
    """Return the Plot 1 output folder for one device."""
    output_dir = OUTPUT_DIR / device
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_figure(fig, stem: str, *, device: str) -> None:
    """Save one building block in vector and high-resolution raster formats."""
    output_dir = device_output_dir(device)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
    )


def sparse_checkpoint_rows(rows: np.ndarray, maximum: int = 5) -> np.ndarray:
    """Select a small set of exact checkpoints, including both endpoints."""
    count = min(maximum, len(rows))
    indices = np.unique(
        np.rint(np.linspace(0, len(rows) - 1, count)).astype(int)
    )
    return rows[indices]


def add_measurement_shot_axes(
    ax,
    pulse_rows: np.ndarray,
    flare_rows: np.ndarray,
) -> None:
    """Add PULSE and FLARE shot labels to one shared top-axis spine."""
    axes = []
    for rows, color, label_pad in (
        (pulse_rows, PULSE_COLOR, 2),
        (flare_rows, FLARE_COLOR, 16),
    ):
        xmin, xmax = ax.get_xlim()
        visible_rows = rows[(rows[:, 0] >= xmin) & (rows[:, 0] <= xmax)]
        checkpoints = sparse_checkpoint_rows(
            visible_rows if len(visible_rows) else rows
        )
        top_ax = ax.twiny()
        top_ax.set_xlim(ax.get_xlim())
        top_ax.set_xticks(checkpoints[:, 0])
        top_ax.set_xticklabels(
            [f"{int(shots):,}" for shots in checkpoints[:, 4]]
        )
        top_ax.tick_params(
            axis="x",
            top=True,
            bottom=False,
            labeltop=True,
            labelbottom=False,
            colors=color,
            labelsize=8,
            pad=label_pad,
        )
        top_ax.spines["top"].set_position(("axes", 1.0))
        top_ax.spines["top"].set_color("black")
        top_ax.patch.set_visible(False)
        axes.append(top_ax)

    axes[-1].set_xlabel("Measurement-shots", labelpad=8)


def plot_representative_volumes(
    pulse_rows: np.ndarray,
    flare_rows: np.ndarray,
    pulse_final: tuple[float, float, float],
    flare_final: tuple[float, float, float],
    *,
    device: str,
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

    add_final_reference(
        ax,
        pulse_final,
        color=PULSE_COLOR,
        method="PULSE",
        shade_bounds=True,
        bounds_alpha=0.25,
    )
    add_final_reference(
        ax,
        flare_final,
        color=FLARE_COLOR,
        method="FLARE",
        shade_bounds=True,
    )
    ax.axvline(
        SOBOL_HQC_LIMIT,
        color=FLARE_COLOR,
        linestyle=":",
        linewidth=1.5,
        label="Sobol limit",
    )

    ax.set_xlim(25, 501)
    ax.set_ylim((20, 60) if device == "H2-2" else (14, 60))
    ax.set_yscale("log")
    ax.set_xlabel("HQC spent")
    ax.set_ylabel("QARB-Volume")
    ax.set_title(device, y=0.95)
    add_measurement_shot_axes(ax, pulse_rows, flare_rows)
    legend = ax.legend(frameon=False, fontsize=9, loc="best")
    legend.set_visible(SHOW_VOLUME_LEGENDS)
    fig.tight_layout()
    device_slug = device.lower().replace("-", "_")
    save_figure(
        fig,
        f"plot1_{device_slug}_pulse_flare_comparison",
        device=device,
    )
    return fig


def plot_all_seed_volumes(
    pulse_results: dict[str, np.ndarray],
    flare_results: dict[str, np.ndarray],
    pulse_measurement_seed: str,
    flare_measurement_seed: str,
    pulse_final: tuple[float, float, float],
    flare_final: tuple[float, float, float],
    *,
    device: str,
):
    """Plot all trajectories with accumulated final-volume uncertainty bands."""
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
    add_final_reference(
        ax,
        pulse_final,
        color=PULSE_COLOR,
        method="PULSE",
        shade_bounds=True,
        bounds_alpha=0.25,
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
    add_final_reference(
        ax,
        flare_final,
        color=FLARE_COLOR,
        method="FLARE",
        shade_bounds=True,
    )

    pulse_rows = pulse_results[pulse_measurement_seed]
    flare_rows = flare_results[flare_measurement_seed]

    ax.axvline(
        SOBOL_HQC_LIMIT,
        color=FLARE_COLOR,
        linestyle=":",
        linewidth=1.5,
    )

    ax.set_xlim(100, 503)
    ax.set_ylim((45, 60) if device == "H2-2" else (40, 60))
    ax.set_xlabel("HQC spent")
    ax.set_ylabel("QARB-Volume")
    add_measurement_shot_axes(ax, pulse_rows, flare_rows)
    legend = ax.legend(frameon=False, fontsize=8, ncol=2)
    legend.set_visible(SHOW_VOLUME_LEGENDS)
    fig.tight_layout()
    device_slug = device.lower().replace("-", "_")
    save_figure(
        fig,
        f"plot1_{device_slug}_pulse_flare_seed_comparison",
        device=device,
    )
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
    pulse_root: Path,
    flare_root: Path,
    pulse_seed: str,
    flare_seed: str,
) -> dict[str, tuple[Path, Path]]:
    """Resolve final and intermediate JSON/grid pairs for both methods."""
    pulse_dir = pulse_root / pulse_seed
    flare_dir = flare_root / flare_seed
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
    device: str,
    pulse_root: Path,
    flare_root: Path,
    pulse_seed: str,
    flare_seed: str,
) -> None:
    """Create final and intermediate static 3-D surfaces without showing yet."""
    switches = PLOT_SWITCHES[device]
    plot_names = (
        "pulse_final_3d",
        "flare_final_3d",
        "pulse_intermediate_3d",
        "flare_intermediate_3d",
    )
    if not any(switches[name] for name in plot_names):
        return

    output_dir = device_output_dir(device)
    inputs = selected_grid_inputs(
        pulse_root,
        flare_root,
        pulse_seed,
        flare_seed,
    )
    pulse_final_json, pulse_final_grid = inputs["pulse_final"]
    flare_final_json, flare_final_grid = inputs["flare_final"]
    pulse_mid_json, pulse_mid_grid = inputs["pulse_intermediate"]
    flare_mid_json, flare_mid_grid = inputs["flare_intermediate"]

    print(f"[{device}] PULSE seed: {pulse_seed}")
    print(
        f"  first >={PULSE_INTERMEDIATE_HQC:g} HQC: "
        f"{hqc_for_json(pulse_mid_json):.3f}"
    )
    print(f"  final:          {hqc_for_json(pulse_final_json):.3f}")
    print(f"[{device}] FLARE seed: {flare_seed}")
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

    if switches["pulse_final_3d"]:
        plot3d.plot_fle_isosurface_3d_matplotlib(
            json_path=pulse_final_json,
            grid_path=pulse_final_grid,
            png_path=output_dir / "pulse_final.pdf",
            xlim=(0.1, 0.97),
            ylim=(20, 56),
            legend_labels=pulse_labels,
            show=False,
        )
    if switches["flare_final_3d"]:
        plot3d.plot_fle_isosurface_3d_matplotlib(
            json_path=flare_final_json,
            grid_path=flare_final_grid,
            png_path=output_dir / "flare_final.pdf",
            xlim=(0.1, 1.0),
            ylim=(20, 56),
            legend_labels=flare_labels,
            show=False,
        )
    if switches["pulse_intermediate_3d"]:
        plot3d.plot_fle_isosurface_3d_matplotlib(
            json_path=pulse_mid_json,
            grid_path=pulse_mid_grid,
            png_path=(
                output_dir
                / (
                    f"{pulse_seed}_pulse_first_above_"
                    f"{PULSE_INTERMEDIATE_HQC:g}_hqc_3d.pdf"
                )
            ),
            xlim=(0.1, 0.97),
            ylim=(20, 56),
            legend_labels=pulse_labels,
            show=False,
        )
    if switches["flare_intermediate_3d"]:
        plot3d.plot_fle_isosurface_3d_matplotlib(
            json_path=flare_mid_json,
            grid_path=flare_mid_grid,
            png_path=(
                output_dir
                / f"{flare_seed}_flare_first_above_200_hqc_3d.pdf"
            ),
            xlim=(0.1, 0.97),
            ylim=(20, 56),
            zlim=(1.5, 3.45),
            legend_labels=flare_labels,
            show=False,
        )


def plot_volume_building_blocks(
    device: str,
    pulse_root: Path,
    flare_root: Path,
    accumulated_root: Path,
) -> tuple[str, str]:
    """Create both volume-curve figures for one H2 device."""
    pulse_results = load_seed_results(pulse_root, native=True)
    flare_results = load_seed_results(flare_root, native=False)

    pulse_grid = latest_accumulated_grid(
        accumulated_root,
        device,
        native=True,
    )
    flare_grid = latest_accumulated_grid(
        accumulated_root,
        device,
        native=False,
    )
    pulse_final = success_volumes(pulse_grid, native=True)
    flare_final = success_volumes(flare_grid)

    # Each device uses the same reproducible selection rule.
    rng = np.random.default_rng(REPRESENTATIVE_RNG_SEEDS[device])
    pulse_seed = str(rng.choice(sorted(pulse_results)))
    flare_seed = str(rng.choice(sorted(flare_results)))
    if device == "H2-2":
        flare_seed = H2_2_FLARE_REPRESENTATIVE
        if flare_seed not in flare_results:
            raise FileNotFoundError(f"Missing H2-2 FLARE {flare_seed}")
    print(f"[{device}] PULSE representative seed: {pulse_seed}")
    print(f"[{device}] FLARE representative seed: {flare_seed}")

    if PLOT_SWITCHES[device]["representative_volumes"]:
        plot_representative_volumes(
            pulse_results[pulse_seed],
            flare_results[flare_seed],
            pulse_final,
            flare_final,
            device=device,
        )

    # Select reproducible runs for the detail panel's measurement-shot axes.
    pulse_measurement_seed = str(
        np.random.default_rng(RNG_SEED).choice(sorted(pulse_results))
    )
    flare_cluster = [
        seed
        for seed in ("seed_2026", "seed_2028", "seed_2029")
        if seed in flare_results
    ]
    if not flare_cluster:
        flare_cluster = sorted(flare_results)
    flare_measurement_seed = str(
        np.random.default_rng(RNG_SEED).choice(flare_cluster)
    )
    if PLOT_SWITCHES[device]["all_seed_volumes"]:
        plot_all_seed_volumes(
            pulse_results,
            flare_results,
            pulse_measurement_seed,
            flare_measurement_seed,
            pulse_final,
            flare_final,
            device=device,
        )

    print(f"[{device}] PULSE measurement-axis seed: {pulse_measurement_seed}")
    print(f"[{device}] FLARE measurement-axis seed: {flare_measurement_seed}")
    print(f"[{device}] PULSE accumulated grid: {pulse_grid}")
    print(f"[{device}] FLARE accumulated grid: {flare_grid}")
    return pulse_seed, flare_seed


def main() -> None:
    """Generate and save every Plot 1 building block, then display them together."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    representatives = {
        device: plot_volume_building_blocks(
            device,
            pulse_root,
            flare_root,
            accumulated_root,
        )
        for device, (
            pulse_root,
            flare_root,
            accumulated_root,
        ) in VOLUME_ROOTS.items()
    }

    for device, (pulse_root, flare_root, _) in VOLUME_ROOTS.items():
        pulse_seed, flare_seed = representatives[device]
        plot_3d_building_blocks(
            device,
            pulse_root,
            flare_root,
            pulse_seed,
            flare_seed,
        )

    print(f"Saved H2-1 outputs in {device_output_dir('H2-1')}")
    print(f"Saved H2-2 outputs in {device_output_dir('H2-2')}")

    if plt.get_fignums():
        plt.show()


if __name__ == "__main__":
    main()
