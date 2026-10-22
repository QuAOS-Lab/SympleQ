import json
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


def axis_spacing(values, log=False):
    axis = np.unique(np.asarray(values, dtype=float))
    if log:
        axis = np.log10(axis)
    return float(np.mean(np.diff(np.sort(axis))))


def success_volumes(grid_path, native=False):
    with np.load(grid_path) as grid:
        gates_grid = np.asarray(grid["gates_grid"], dtype=float)

        if native:
            boundaries = (
                np.asarray(grid["native_boundary_mean"], dtype=float),
                np.asarray(
                    grid["native_boundary_lower_2sigma"],
                    dtype=float,
                ),
                np.asarray(
                    grid["native_boundary_upper_2sigma"],
                    dtype=float,
                ),
            )
            success_masks = [
                np.isfinite(boundary)[..., np.newaxis]
                & (gates_grid <= boundary[..., np.newaxis])
                for boundary in boundaries
            ]
        else:
            mean = np.asarray(grid["latent_mean"], dtype=float)
            std = np.sqrt(
                np.maximum(
                    np.asarray(grid["latent_variance"], dtype=float),
                    0.0,
                )
            )
            level = ndtri(float(np.asarray(grid["target"]).item()))
            finite = np.isfinite(mean) & np.isfinite(std)
            success_masks = [
                finite & (field >= level)
                for field in (
                    mean,
                    mean - 2 * std,
                    mean + 2 * std,
                )
            ]

        voxel_volume = (
            axis_spacing(gates_grid, log=True)
            * axis_spacing(grid["ratio_grid"])
            * axis_spacing(grid["qubits_grid"])
        )

    return np.asarray([
        np.sum(mask) * voxel_volume
        for mask in success_masks
    ])


pulse_results = {}

for seed_dir in sorted(PULSE_ROOT.glob("seed_*")):
    checkpoint = json.loads(
        (seed_dir / "CostAware_restartable_checkpoint.json")
        .read_text(encoding="utf-8")
    )
    final_hqc = float(checkpoint["spent_hqc"])
    rows = []

    for json_path in seed_dir.joinpath("ActualGR").glob(
        "*_actual_gates.json"
    ):
        grid_path = (
            seed_dir
            / "ActualGR_native_grid"
            / f"{json_path.stem}_native_gp_grid_3d.npz"
        )
        if not grid_path.exists():
            continue

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        hqc = float(
            payload["experiment"]["spent_hqc"]
            if "experiment" in payload
            else final_hqc
        )
        mean, lower, upper = success_volumes(
            grid_path,
            native=True,
        )
        rows.append((hqc, mean, lower, upper))

    pulse_results[seed_dir.name] = np.asarray(
        sorted(rows, key=lambda row: row[0]),
        dtype=float,
    )


flare_results = {}

for seed_dir in sorted(FLARE_ROOT.glob("seed_*")):
    rows = []

    for json_path in seed_dir.joinpath("ActualGR").glob(
        "*_actual_gates.json"
    ):
        grid_path = (
            seed_dir
            / "ActualGR_grid"
            / f"{json_path.stem}_gp_grid_3d.npz"
        )
        if not grid_path.exists():
            continue

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        hqc = float(payload["experiment"]["spent_hqc"])
        mean, lower, upper = success_volumes(grid_path)
        rows.append((hqc, mean, lower, upper))

    flare_results[seed_dir.name] = np.asarray(
        sorted(rows, key=lambda row: row[0]),
        dtype=float,
    )


pulse_grid = max(
    ACCUMULATED_ROOT.glob(
        "accumulated_actualgr_H2-1_costaware_*"
        "_native_gp_grid_3d.npz"
    ),
    key=lambda path: path.stat().st_mtime,
)
pulse_final, pulse_lower, pulse_upper = success_volumes(
    pulse_grid,
    native=True,
)

flare_grids = [
    path
    for path in ACCUMULATED_ROOT.glob(
        "accumulated_actualgr_H2-1_fle_*_gp_grid_3d.npz"
    )
    if "_fle_costaware_" not in path.name
]
flare_grid = max(
    flare_grids,
    key=lambda path: path.stat().st_mtime,
)
flare_final, flare_lower, flare_upper = success_volumes(
    flare_grid
)


pulse_seeds = sorted(pulse_results)
flare_seeds = sorted(flare_results)

rng = np.random.default_rng(42)
pulse_seed = rng.choice(pulse_seeds)
flare_seed = rng.choice(flare_seeds)

pulse_rows = pulse_results[pulse_seed]
flare_rows = flare_results[flare_seed]

print(f"PULSE representative seed: {pulse_seed}")

print(f"FLARE representative seed: {flare_seed}")


fig, ax = plt.subplots(figsize=(8.2, 5.8))

ax.plot(
    pulse_rows[:, 0],
    pulse_rows[:, 1],
    color="#d85454",
    marker="o",
    markersize=3.5,
    linewidth=1.8,
    label=f"PULSE Representative",
)
ax.fill_between(
    pulse_rows[:, 0],
    pulse_rows[:, 2],
    pulse_rows[:, 3],
    color="#ef6a6a",
    alpha=0.18,
    linewidth=0,
)

ax.plot(
    flare_rows[:, 0],
    flare_rows[:, 1],
    color="navy",
    marker="o",
    markersize=3.5,
    linewidth=1.8,
    label=f"FLARE Representative",
)
ax.fill_between(
    flare_rows[:, 0],
    flare_rows[:, 2],
    flare_rows[:, 3],
    color="#6baed6",
    alpha=0.18,
    linewidth=0,
)

ax.axhline(
    pulse_final,
    color="#d85454",
    linestyle="--",
    linewidth=2.3,
    label="Final volume (PULSE)",
)
ax.axhline(
    pulse_lower,
    color="#d85454",
    linestyle="-.",
    linewidth=1.5,
)
ax.axhline(
    pulse_upper,
    color="#d85454",
    linestyle="-.",
    linewidth=1.5,
)

ax.axhline(
    flare_final,
    color="navy",
    linestyle="--",
    linewidth=2.3,
    label="Final volume (FLARE)",
)
ax.axhline(
    flare_lower,
    color="navy",
    linestyle="-.",
    linewidth=1.5,
)
ax.axhline(
    flare_upper,
    color="navy",
    linestyle="-.",
    linewidth=1.5,
)

ax.axvline(
    60.0,
    color="navy",
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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(
    OUTPUT_DIR / "plot1_h2_1_pulse_flare_comparison.pdf",
    bbox_inches="tight",
)
fig.savefig(
    OUTPUT_DIR / "plot1_h2_1_pulse_flare_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

print(f"PULSE seed: {pulse_seed}")
print(f"FLARE seed: {flare_seed}")
print(f"PULSE accumulated grid: {pulse_grid}")
print(f"FLARE accumulated grid: {flare_grid}")

plt.show()


#### INSET (H2-1 PANEL-1)
import json
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


def axis_spacing(values, log=False):
    axis = np.unique(np.asarray(values, dtype=float))
    if log:
        axis = np.log10(axis)
    return float(np.mean(np.diff(np.sort(axis))))


def success_volumes(grid_path, native=False):
    with np.load(grid_path) as grid:
        gates_grid = np.asarray(grid["gates_grid"], dtype=float)

        if native:
            boundaries = (
                np.asarray(grid["native_boundary_mean"], dtype=float),
                np.asarray(
                    grid["native_boundary_lower_2sigma"],
                    dtype=float,
                ),
                np.asarray(
                    grid["native_boundary_upper_2sigma"],
                    dtype=float,
                ),
            )
            success_masks = [
                np.isfinite(boundary)[..., np.newaxis]
                & (gates_grid <= boundary[..., np.newaxis])
                for boundary in boundaries
            ]
        else:
            mean = np.asarray(grid["latent_mean"], dtype=float)
            std = np.sqrt(
                np.maximum(
                    np.asarray(grid["latent_variance"], dtype=float),
                    0.0,
                )
            )
            level = ndtri(float(np.asarray(grid["target"]).item()))
            finite = np.isfinite(mean) & np.isfinite(std)
            success_masks = [
                finite & (field >= level)
                for field in (
                    mean,
                    mean - 2 * std,
                    mean + 2 * std,
                )
            ]

        voxel_volume = (
            axis_spacing(gates_grid, log=True)
            * axis_spacing(grid["ratio_grid"])
            * axis_spacing(grid["qubits_grid"])
        )

    return np.asarray([
        np.sum(mask) * voxel_volume
        for mask in success_masks
    ])


pulse_results = {}

for seed_dir in sorted(PULSE_ROOT.glob("seed_*")):
    checkpoint = json.loads(
        (seed_dir / "CostAware_restartable_checkpoint.json")
        .read_text(encoding="utf-8")
    )
    final_hqc = float(checkpoint["spent_hqc"])
    rows = []

    for json_path in seed_dir.joinpath("ActualGR").glob(
        "*_actual_gates.json"
    ):
        grid_path = (
            seed_dir
            / "ActualGR_native_grid"
            / f"{json_path.stem}_native_gp_grid_3d.npz"
        )
        if not grid_path.exists():
            continue

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        hqc = float(
            payload["experiment"]["spent_hqc"]
            if "experiment" in payload
            else final_hqc
        )
        mean, lower, upper = success_volumes(
            grid_path,
            native=True,
        )
        rows.append((hqc, mean, lower, upper))

    pulse_results[seed_dir.name] = np.asarray(
        sorted(rows, key=lambda row: row[0]),
        dtype=float,
    )


flare_results = {}

for seed_dir in sorted(FLARE_ROOT.glob("seed_*")):
    rows = []

    for json_path in seed_dir.joinpath("ActualGR").glob(
        "*_actual_gates.json"
    ):
        grid_path = (
            seed_dir
            / "ActualGR_grid"
            / f"{json_path.stem}_gp_grid_3d.npz"
        )
        if not grid_path.exists():
            continue

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        hqc = float(payload["experiment"]["spent_hqc"])
        mean, lower, upper = success_volumes(grid_path)
        rows.append((hqc, mean, lower, upper))

    flare_results[seed_dir.name] = np.asarray(
        sorted(rows, key=lambda row: row[0]),
        dtype=float,
    )


pulse_grid = max(
    ACCUMULATED_ROOT.glob(
        "accumulated_actualgr_H2-1_costaware_*"
        "_native_gp_grid_3d.npz"
    ),
    key=lambda path: path.stat().st_mtime,
)
pulse_final, pulse_lower, pulse_upper = success_volumes(
    pulse_grid,
    native=True,
)

flare_grids = [
    path
    for path in ACCUMULATED_ROOT.glob(
        "accumulated_actualgr_H2-1_fle_*_gp_grid_3d.npz"
    )
    if "_fle_costaware_" not in path.name
]
flare_grid = max(
    flare_grids,
    key=lambda path: path.stat().st_mtime,
)
flare_final, flare_lower, flare_upper = success_volumes(
    flare_grid,
)

pulse_seeds = sorted(pulse_results)
flare_seeds = sorted(flare_results)

pulse_representative_seed = np.random.default_rng(42).choice(
    pulse_seeds
)
flare_cluster = [
    seed
    for seed in ("seed_2026", "seed_2028", "seed_2029")
    if seed in flare_results
]
flare_representative_seed = np.random.default_rng(42).choice(
    flare_cluster
)

pulse_representative = pulse_results[pulse_representative_seed]
flare_representative = flare_results[flare_representative_seed]

fig, ax = plt.subplots(figsize=(8.2, 5.8))

pulse_colors = plt.cm.Reds(
    np.linspace(0.85, 0.35, len(pulse_seeds))
)
flare_colors = plt.cm.Blues(
    np.linspace(0.85, 0.35, len(flare_seeds))
)

for color, seed in zip(pulse_colors, pulse_seeds):
    rows = pulse_results[seed]
    ax.plot(
        rows[:, 0],
        rows[:, 1],
        color=color,
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        label=f"PULSE {seed.replace('_', ' ')}",
    )

ax.axhline(
    pulse_final,
    color="#d85454",
    linestyle="--",
    linewidth=2.3,
    label="Final volume (PULSE)",
)
ax.axhline(
    pulse_lower,
    color="#d85454",
    linestyle="-.",
    linewidth=1.5,
)
ax.axhline(
    pulse_upper,
    color="#d85454",
    linestyle="-.",
    linewidth=1.5,
)


for color, seed in zip(flare_colors, flare_seeds):
    rows = flare_results[seed]
    ax.plot(
        rows[:, 0],
        rows[:, 1],
        color=color,
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        label=f"FLARE {seed.replace('_', ' ')}",
    )

ax.fill_between(
    pulse_representative[:, 0],
    pulse_representative[:, 2],
    pulse_representative[:, 3],
    color="#cb4d4d",
    alpha=0.18,
    linewidth=0,
)

ax.fill_between(
    flare_representative[:, 0],
    flare_representative[:, 2],
    flare_representative[:, 3],
    color="#6baed6",
    alpha=0.18,
    linewidth=0,
)


ax.axhline(
    flare_final,
    color="navy",
    linestyle="--",
    linewidth=2.3,
    label="Final volume (FLARE)",
)
ax.axhline(
    flare_lower,
    color="navy",
    linestyle="-.",
    linewidth=1.5,
)
ax.axhline(
    flare_upper,
    color="navy",
    linestyle="-.",
    linewidth=1.5,
)


ax.axvline(
    60.0,
    color="navy",
    linestyle=":",
    linewidth=1.5,
)

ax.set_xlim(60, 503)
ax.set_ylim(40, 65)
ax.set_xlabel("HQC spent")
ax.set_ylabel("QARB-Volume")
ax.set_title("H2-1")
ax.legend(
    frameon=False,
    fontsize=8,
    ncol=2,
)

fig.tight_layout()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(
    OUTPUT_DIR / "plot1_h2_1_pulse_flare_seed_comparison.pdf",
    bbox_inches="tight",
)
fig.savefig(
    OUTPUT_DIR / "plot1_h2_1_pulse_flare_seed_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

print(f"PULSE uncertainty: {pulse_representative_seed}")
print(f"FLARE uncertainty: {flare_representative_seed}")
print(f"PULSE accumulated grid: {pulse_grid}")
print(f"FLARE accumulated grid: {flare_grid}")

plt.show()


###### FINAL GRIDS #######
import importlib
import json
import sys
from pathlib import Path

SRC_ROOT = Path("src").resolve()
PLOTTER_DIR = (
    SRC_ROOT
    / "sympleq/applications/randomized_benchmarking/"
      "experiments/GP_Levelset_estimation/3d"
)

for path in (SRC_ROOT, PLOTTER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import plot3dFLE as plot3d

plot3d = importlib.reload(plot3d)
plot3d.SHOW_ONE_SIGMA_SURFACES = False
plot3d.SHOW_TWO_SIGMA_SURFACES = True
plot3d.SHOW_MEASURED_POINTS = True

PULSE_ROOT = Path("Personal/Data/Cost_aware/H2-1")
FLARE_ROOT = Path("Personal/Data/FLE/H2_1/qband_4")
OUTPUT_DIR = Path("Personal/Data/Paper_plots/plot_1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if "pulse_seed" not in globals() or "flare_seed" not in globals():
    raise NameError(
        "Run the seed-selection cell first so pulse_seed and flare_seed exist."
    )

pulse_seed = str(pulse_seed)
flare_seed = str(flare_seed)

if not pulse_seed.startswith("seed_"):
    pulse_seed = f"seed_{pulse_seed}"
if not flare_seed.startswith("seed_"):
    flare_seed = f"seed_{flare_seed}"


def hqc_for_json(json_path):
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    experiment_hqc = payload.get("experiment", {}).get("spent_hqc")
    if experiment_hqc is not None:
        return float(experiment_hqc)

    rewrite = payload.get("actual_gates_rewrite", {})
    rewrite_hqc = rewrite.get("spent_hqc")
    if rewrite_hqc is not None:
        return float(rewrite_hqc)

    checkpoint_path = Path(
        rewrite.get("checkpoint_json", "")
    )

    if not checkpoint_path.exists():
        checkpoint_path = (
            json_path.parent.parent
            / "CostAware_restartable_checkpoint.json"
        )

    checkpoint = json.loads(
        checkpoint_path.read_text(encoding="utf-8")
    )
    return float(checkpoint["spent_hqc"])


pulse_dir = PULSE_ROOT / pulse_seed
flare_dir = FLARE_ROOT / flare_seed

pulse_jsons = list(
    pulse_dir.joinpath("ActualGR").glob("*_actual_gates.json")
)
flare_jsons = list(
    flare_dir.joinpath("ActualGR").glob("*_actual_gates.json")
)

pulse_final_json = max(pulse_jsons, key=hqc_for_json)
flare_final_json = max(flare_jsons, key=hqc_for_json)

pulse_100_json = min(
    (
        path
        for path in pulse_jsons
        if hqc_for_json(path) >= 100
    ),
    key=hqc_for_json,
)
flare_200_json = min(
    (
        path
        for path in flare_jsons
        if hqc_for_json(path) >= 200
    ),
    key=hqc_for_json,
)

pulse_final_grid = (
    pulse_dir
    / "ActualGR_native_grid"
    / f"{pulse_final_json.stem}_native_gp_grid_3d.npz"
)
pulse_100_grid = (
    pulse_dir
    / "ActualGR_native_grid"
    / f"{pulse_100_json.stem}_native_gp_grid_3d.npz"
)

flare_final_grid = (
    flare_dir
    / "ActualGR_grid"
    / f"{flare_final_json.stem}_gp_grid_3d.npz"
)
flare_200_grid = (
    flare_dir
    / "ActualGR_grid"
    / f"{flare_200_json.stem}_gp_grid_3d.npz"
)

for path in (
    pulse_final_grid,
    pulse_100_grid,
    flare_final_grid,
    flare_200_grid,
):
    if not path.exists():
        raise FileNotFoundError(path)

print(f"PULSE seed: {pulse_seed}")
print(f"  first >=100 HQC: {hqc_for_json(pulse_100_json):.3f}")
print(f"  final:           {hqc_for_json(pulse_final_json):.3f}")
print(f"FLARE seed: {flare_seed}")
print(f"  first >=200 HQC: {hqc_for_json(flare_200_json):.3f}")
print(f"  final:           {hqc_for_json(flare_final_json):.3f}")


plot3d.plot_fle_isosurface_3d_matplotlib(
    json_path=pulse_final_json,
    grid_path=pulse_final_grid,
    png_path=OUTPUT_DIR / "pulse_final.png",
    xlim=(0.1, 0.97),
    ylim=(20, 56),
    legend_labels={
        "Success-side volume": "Success volume",
        "PULSE posterior mean boundary": "Levelset",
        "PULSE - 2 sigma": None,
        "PULSE + 2 sigma":None,
    },
)

plot3d.plot_fle_isosurface_3d_matplotlib(
    json_path=flare_final_json,
    grid_path=flare_final_grid,
    png_path=OUTPUT_DIR / "flare_final.png",
    xlim=(0.1, 1),
    ylim=(20, 56),
    legend_labels={
        "Success-side volume": "Success volume",
        "FLARE GP P(success) = 0.5": "Levelset",
        "FLARE - 2 sigma": None,
        "FLARE + 2 sigma": None,
    },
)


#######SATuRATION GRIDS######
plot3d.plot_fle_isosurface_3d_matplotlib(
    json_path=pulse_100_json,
    grid_path=pulse_100_grid,
    png_path=(
        OUTPUT_DIR
        / f"{pulse_seed}_pulse_first_above_100_hqc_3d.png"
    ),
    xlim=(0.1, 0.97),
    ylim=(20, 56),
    legend_labels={
        "Success-side volume": "Success volume",
        "PULSE posterior mean boundary": "Levelset",
        "PULSE - 2 sigma": None,
        "PULSE + 2 sigma": None,
    },
)

plot3d.plot_fle_isosurface_3d_matplotlib(
    json_path=flare_200_json,
    grid_path=flare_200_grid,
    png_path=(
        OUTPUT_DIR
        / f"{flare_seed}_flare_first_above_200_hqc_3d.png"
    ),
    xlim=(0.1, 0.97),
    ylim=(20, 56),
    zlim=(1.5, 3.45),
    legend_labels={
        "Success-side volume": "Success volume",
        "FLARE GP P(success) = 0.5": "Levelset",
        "FLARE - 2 sigma": None,
        "FLARE + 2 sigma": None,
    },
)