"""Plot q26 comparisons with and without Fake Hadamard."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtri


OUT_PATH_WITH_FAKE_H = Path(
    r"Personal\Data\Paper_plots\plot_2\plot2_q26_with_fake_h.png"
)
OUT_PATH_WITHOUT_FAKE_H = Path(
    r"Personal\Data\Paper_plots\plot_2\plot2_q26_without_fake_h.png"
)

# With Fake Hadamard
Q26_FAKE_H_H21_JSON_PATH = Path(
    r"Personal\FLE\H_wrapper\H2_1\q26\seed_42\FLE_H_wrapper_20260827_151619"
    r"\measurement_017_globalsur_20260831_225325_588812_actual_gates.json"
)
Q26_FAKE_H_H21_GRID_PATH = Path(
    r"Personal\FLE\H_wrapper\H2_1\q26\seed_42\FLE_H_wrapper_20260827_151619"
    r"\measurement_017_globalsur_20260831_225325_588812_actual_gates_gp_grid.npz"
)
Q26_FAKE_H_H21E_JSON_PATH = Path(
    r"Personal\FLE\Fancy_emulator\V_warp\H2_1E\q26\seed_42"
    r"\FLE_V_warp_20260827_121746"
    r"\reconstructed_native_gateset_measurement_019_globalsur_20260831_075831_968266_actual_gates.json"
)
Q26_FAKE_H_H21E_GRID_PATH = Path(
    r"Personal\FLE\Fancy_emulator\V_warp\H2_1E\q26\seed_42"
    r"\FLE_V_warp_20260827_121746"
    r"\reconstructed_native_gateset_measurement_019_globalsur_20260831_075831_968266_actual_gates_gp_grid.npz"
)

# Without Fake Hadamard
Q26_H21_JSON_PATH = Path(
    r"Personal\Data\accumulated\H2-1"
    r"\accumulated_actualgr_H2-1_fle_costaware_20260813_175523_q_slices"
    r"\accumulated_actualgr_H2-1_fle_costaware_20260813_175523_q26.json"
)
Q26_H21_GRID_PATH = Path(
    r"Personal\Data\accumulated\H2-1"
    r"\accumulated_actualgr_H2-1_fle_costaware_20260813_175523_q_slices"
    r"\accumulated_actualgr_H2-1_fle_costaware_20260813_175523_q26_gp_grid.npz"
)
Q26_H21E_JSON_PATH = Path(
    r"Personal\FLE\H2_1E\q26\seed_72\FLE_20260827_120009"
    r"\measurement_012_globalsur_20260829_170736_456309.json"
)
Q26_H21E_GRID_PATH = Path(
    r"Personal\FLE\H2_1E\q26\seed_72\FLE_20260827_120009"
    r"\measurement_012_globalsur_20260829_170736_456309_gp_grid.npz"
)
Q26_H22_JSON_PATH = Path(
    r"Personal\Data\accumulated\H2-2"
    r"\accumulated_actualgr_H2-2_fle_costaware_20260813_175532_q_slices"
    r"\accumulated_actualgr_H2-2_fle_costaware_20260813_175532_q26.json"
)
Q26_H22_GRID_PATH = Path(
    r"Personal\Data\accumulated\H2-2"
    r"\accumulated_actualgr_H2-2_fle_costaware_20260813_175532_q_slices"
    r"\accumulated_actualgr_H2-2_fle_costaware_20260813_175532_q26_gp_grid.npz"
)
Q26_H22E_JSON_PATH = Path(
    r"Personal\FLE\H2_2E\q26\seed_42\FLE_20260827_115747"
    r"\measurement_045_globalsur_20260831_054718_316943_actual_gates.json"
)
Q26_H22E_GRID_PATH = Path(
    r"Personal\FLE\H2_2E\q26\seed_42\FLE_20260827_115747"
    r"\measurement_045_globalsur_20260831_054718_316943_actual_gates_gp_grid.npz"
)

GATES_AXIS_LIMITS = (200.0, 2000.0)
RATIO_AXIS_LIMITS = (0.1, 0.9)
SHOW_PREDICTED_FIDELITY_HUE = False
SHOW_UNCERTAINTY = True
SIGMA_ALPHA = 0.22

SHOW_POINTS_FAKE_H_H21 = False
SHOW_POINTS_FAKE_H_H21E = False
SHOW_POINTS_H21 = False
SHOW_POINTS_H21E = False
SHOW_POINTS_H22 = False
SHOW_POINTS_H22E = False


def load_points(json_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    failure_ratios: list[float] = []
    failure_gates: list[float] = []
    success_ratios: list[float] = []
    success_gates: list[float] = []
    observations = 0

    for record in payload.get("data", []):
        n_1q = int(record["n_1qb_gates"])
        n_2q = int(record["n_2qb_gates"])
        total = n_1q + n_2q
        if total <= 0:
            continue
        ratio = n_2q / total
        counts = {int(outcome): int(count) for outcome, count in record.get("results", [])}
        observations += counts.get(0, 0) + counts.get(1, 0)
        if counts.get(0, 0):
            failure_ratios.append(ratio)
            failure_gates.append(total)
        if counts.get(1, 0):
            success_ratios.append(ratio)
            success_gates.append(total)

    return (
        np.asarray(failure_ratios),
        np.asarray(failure_gates),
        np.asarray(success_ratios),
        np.asarray(success_gates),
        observations,
    )


def add_dataset(fig, ax, dataset: dict, *, show_hue: bool) -> None:
    grid_path = Path(dataset["grid"])
    with np.load(grid_path) as grid:
        ratio_grid = np.asarray(grid["ratio_grid"], dtype=float)
        gates_grid = np.asarray(grid["gates_grid"], dtype=float)
        probabilities = np.asarray(grid["probabilities"], dtype=float)
        latent_mean = np.asarray(grid["latent_mean"], dtype=float)
        latent_std = np.sqrt(
            np.maximum(np.asarray(grid["latent_variance"], dtype=float), 0.0)
        )
        target = float(np.asarray(grid["target"]).item())

    latent_target = float(ndtri(target))
    color = str(dataset["color"])
    linestyle = str(dataset["linestyle"])
    label = str(dataset["label"])
    print(f"[plot] {label}: {grid_path}")
    print(f"[plot] {label} contour target: p={target:g}")

    if show_hue:
        surface = ax.contourf(
            ratio_grid,
            gates_grid,
            probabilities,
            levels=np.linspace(0.0, 1.0, 21),
            cmap="RdYlGn",
            alpha=0.85,
        )
        fig.colorbar(surface, ax=ax, label="Predicted fidelity")

    if SHOW_UNCERTAINTY:
        band = np.abs(latent_mean - latent_target) - latent_std
        finite_band = band[np.isfinite(band)]
        if finite_band.size and float(np.min(finite_band)) < 0.0:
            ax.contourf(
                ratio_grid,
                gates_grid,
                band,
                levels=[float(np.min(finite_band)), 0.0],
                colors=[color],
                alpha=SIGMA_ALPHA,
                zorder=3,
            )
            ax.plot(
                [],
                [],
                color=color,
                linewidth=8.0,
                alpha=SIGMA_ALPHA,
                label=rf"{label} $\mu \pm 1\sigma$",
            )

    ax.contour(
        ratio_grid,
        gates_grid,
        latent_mean,
        levels=[latent_target],
        colors=color,
        linestyles=linestyle,
        linewidths=2.2,
        zorder=6,
    )
    ax.plot(
        [],
        [],
        color=color,
        linestyle=linestyle,
        linewidth=2.2,
        label=f"{label} GP mean p={target:g}",
    )

    failures_x, failures_y, successes_x, successes_y, observations = load_points(
        Path(dataset["json"])
    )
    if bool(dataset["show_points"]):
        ax.scatter(
            failures_x,
            failures_y,
            marker="x",
            s=34,
            color=color,
            linewidths=1.4,
            label=f"{label} failure",
            zorder=8,
        )
        ax.scatter(
            successes_x,
            successes_y,
            marker="o",
            s=40,
            facecolors="white",
            edgecolors=color,
            linewidths=1.2,
            label=f"{label} success",
            zorder=9,
        )
        ax.plot([], [], color="none", label=f"{label} data: {observations} obs")


def make_plot(datasets: tuple[dict, ...], *, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.6))
    for index, dataset in enumerate(datasets):
        add_dataset(
            fig,
            ax,
            dataset,
            show_hue=SHOW_PREDICTED_FIDELITY_HUE and index == 0,
        )

    ax.set_yscale("log")
    ax.set_xlim(*RATIO_AXIS_LIMITS)
    ax.set_ylim(*GATES_AXIS_LIMITS)
    ax.set_xlabel("Two-qubit gate ratio")
    ax.set_ylabel("Total gates")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"[saved] plot: {output_path}")


def main() -> None:
    with_fake_h = (
        {
            "label": "H2-1",
            "json": Q26_FAKE_H_H21_JSON_PATH,
            "grid": Q26_FAKE_H_H21_GRID_PATH,
            "color": "navy",
            "linestyle": "-",
            "show_points": SHOW_POINTS_FAKE_H_H21,
        },
        {
            "label": "H2-1E",
            "json": Q26_FAKE_H_H21E_JSON_PATH,
            "grid": Q26_FAKE_H_H21E_GRID_PATH,
            "color": "cornflowerblue",
            "linestyle": "--",
            "show_points": SHOW_POINTS_FAKE_H_H21E,
        },
    )
    without_fake_h = (
        {
            "label": "H2-1",
            "json": Q26_H21_JSON_PATH,
            "grid": Q26_H21_GRID_PATH,
            "color": "navy",
            "linestyle": "-",
            "show_points": SHOW_POINTS_H21,
        },
        {
            "label": "H2-1E",
            "json": Q26_H21E_JSON_PATH,
            "grid": Q26_H21E_GRID_PATH,
            "color": "cornflowerblue",
            "linestyle": "--",
            "show_points": SHOW_POINTS_H21E,
        },
        {
            "label": "H2-2",
            "json": Q26_H22_JSON_PATH,
            "grid": Q26_H22_GRID_PATH,
            "color": "darkred",
            "linestyle": "-",
            "show_points": SHOW_POINTS_H22,
        },
        {
            "label": "H2-2E",
            "json": Q26_H22E_JSON_PATH,
            "grid": Q26_H22E_GRID_PATH,
            "color": "lightcoral",
            "linestyle": "--",
            "show_points": SHOW_POINTS_H22E,
        },
    )

    make_plot(
        with_fake_h,
        title="With Fake Hadamard",
        output_path=OUT_PATH_WITH_FAKE_H,
    )
    make_plot(
        without_fake_h,
        title="Without Fake Hadamard",
        output_path=OUT_PATH_WITHOUT_FAKE_H,
    )
    plt.show()


if __name__ == "__main__":
    main()
