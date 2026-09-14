"""Plot H2-1 q56 level sets with and without Fake Hadamard."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtri


OUT_PATH_WITH_FAKE_H = Path(
    r"Personal\Data\Paper_plots\plot_2\plot2_q56_with_fake_h.png"
)
OUT_PATH_WITHOUT_FAKE_H = Path(
    r"Personal\Data\Paper_plots\plot_2\plot2_q56_without_fake_h.png"
)

Q56_FAKE_H_JSON_PATH = Path(
    r"Personal\FLE\H_wrapper\H2_1\q56\seed_42\FLE_H_wrapper_20260825_160151"
    r"\measurement_026_globalsur_20260827_210327_151079_actual_gates.json"
)
Q56_FAKE_H_GRID_PATH = Path(
    r"Personal\FLE\H_wrapper\H2_1\q56\seed_42\FLE_H_wrapper_20260825_160151"
    r"\measurement_026_globalsur_20260827_210327_151079_actual_gates_gp_grid.npz"
)

Q56_ORDINARY_JSON_PATH = Path(
    r"Personal\Data\accumulated\H2-1"
    r"\accumulated_actualgr_H2-1_fle_costaware_20260813_175523_q_slices"
    r"\accumulated_actualgr_H2-1_fle_costaware_20260813_175523_q56.json"
)
Q56_ORDINARY_GRID_PATH = Path(
    r"Personal\Data\accumulated\H2-1"
    r"\accumulated_actualgr_H2-1_fle_costaware_20260813_175523_q_slices"
    r"\accumulated_actualgr_H2-1_fle_costaware_20260813_175523_q56_gp_grid.npz"
)

GATES_AXIS_LIMITS = (200.0, 2000.0)
RATIO_AXIS_LIMITS = (0.1, 0.9)
SHOW_PREDICTED_FIDELITY_HUE = False
SHOW_UNCERTAINTY = True
SHOW_DATA_POINTS_WITH_FAKE_H = False
SHOW_DATA_POINTS_WITHOUT_FAKE_H = False
H2_1_COLOR = "navy"
SIGMA_ALPHA = 0.25


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


def plot_panel(
    fig,
    ax,
    *,
    json_path: Path,
    grid_path: Path,
    title: str,
    show_points: bool,
) -> None:
    grid = np.load(grid_path)
    ratio_grid = np.asarray(grid["ratio_grid"], dtype=float)
    gates_grid = np.asarray(grid["gates_grid"], dtype=float)
    probabilities = np.asarray(grid["probabilities"], dtype=float)
    latent_mean = np.asarray(grid["latent_mean"], dtype=float)
    latent_std = np.sqrt(
        np.maximum(np.asarray(grid["latent_variance"], dtype=float), 0.0)
    )
    target = float(np.asarray(grid["target"]).item())
    latent_target = float(ndtri(target))
    print(f"[plot] grid: {grid_path}")
    print(f"[plot] contour target: p={target:g}")

    if SHOW_PREDICTED_FIDELITY_HUE:
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
                colors=[H2_1_COLOR],
                alpha=SIGMA_ALPHA,
                zorder=3,
            )
            ax.plot(
                [],
                [],
                color=H2_1_COLOR,
                linewidth=8.0,
                alpha=SIGMA_ALPHA,
                label=r"H2-1 $\mu \pm 1\sigma$",
            )

    ax.contour(
        ratio_grid,
        gates_grid,
        latent_mean,
        levels=[latent_target],
        colors=H2_1_COLOR,
        linestyles="-",
        linewidths=2.2,
        zorder=6,
    )
    ax.plot(
        [],
        [],
        color=H2_1_COLOR,
        linestyle="-",
        linewidth=2.2,
        label=f"H2-1 GP mean p={target:g}",
    )

    failures_x, failures_y, successes_x, successes_y, observations = load_points(json_path)
    if show_points:
        ax.scatter(
            failures_x,
            failures_y,
            marker="x",
            s=34,
            color=H2_1_COLOR,
            linewidths=1.4,
            label="Failure",
            zorder=8,
        )
        ax.scatter(
            successes_x,
            successes_y,
            marker="o",
            s=40,
            facecolors="white",
            edgecolors=H2_1_COLOR,
            linewidths=1.2,
            label="Success",
            zorder=9,
        )
        ax.plot([], [], color="none", label=f"measured data: {observations} obs")

    ax.set_yscale("log")
    ax.set_xlim(*RATIO_AXIS_LIMITS)
    ax.set_ylim(*GATES_AXIS_LIMITS)
    ax.set_xlabel("Two-qubit gate ratio")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.9)


def main() -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.6))
    plot_panel(
        fig,
        ax,
        json_path=Q56_FAKE_H_JSON_PATH,
        grid_path=Q56_FAKE_H_GRID_PATH,
        title="With Fake Hadamard",
        show_points=SHOW_DATA_POINTS_WITH_FAKE_H,
    )
    ax.set_ylabel("Total gates")
    fig.tight_layout()
    OUT_PATH_WITH_FAKE_H.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH_WITH_FAKE_H, dpi=200, bbox_inches="tight")
    print(f"[saved] plot: {OUT_PATH_WITH_FAKE_H}")

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.6))
    plot_panel(
        fig,
        ax,
        json_path=Q56_ORDINARY_JSON_PATH,
        grid_path=Q56_ORDINARY_GRID_PATH,
        title="Without Fake Hadamard",
        show_points=SHOW_DATA_POINTS_WITHOUT_FAKE_H,
    )
    ax.set_ylabel("Total gates")
    fig.tight_layout()
    OUT_PATH_WITHOUT_FAKE_H.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH_WITHOUT_FAKE_H, dpi=200, bbox_inches="tight")
    print(f"[saved] plot: {OUT_PATH_WITHOUT_FAKE_H}")
    plt.show()


if __name__ == "__main__":
    main()
