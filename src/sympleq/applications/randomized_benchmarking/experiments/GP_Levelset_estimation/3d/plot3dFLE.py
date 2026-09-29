"""Static Matplotlib counterpart of plot_score_fle_3d.py."""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

from plot_score_fle_3d import (
    FAKE_ANCHOR_SLICES,
    GATE_AXIS_MAX_LOG10,
    GATE_AXIS_MIN_LOG10,
    GP_GRID_PATH,
    ISOSURFACE_OPACITY,
    ONE_SIGMA_SURFACE_OPACITY,
    RATIO_AXIS_MAX,
    RATIO_AXIS_MIN,
    RMB_JSON_PATH,
    SHOW_FAILURE_POINTS,
    SHOW_FAKE_ANCHORS,
    SHOW_MEASURED_POINTS,
    SHOW_ONE_SIGMA_SURFACES,
    SHOW_SUCCESS_POINTS,
    SHOW_TWO_SIGMA_SURFACES,
    TWO_SIGMA_SURFACE_OPACITY,
    fake_anchor_points_from_grid,
    json_path_from_grid_path,
    load_3d_grid,
    load_points_from_jsons,
    sibling_grid_path,
    slice_json_paths_for_grid,
    source_json_path_from_grid_metadata,
)


PNG_PATH: Path | None = None
FIGSIZE = (8.8, 7.0)
DPI = 220
VOLUME_RATIO_STRIDE = 4
VOLUME_GATE_STRIDE = 3


def sibling_png_path(json_path: Path) -> Path:
    return json_path.parent / f"{json_path.stem}_fle_isosurface_3d_matplotlib.png"


def png_path_from_grid_path(grid_path: Path) -> Path:
    return grid_path.parent / f"{grid_path.stem}_fle_isosurface_3d_matplotlib.png"


def axis_from_grid(grid_array: np.ndarray, axis: int) -> np.ndarray:
    indices = [0, 0, 0]
    indices[axis] = slice(None)
    return grid_array[tuple(indices)]


def surface_mesh(
    field: np.ndarray,
    level: float,
    qubits_axis: np.ndarray,
    ratio_axis: np.ndarray,
    log_gates_axis: np.ndarray,
    *,
    color: str,
    alpha: float,
) -> Poly3DCollection | None:
    if not np.nanmin(field) <= level <= np.nanmax(field):
        return None
    vertices, faces, _, _ = marching_cubes(field, level=level)
    qubits = np.interp(vertices[:, 0], np.arange(len(qubits_axis)), qubits_axis)
    ratios = np.interp(vertices[:, 1], np.arange(len(ratio_axis)), ratio_axis)
    log_gates = np.interp(
        vertices[:, 2], np.arange(len(log_gates_axis)), log_gates_axis
    )
    coordinates = np.column_stack((ratios, qubits, log_gates))
    return Poly3DCollection(
        coordinates[faces],
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
    )


def axis_edges(centres: np.ndarray) -> np.ndarray:
    midpoints = 0.5 * (centres[:-1] + centres[1:])
    return np.concatenate(
        ([centres[0] - (midpoints[0] - centres[0])], midpoints,
         [centres[-1] + (centres[-1] - midpoints[-1])])
    )


def add_success_volume(
    ax,
    success: np.ndarray,
    ratio_axis: np.ndarray,
    qubits_axis: np.ndarray,
    log_gates_axis: np.ndarray,
    color: str,
    alpha: float,
) -> None:
    ratio_indices = np.arange(0, len(ratio_axis), VOLUME_RATIO_STRIDE)
    gate_indices = np.arange(0, len(log_gates_axis), VOLUME_GATE_STRIDE)
    reduced = success[:, ratio_indices][:, :, gate_indices].transpose(1, 0, 2)
    x, y, z = np.meshgrid(
        axis_edges(ratio_axis[ratio_indices]),
        axis_edges(qubits_axis),
        axis_edges(log_gates_axis[gate_indices]),
        indexing="ij",
    )
    voxels = ax.voxels(
        x,
        y,
        z,
        reduced,
        facecolors=color,
        edgecolors="none",
        alpha=alpha,
        shade=False,
    )
    for collection in voxels.values():
        collection.set_zorder(1)


def plot_fle_isosurface_3d_matplotlib(
    json_path: str | Path = RMB_JSON_PATH,
    grid_path: str | Path | None = GP_GRID_PATH,
    png_path: str | Path | None = PNG_PATH,
) -> None:
    input_path = Path(json_path)

    if input_path.suffix.lower() == ".npz":
        grid_path = input_path if grid_path is None else Path(grid_path)
        json_candidate = json_path_from_grid_path(grid_path)
        if json_candidate is None:
            json_candidate = source_json_path_from_grid_metadata(grid_path)
        measured_json_paths = (
            [json_candidate]
            if json_candidate is not None
            else slice_json_paths_for_grid(grid_path)
        )
        png_path = png_path_from_grid_path(grid_path) if png_path is None else Path(png_path)
    else:
        json_path = input_path
        grid_path = sibling_grid_path(json_path) if grid_path is None else Path(grid_path)
        measured_json_paths = [json_path]
        png_path = sibling_png_path(json_path) if png_path is None else Path(png_path)

    loaded = load_3d_grid(grid_path)
    gates_grid = loaded["gates_grid"]
    ratio_grid = loaded["ratio_grid"]
    qubits_grid = loaded["qubits_grid"]
    probabilities = loaded["probabilities"]
    latent_mean = loaded["latent_mean"]
    latent_variance = loaded["latent_variance"]
    target = float(loaded["target"])
    with np.load(grid_path) as grid_file:
        native_model = (
            str(np.asarray(grid_file["native_model"]).item())
            if "native_model" in grid_file.files
            else ""
        )
    native_costaware = native_model.startswith("cost_aware_")
    method_label = "PULSE" if native_costaware else "FLARE"
    main_color = "#d85454" if native_costaware else  "#6563ee"
    sigma_color = "#3e0404" if native_costaware else  "#093060"
    volume_color = "#f8d7da" if native_costaware else "#95bbe9"
    main_alpha = 0.6 if native_costaware else ISOSURFACE_OPACITY
    sigma_alpha_scale = 0.5 if native_costaware else 1.0
    volume_alpha = 0.5 if native_costaware else 0.5

    qubits_axis = axis_from_grid(qubits_grid, 0)
    ratio_axis = axis_from_grid(ratio_grid, 1)
    log_gates_axis = np.log10(axis_from_grid(gates_grid, 2))
    gate_mask = (
        (log_gates_axis >= GATE_AXIS_MIN_LOG10)
        & (log_gates_axis <= GATE_AXIS_MAX_LOG10)
    )
    if np.count_nonzero(gate_mask) < 2:
        raise ValueError("Fewer than two gate-grid points lie inside the plot range")
    log_gates_axis = log_gates_axis[gate_mask]
    probabilities_for_plot = probabilities[:, :, gate_mask]
    latent_mean_for_plot = (
        None if latent_mean is None else latent_mean[:, :, gate_mask]
    )
    latent_variance_for_plot = (
        None if latent_variance is None else latent_variance[:, :, gate_mask]
    )

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False

    main_field = (
        latent_mean_for_plot
        if native_costaware and latent_mean_for_plot is not None
        else probabilities_for_plot
    )
    main_level = 0.0 if native_costaware else target
    main_label = (
        "PULSE posterior mean boundary"
        if native_costaware
        else f"FLARE GP P(success) = {target:g}"
    )
    add_success_volume(
        ax,
        np.isfinite(main_field) & (main_field >= main_level),
        ratio_axis,
        qubits_axis,
        log_gates_axis,
        volume_color,
        volume_alpha,
    )
    ax.plot(
        [], [], [],
        color=volume_color,
        linewidth=8,
        alpha=0.8,
        label="Success-side volume",
    )
    main_mesh = surface_mesh(
        main_field,
        main_level,
        qubits_axis,
        ratio_axis,
        log_gates_axis,
        color=main_color,
        alpha=main_alpha,
    )
    if main_mesh is not None:
        ax.add_collection3d(main_mesh)
        main_mesh.set_zorder(4)
        ax.plot([], [], [], color=main_color, linewidth=5, label=main_label)

    if SHOW_ONE_SIGMA_SURFACES or SHOW_TWO_SIGMA_SURFACES:
        if latent_mean_for_plot is None or latent_variance_for_plot is None:
            print(
                "[warning] skipped sigma surfaces: "
                "latent_mean/latent_variance not found in grid."
            )
        else:
            latent_std = np.sqrt(np.clip(latent_variance_for_plot, 0.0, None))
            latent_target = 0.0 if native_costaware else NormalDist().inv_cdf(target)
            sigma_surfaces = []
            if SHOW_ONE_SIGMA_SURFACES:
                sigma_surfaces.extend(
                    [
                        (f"{method_label} - 1 sigma", latent_mean_for_plot - latent_std,
                         sigma_color, ONE_SIGMA_SURFACE_OPACITY * sigma_alpha_scale),
                        (f"{method_label} + 1 sigma", latent_mean_for_plot + latent_std,
                         sigma_color, ONE_SIGMA_SURFACE_OPACITY * sigma_alpha_scale),
                    ]
                )
            if SHOW_TWO_SIGMA_SURFACES:
                sigma_surfaces.extend(
                    [
                        (f"{method_label} - 2 sigma", latent_mean_for_plot - 2.0 * latent_std,
                         sigma_color, TWO_SIGMA_SURFACE_OPACITY * sigma_alpha_scale),
                        (f"{method_label} + 2 sigma", latent_mean_for_plot + 2.0 * latent_std,
                         sigma_color, TWO_SIGMA_SURFACE_OPACITY * sigma_alpha_scale),
                    ]
                )
            for label, field, color, alpha in sigma_surfaces:
                mesh = surface_mesh(
                    field,
                    latent_target,
                    qubits_axis,
                    ratio_axis,
                    log_gates_axis,
                    color=color,
                    alpha=alpha,
                )
                if mesh is None:
                    print(f"[warning] skipped {label}: it does not cross the grid")
                    continue
                ax.add_collection3d(mesh)
                mesh.set_zorder(3)
                ax.plot([], [], [], color=color, linewidth=5, label=label)

    if SHOW_FAKE_ANCHORS:
        gates, ratios, qubits = fake_anchor_points_from_grid(
            gates_grid,
            ratio_grid,
            qubits_grid,
            n_qubit_slices=FAKE_ANCHOR_SLICES,
        )
        ax.scatter(
            ratios,
            qubits,
            np.log10(gates),
            marker="s",
            color="gray",
            s=34,
            depthshade=False,
            label="Fake corner anchors",
        )

    if SHOW_MEASURED_POINTS:
        gates, ratios, qubits, outcomes = load_points_from_jsons(measured_json_paths)
        if SHOW_FAILURE_POINTS and np.any(~outcomes):
            ax.scatter(
                ratios[~outcomes],
                qubits[~outcomes],
                np.log10(gates[~outcomes]),
                marker="x",
                color="seagreen",
                s=36,
                linewidths=1.4,
                depthshade=False,
                label="Failure",
            )
        if SHOW_SUCCESS_POINTS and np.any(outcomes):
            ax.scatter(
                ratios[outcomes],
                qubits[outcomes],
                np.log10(gates[outcomes]),
                marker="o",
                facecolors="none",
                edgecolors="mediumpurple",
                s=34,
                linewidths=1.5,
                depthshade=False,
                label="Success",
            )

    ax.set_xlim(RATIO_AXIS_MIN, RATIO_AXIS_MAX)
    ax.set_ylim(20, 60)
    ax.set_zlim(GATE_AXIS_MIN_LOG10, GATE_AXIS_MAX_LOG10)
    ax.set_xlabel("Two-qubit gate ratio")
    ax.set_ylabel("Qubits", labelpad=10)
    ax.set_zlabel("log10(total gates)")
    title_prefix = "3D PULSE level set" if native_costaware else "3D FLARE level set"
    ax.set_title(f"{title_prefix} | P(success)={target:g}")
    ax.view_init(elev=24, azim=-60)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout(rect=(0.0, 0.0, 0.94, 1.0))

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    print(f"[grid] loaded: {grid_path}")
    print(f"[grid] probabilities shape: {probabilities.shape}")
    print(
        "[grid] probability range: "
        f"{float(np.nanmin(probabilities)):.6g} to "
        f"{float(np.nanmax(probabilities)):.6g}"
    )
    print(f"[grid] target: {target}")
    print(f"[saved] Matplotlib 3D isosurface: {png_path}")
    plt.show()


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else RMB_JSON_PATH
    plot_fle_isosurface_3d_matplotlib(path)
