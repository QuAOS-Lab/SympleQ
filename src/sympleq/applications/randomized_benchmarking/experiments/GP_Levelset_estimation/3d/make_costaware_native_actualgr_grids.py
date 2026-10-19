from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sympleq.applications.randomized_benchmarking.config import RMBConfig
from sympleq.applications.randomized_benchmarking.experiments.cost_aware_surface_method import (
    CostAwareSurfaceSettings,
    _link_probability,
    _logn_mean_sigma_mesh,
    _stateless_posterior,
)
from sympleq.core.bayesian_estimation import BayesianEstimator


ACCUMULATED_ROOT = Path(r"Personal\Data\accumulated")
OUTPUT_ROOT = ACCUMULATED_ROOT / "uniform_10_5000_grids"
GATES_AXIS = np.geomspace(10.0, 5000.0, 80)
RATIO_AXIS = np.linspace(0.1, 0.97, 80)
QUBITS_AXIS = np.linspace(26.0, 56.0, 16)


def latest_inputs() -> list[Path]:
    paths = []
    for device in ("H2-1", "H2-2"):
        candidates = sorted(
            (ACCUMULATED_ROOT / device).glob(
                f"accumulated_actualgr_{device}_costaware_*.json"
            ),
            key=lambda path: path.stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError(f"No Cost-aware accumulated JSON found for {device}")
        paths.append(candidates[-1])
    return paths


def load_actual_data(payload: dict) -> dict:
    data = {}
    for record in payload["data"]:
        config = RMBConfig(
            n_1qb_gates=int(record["n_1qb_gates"]),
            n_2qb_gates=int(record["n_2qb_gates"]),
            n_qubits=int(record["n_qubits"]),
            random_elimination=float(record.get("random_elimination", 0.1)),
            use_scrambler=bool(record.get("use_scrambler", True)),
        )
        estimator = data.setdefault(config, BayesianEstimator.default())
        for outcome, count in record["results"]:
            estimator.record(bool(outcome), int(count))
    return data


def checkpoint_settings(payload: dict, device: str) -> CostAwareSurfaceSettings:
    source_jsons = payload.get("experiment", {}).get("source_jsons")
    checkpoint = (
        Path(source_jsons[0]).parent.parent / "CostAware_restartable_checkpoint.json"
        if source_jsons
        else Path(payload["actual_gates_rewrite"]["checkpoint_json"])
    )
    saved = {}
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding="utf-8")).get("settings", {})

    fields = CostAwareSurfaceSettings.__dataclass_fields__
    kwargs = {key: value for key, value in saved.items() if key in fields}
    for key in ("q_values", "grid_resolution", "boundary_fit_resolution"):
        if key in kwargs:
            kwargs[key] = tuple(kwargs[key])
    kwargs.update(
        backend_model=device,
        n_gates_bounds=(10, 5000),
        ratio_bounds=(0.1, 0.97),
        save_path=None,
        plot=False,
    )
    return CostAwareSurfaceSettings(**kwargs)


def make_grid(json_path: Path) -> Path:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    device = payload.get("experiment", {}).get("h2_device") or payload["backend"][
        "device_name"
    ]
    data = load_actual_data(payload)
    settings = checkpoint_settings(payload, device)

    print(f"[native-cost-aware] fitting {device}: {len(data)} configs", flush=True)
    params, weights = _stateless_posterior(data, settings)
    if params is None or weights is None:
        raise RuntimeError(f"Native Cost-aware fit failed for {json_path}")

    ratio_surface, qubits_surface = np.meshgrid(RATIO_AXIS, QUBITS_AXIS)
    mean_log_boundary, sigma_log_boundary, mean_boundary = _logn_mean_sigma_mesh(
        params,
        weights,
        settings,
        ratio_surface,
        qubits_surface,
        raw=True,
    )
    lower_2sigma_boundary = np.exp(mean_log_boundary - 2.0 * sigma_log_boundary)
    upper_2sigma_boundary = np.exp(mean_log_boundary + 2.0 * sigma_log_boundary)
    representative_params = np.sum(weights[:, np.newaxis] * params, axis=0)

    qubits_grid, ratio_grid, gates_grid = np.meshgrid(
        QUBITS_AXIS,
        RATIO_AXIS,
        GATES_AXIS,
        indexing="ij",
    )
    probabilities = np.empty_like(gates_grid, dtype=float)
    for q_index, q in enumerate(QUBITS_AXIS):
        probabilities[q_index] = _link_probability(
            representative_params,
            gates_grid[q_index],
            ratio_grid[q_index],
            float(q),
            settings,
        )
    violations = int(np.count_nonzero(np.diff(probabilities, axis=2) > 1e-12))
    if violations:
        raise RuntimeError(f"Native Cost-aware grid has {violations} monotonicity violations")
    latent_mean = mean_log_boundary[..., np.newaxis] - np.log(gates_grid)
    latent_variance = np.broadcast_to(
        (sigma_log_boundary**2)[..., np.newaxis],
        gates_grid.shape,
    ).copy()

    output_dir = (
        json_path.parent.parent / "ActualGR_native_grid"
        if json_path.parent.name == "ActualGR"
        else OUTPUT_ROOT / device
    )
    output_path = output_dir / f"{json_path.stem}_native_gp_grid_3d.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        gates_grid=gates_grid,
        ratio_grid=ratio_grid,
        qubits_grid=qubits_grid,
        gates_axis=GATES_AXIS,
        ratio_axis=RATIO_AXIS,
        qubits_axis=QUBITS_AXIS,
        probabilities=probabilities,
        latent_mean=latent_mean,
        latent_variance=latent_variance,
        target=np.asarray(0.5),
        coordinate_system=np.asarray("actual_total_ratio_n_qubits_3d"),
        gate_axis_spacing=np.asarray("geomspace"),
        source_json=np.asarray(str(json_path)),
        native_model=np.asarray("cost_aware_two_rate_exponential_posterior"),
        representative_parameters=np.asarray(representative_params, dtype=float),
        native_log_boundary_mean=mean_log_boundary,
        native_log_boundary_variance=sigma_log_boundary**2,
        native_boundary_mean=mean_boundary,
        native_boundary_lower_2sigma=lower_2sigma_boundary,
        native_boundary_upper_2sigma=upper_2sigma_boundary,
        native_sigma_level=np.asarray(2.0),
        posterior_nodes=np.asarray(len(params), dtype=int),
    )
    print(f"[native-cost-aware] {device}: {len(data)} configs")
    print(f"[native-cost-aware] posterior nodes={len(params)}")
    print(f"[native-cost-aware] saved {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_paths", nargs="*", type=Path)
    args = parser.parse_args()
    for json_path in args.json_paths or latest_inputs():
        make_grid(json_path)


if __name__ == "__main__":
    main()
