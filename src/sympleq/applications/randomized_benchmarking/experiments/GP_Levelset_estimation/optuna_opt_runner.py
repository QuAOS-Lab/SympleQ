import csv
import json
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from numpy.random import default_rng
from scipy.special import ndtri

from sympleq.applications.randomized_benchmarking.config import RMBConfig
from sympleq.applications.randomized_benchmarking.experiments.common import (
    measurement_rng,
    mixed_sympleq_backend_factory,
)
from sympleq.applications.randomized_benchmarking.experiments.GP_Levelset_estimation.fantasy_levelset_estimation import (
    Observation,
    add_observation_to_strategy,
    build_strategy,
    raw_point,
    refreshed_strategy_for_prediction,
    seed_fake_corners,
)
from sympleq.applications.randomized_benchmarking.experiments.GP_Levelset_estimation.fantasy_levelset_settings import (
    FantasySettings,
    control_panel_settings_kwargs,
)
from sympleq.applications.randomized_benchmarking.experiments.GP_Levelset_estimation.H_wrapper.h_wrapper_config import (
    HWrappedRMBConfig,
)
from sympleq.applications.randomized_benchmarking.experiments.GP_Levelset_estimation.run_FLE import (
    save_gp_prediction_grid,
)
from sympleq.integrations.quantinuum.utils import NATIVE_GATES_SET


ROOT = Path("Personal/Data/optuna_opt")
SUMMARY_CSV = ROOT / "optuna_trials.csv"
N_TRIALS = 2000
GRID_GATES_BOUNDS = (100, 3000)
GRID_RATIO_BOUNDS = (0.1, 0.97)
NO_CONTOUR_LOSS = 1.0e9

DATASETS = {
    "q26_with_vwrap": (
        ROOT /"manifests/h2_1_q26_with_vwrap_manifest.json",
        ROOT / "contours/plot2_q26_with_fake_h_h2_1_contour.csv",
    ),
    "q26_without_vwrap": (
        ROOT / "manifests/h2_1_q26_without_vwrap_manifest.json",
        ROOT / "contours/plot2_q26_without_fake_h_h2_1_contour.csv",
    ),
    "q56_with_vwrap": (
        ROOT / "manifests/h2_1_q56_with_vwrap_manifest.json",
        ROOT / "contours/plot2_q56_with_fake_h_h2_1_contour.csv",
    ),
    "q56_without_vwrap": (
        ROOT / "manifests/h2_1_q56_without_vwrap_manifest.json",
        ROOT / "contours/plot2_q56_without_fake_h_h2_1_contour.csv",
    ),
}


def build_circuits(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
    config_type = HWrappedRMBConfig if manifest["protected_h_wrapper"] else RMBConfig
    circuits = []
    for item in manifest["circuits"]:
        record = item["config"]
        config = (
            config_type.default()
            .with_n_qubits(int(record["n_qubits"]))
            .with_n_1qb_gates(int(record["n_1qb_gates"]))
            .with_n_2qb_gates(int(record["n_2qb_gates"]))
            .with_random_elimination(float(record["random_elimination"]))
            .with_use_scrambler(bool(record["use_scrambler"]))
            .with_gates_set(tuple(NATIVE_GATES_SET))
        )
        rng = measurement_rng(int(item["seed"]), config, int(item["shot_index"]))
        circuit = config.random_circuit(rng=rng)
        n_1q = sum(g.n_qudits == 1 and g.name != "Id" for g in circuit.gates)
        n_2q = sum(g.n_qudits == 2 and g.name != "Id" for g in circuit.gates)
        if "actual_n_1qb_gates" in item and (
            n_1q != int(item["actual_n_1qb_gates"])
            or n_2q != int(item["actual_n_2qb_gates"])
        ):
            raise ValueError(f"Circuit {item['index']} does not match {manifest_path}")
        circuits.append((config, circuit, n_1q, n_2q, deepcopy(rng.bit_generator.state)))
    print(f"[build] {manifest_path}: {len(circuits)} circuits", flush=True)
    return circuits


def load_reference(csv_path):
    data = np.atleast_1d(np.genfromtxt(csv_path, delimiter=",", names=True))
    data = data[np.isclose(data["sigma_level"], 0.0)]
    segment = max(
        (data[data["segment"] == index] for index in np.unique(data["segment"])),
        key=len,
    )
    segment = np.sort(segment, order="point_index")
    return np.column_stack((
        segment["two_qubit_gate_ratio"],
        np.log10(segment["total_gates"]),
    ))


def chamfer_loss(ref, target):
    distances_squared = np.sum(
        (ref[:, None, :] - target[None, :, :]) ** 2,
        axis=2,
    )
    ref_to_target = distances_squared.min(axis=1).sum()
    target_to_ref = distances_squared.min(axis=0).sum()
    return ref_to_target + target_to_ref


def save_contour(grid_path, csv_path):
    with np.load(grid_path) as grid:
        ratio = np.asarray(grid["ratio_grid"], dtype=float)
        gates = np.asarray(grid["gates_grid"], dtype=float)
        mean = np.asarray(grid["latent_mean"], dtype=float)
        target = float(np.asarray(grid["target"]).item())

    fig, ax = plt.subplots()
    contour = ax.contour(ratio, gates, mean, levels=[ndtri(target)])
    segments = [segment for segment in contour.allsegs[0] if len(segment)]
    plt.close(fig)

    with Path(csv_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "segment", "point_index", "two_qubit_gate_ratio",
            "total_gates", "target_probability",
        ])
        if not segments:
            return None
        segment = max(segments, key=len)
        for index, (ratio_value, gates_value) in enumerate(segment):
            writer.writerow([0, index, ratio_value, gates_value, target])

    return np.column_stack((segment[:, 0], np.log10(segment[:, 1])))


def run_dataset(name, prepared, backend, run_dir):
    q = prepared[0][0].n_qubits
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(42)
    device = torch.device("cpu")
    kwargs = control_panel_settings_kwargs()
    kwargs.update(
        n_qubits=q,
        n_gates_bounds=GRID_GATES_BOUNDS,
        ratio_bounds=GRID_RATIO_BOUNDS,
        rng_seed=42,
        use_gpu=False,
    )
    settings = FantasySettings(**kwargs)
    strategy = build_strategy(settings)
    observations = []
    seed_fake_corners(strategy, settings, observations, [], device=device)
    results = []

    for config, circuit, n_1q, n_2q, rng_state in prepared:
        rng = default_rng()
        rng.bit_generator.state = deepcopy(rng_state)
        backend.noise_model.rng = rng
        backend.two_qubit_noise_model.rng = rng
        circuit.with_noise(backend.noise_model).with_two_qudit_noise(
            backend.two_qubit_noise_model
        )
        success = int(circuit.act(config.initial_state()) == config.initial_state())
        point = raw_point(
            n_1q + n_2q,
            n_2q / (n_1q + n_2q),
            device=device,
        )
        observation = Observation(point, success, "fixed_circuit")
        add_observation_to_strategy(strategy, observation, device=device)
        observations.append(observation)
        results.append({"n_1q": n_1q, "n_2q": n_2q, "success": success})

    dataset_dir = run_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    results_path = dataset_dir / "sympleq.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    prediction = refreshed_strategy_for_prediction(
        strategy, settings, observations, device=device
    )
    grid_path = save_gp_prediction_grid(
        prediction, settings, device=device, json_path=results_path
    )
    contour_path = dataset_dir / "contour.csv"
    contour = save_contour(grid_path, contour_path)
    return contour, contour_path


def objective(trial, prepared, references):
    params = {
        "alpha_1q_depol": trial.suggest_float("one_q_depol", 0.1, 10),
        "alpha_2q_depol": trial.suggest_float("two_q_depol", 0.1, 10),
        "alpha_1q_dephase": trial.suggest_float("one_q_dephase", 0.1, 10),
        "alpha_2q_dephase": trial.suggest_float("two_q_dephase", 0.1, 10),
    }
    backend = mixed_sympleq_backend_factory(
        settings=None,
        rng=default_rng(42),
        **params,
    )
    run_dir = ROOT / f"trial_{trial.number:04d}"
    losses = {}
    for name in DATASETS:
        contour, contour_path = run_dataset(name, prepared[name], backend, run_dir)
        losses[name] = (
            NO_CONTOUR_LOSS
            if contour is None
            else float(chamfer_loss(references[name], contour))
        )
        trial.set_user_attr(f"{name}_contour", str(contour_path))
    for name, loss in losses.items():
        trial.set_user_attr(f"{name}_loss", loss)
    return float(sum(losses.values()))


def save_trial(study, trial):
    if trial.state != optuna.trial.TrialState.COMPLETE:
        return
    with SUMMARY_CSV.open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow([
            trial.number,
            trial.params["one_q_depol"],
            trial.params["two_q_depol"],
            trial.params["one_q_dephase"],
            trial.params["two_q_dephase"],
            *(trial.user_attrs[f"{name}_loss"] for name in DATASETS),
            trial.value,
            *(trial.user_attrs[f"{name}_contour"] for name in DATASETS),
        ])
    print(
        f"[optuna] trial={trial.number} loss={trial.value:.8g} "
        f"best_loss={study.best_value:.8g}",
        flush=True,
    )


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    prepared = {name: build_circuits(paths[0]) for name, paths in DATASETS.items()}
    references = {name: load_reference(paths[1]) for name, paths in DATASETS.items()}

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow([
            "trial", "one_q_depol", "two_q_depol",
            "one_q_dephase", "two_q_dephase",
            *(f"{name}_loss" for name in DATASETS),
            "loss",
            *(f"{name}_contour" for name in DATASETS),
        ])

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: objective(trial, prepared, references),
        n_trials=N_TRIALS,
        callbacks=[save_trial],
    )
