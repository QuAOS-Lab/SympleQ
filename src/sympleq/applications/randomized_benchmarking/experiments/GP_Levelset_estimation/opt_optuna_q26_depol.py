"""Fit depolarizing noise to fixed H2-1 H-wrapper circuits."""

import csv
import json
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from scipy.special import ndtri

from opt_optuna import contour_losses, contour_points
from sympleq.applications.randomized_benchmarking.experiments.common import measurement_rng
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
from sympleq.core.noise.noise_model import DepolarizingNoise
from sympleq.integrations.quantinuum.utils import NATIVE_GATES_SET


ONE_Q_NOISE = 0.000025
TWO_Q_NOISE = 0.00079
N_TRIALS = 1000

RUNS = {
    26: {
        "target": Path(
            "Personal/FLE/H_wrapper/H2_1/q26/seed_42/FLE_H_wrapper_20260827_151619/"
            "measurement_017_globalsur_20260831_225325_588812.json"
        ),
        "reference_csv": Path(
            "Personal/Data/Paper_plots/plot_2/plot2_q26_with_fake_h_h2_1_contour.csv"
        ),
        "reference_grid": Path(
            "Personal/FLE/H_wrapper/H2_1/q26/seed_42/FLE_H_wrapper_20260827_151619/"
            "measurement_017_globalsur_20260831_225325_588812_actual_gates_gp_grid.npz"
        ),
        "output": Path("Personal/Data/Paper_plots/optuna_depol_26"),
    },
    56: {
        "target": Path(
            "Personal/FLE/H_wrapper/H2_1/q56/seed_42/FLE_H_wrapper_20260825_160151/"
            "measurement_026_globalsur_20260827_210327_151079.json"
        ),
        "reference_csv": Path(
            "Personal/Data/Paper_plots/plot_2/plot2_q56_with_fake_h_h2_1_contour.csv"
        ),
        "reference_grid": Path(
            "Personal/FLE/H_wrapper/H2_1/q56/seed_42/FLE_H_wrapper_20260825_160151/"
            "measurement_026_globalsur_20260827_210327_151079_actual_gates_gp_grid.npz"
        ),
        "output": Path("Personal/Data/Paper_plots/optuna_depol_56"),
    },
}


def load_hwrapped_circuits(target_path, n_qubits):
    target_step = int(re.match(r"measurement_(\d+)_", target_path.name).group(1))
    checkpoints = []
    for path in target_path.parent.glob("measurement_*.json"):
        match = re.fullmatch(
            r"measurement_(\d+)_(?:sobol|globalsur)_\d{8}_\d{6}_\d{6}\.json",
            path.name,
        )
        if match and int(match.group(1)) <= target_step:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort()

    circuits = []
    for step, path in checkpoints:
        payload = json.loads(path.read_text(encoding="utf-8"))
        experiment = payload.get("experiment", {})
        sent = experiment.get("sent_configs", [])
        actual = experiment.get("actual_after_elimination", [])
        if len(sent) != len(actual):
            raise ValueError(f"{path}: sent/actual metadata length mismatch")

        for request_index, (record, realized) in enumerate(zip(sent, actual)):
            if int(record["n_qubits"]) != n_qubits:
                raise ValueError(f"{path}: expected q={n_qubits}")
            shot_index = realized.get("first_shot_index")
            if shot_index is None:
                raise ValueError(f"{path}: missing first_shot_index for request {request_index}")
            expected = (int(realized["n_1qb_gates"]), int(realized["n_2qb_gates"]))
            config = (
                HWrappedRMBConfig.default()
                .with_n_qubits(n_qubits)
                .with_n_1qb_gates(int(record["n_1qb_gates"]))
                .with_n_2qb_gates(int(record["n_2qb_gates"]))
                .with_random_elimination(float(record.get("random_elimination", 0.0)))
                .with_use_scrambler(bool(record.get("use_scrambler", True)))
                .with_gates_set(tuple(NATIVE_GATES_SET))
            )
            rng = measurement_rng(42, config, int(shot_index))
            circuit = config.random_circuit(rng=rng)
            n_1q = sum(g.n_qudits == 1 and g.name != "Id" for g in circuit.gates)
            n_2q = sum(g.n_qudits == 2 and g.name != "Id" for g in circuit.gates)
            if (n_1q, n_2q) != expected:
                raise ValueError(
                    f"{path}: request {request_index} native gate mismatch "
                    f"{(n_1q, n_2q)} != {expected}"
                )
            circuits.append(
                (step, request_index, config, circuit, n_1q, n_2q,
                 rng.bit_generator.state)
            )
    return circuits


def simulate_grid(circuits, n_qubits, one_q_multiplier, two_q_multiplier, run_dir):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(42)
    device = torch.device("cpu")
    kwargs = control_panel_settings_kwargs()
    kwargs.update(n_qubits=n_qubits, rng_seed=42, use_gpu=False)
    settings = FantasySettings(**kwargs)
    strategy = build_strategy(settings)
    observations = []
    seed_fake_corners(strategy, settings, observations, [], device=device)
    results = []

    for step, request_index, config, circuit, n_1q, n_2q, rng_state in circuits:
        rng = np.random.default_rng()
        rng.bit_generator.state = rng_state
        one_q_noise = DepolarizingNoise(one_q_multiplier * ONE_Q_NOISE, rng=rng)
        two_q_noise = DepolarizingNoise(two_q_multiplier * TWO_Q_NOISE, rng=rng)
        circuit.with_noise(one_q_noise).with_two_qudit_noise(two_q_noise)
        success = int(circuit.act(config.initial_state()) == config.initial_state())
        point = raw_point(n_1q + n_2q, n_2q / (n_1q + n_2q), device=device)
        observation = Observation(point, success, "fixed_hwrapped_circuit")
        add_observation_to_strategy(strategy, observation, device=device)
        observations.append(observation)
        results.append({
            "step": step,
            "request_index": request_index,
            "n_1q": n_1q,
            "n_2q": n_2q,
            "success": success,
        })

    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / f"fixed_hwrapped_q{n_qubits}.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    prediction = refreshed_strategy_for_prediction(
        strategy, settings, observations, device=device
    )
    grid_settings = replace(settings, n_gates_bounds=(200, 2000), ratio_bounds=(0.1, 0.97))
    return save_gp_prediction_grid(
        prediction, grid_settings, device=device, json_path=results_path
    )


def save_plot(reference_points, segments, grid_path, reference_grid, path,
              n_qubits, one_q_multiplier, two_q_multiplier):
    fig, ax = plt.subplots(figsize=(9, 6))
    for source, color, label in (
        (reference_grid, "navy", "H2-1 with Fake H"),
        (grid_path, "firebrick", "SympleQ depolarizing"),
    ):
        with np.load(source) as grid:
            target = float(grid["target"].item())
            band = np.abs(grid["latent_mean"] - ndtri(target)) - np.sqrt(
                np.maximum(grid["latent_variance"], 0.0)
            )
            finite = band[np.isfinite(band)]
            if finite.size and finite.min() < 0.0:
                ax.contourf(grid["ratio_grid"], grid["gates_grid"], band,
                            levels=[float(finite.min()), 0.0], colors=[color], alpha=0.18)
                ax.plot([], [], color=color, linewidth=8, alpha=0.18,
                        label=rf"{label} $\mu \pm 1\sigma$")
    ax.plot(reference_points[:, 0], reference_points[:, 1], color="navy",
            label=f"H2-1 with Fake H q{n_qubits}")
    for index, segment in enumerate(segments):
        points = np.asarray(segment)
        ax.plot(points[:, 0], points[:, 1], color="firebrick", linestyle="--",
                label="SympleQ depolarizing" if index == 0 else None)
    ax.set(
        xlabel="Two-qubit gate ratio",
        ylabel="Total gates",
        title=f"1q={one_q_multiplier:.3f}x, 2q={two_q_multiplier:.3f}x",
    )
    ax.set_yscale("log")
    ax.set_xlim(0.1, 0.97)
    ax.set_ylim(200, 2000)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_sweep(n_qubits):
    paths = RUNS[n_qubits]
    circuits = load_hwrapped_circuits(paths["target"], n_qubits)
    reference = np.atleast_1d(
        np.genfromtxt(paths["reference_csv"], delimiter=",", names=True)
    )
    segment = max(
        (reference[reference["segment"] == index]
         for index in np.unique(reference["segment"])),
        key=len,
    )
    reference_points = np.column_stack(
        (segment["two_qubit_gate_ratio"], segment["total_gates"])
    )
    output_dir = paths["output"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"depolarizing_shape_optuna_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    print(f"Loaded {len(circuits)} fixed H2-1 H-wrapper q{n_qubits} circuits")

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "trial", "one_q_multiplier", "two_q_multiplier",
            f"q{n_qubits}_contour", f"q{n_qubits}_vertical_mse",
            f"q{n_qubits}_shape_mse",
            f"q{n_qubits}_slope_mse", f"q{n_qubits}_minimum_loss",
            f"q{n_qubits}_rmin_ref", f"q{n_qubits}_rmin_test",
            f"q{n_qubits}_total_loss",
        ])

        def objective(trial):
            one_q = trial.suggest_float("one_q_multiplier", 0.5, 5.0)
            two_q = trial.suggest_float("two_q_multiplier", 0.5, 5.0)
            run_dir = output_dir / (
                f"trial_{trial.number:04d}_oneq_{one_q:.5f}_twoq_{two_q:.5f}"
            ) / f"q{n_qubits}"
            grid_path = simulate_grid(circuits, n_qubits, one_q, two_q, run_dir)
            segments = contour_points(grid_path)
            plot_path = run_dir / "contour.png"
            save_plot(reference_points, segments, grid_path, paths["reference_grid"],
                      plot_path, n_qubits, one_q, two_q)
            losses = contour_losses(reference_points, max(segments, key=len)) if segments else {
                "vertical_mse": float("nan"), "shape_mse": float("nan"),
                "slope_mse": float("nan"), "minimum_loss": float("nan"),
                "rmin_ref": float("nan"), "rmin_test": float("nan"),
                "total_loss": 1.0e9,
            }
            if not np.isfinite(losses["total_loss"]):
                losses["total_loss"] = 1.0e9
            writer.writerow([
                trial.number, one_q, two_q, json.dumps(segments),
                *(losses[name] for name in (
                    "vertical_mse", "shape_mse", "slope_mse", "minimum_loss",
                    "rmin_ref", "rmin_test", "total_loss",
                )),
            ])
            handle.flush()
            trial.set_user_attr("plot_path", str(plot_path))
            print(f"Trial {trial.number}: 1q={one_q:.6f}x 2q={two_q:.6f}x "
                  f"q{n_qubits}_loss={losses['total_loss']:.8f} plot={plot_path}")
            return losses["total_loss"]

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.enqueue_trial({"one_q_multiplier": 1.0, "two_q_multiplier": 1.0})
        study.optimize(objective, n_trials=N_TRIALS)

    best = study.best_trial
    print(f"Best trial: {best.number} 1q={best.params['one_q_multiplier']:.8f}x "
          f"2q={best.params['two_q_multiplier']:.8f}x loss={best.value:.8f}")
    print(f"Best plot: {best.user_attrs['plot_path']}")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    run_sweep(26)
