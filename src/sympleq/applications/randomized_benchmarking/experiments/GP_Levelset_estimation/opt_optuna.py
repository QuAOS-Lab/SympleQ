"""Optimize dephasing noise on fixed H2-1 q56 circuits."""

import csv
import json
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

from sympleq.applications.randomized_benchmarking.config import RMBConfig
from sympleq.applications.randomized_benchmarking.experiments.common import (
    ONE_Q_NOISE,
    TWO_Q_NOISE,
    measurement_rng,
)
from sympleq.applications.randomized_benchmarking.backends.sympleq import SympleqBackend
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
from sympleq.applications.randomized_benchmarking.experiments.GP_Levelset_estimation.run_FLE import save_gp_prediction_grid
from sympleq.core.circuits.gates import GATES
from sympleq.core.noise.noise_model import DephasingNoise
from sympleq.integrations.quantinuum.utils import to_pytket_circuit


# -------------------------------------------------------------------------
# Optuna settings
# -------------------------------------------------------------------------

ONE_Q_MULTIPLIER_MIN = 0.5
ONE_Q_MULTIPLIER_MAX = 2.5

TWO_Q_MULTIPLIER_MIN = 2.0
TWO_Q_MULTIPLIER_MAX = 3.5

N_TRIALS = 1000

LAMBDA_SLOPE = 1.0
LAMBDA_MINIMUM = 1.0


reference_paths = {
    56: Path("Personal/Data/Paper_plots/plot_2/plot2_q56_without_fake_h_h2_1_contour.csv"),
}

reference_grid_paths = {
    q: Path(
        "Personal/Data/accumulated/H2-1/"
        "accumulated_actualgr_H2-1_fle_costaware_20260813_175523_q_slices/"
        f"accumulated_actualgr_H2-1_fle_costaware_20260813_175523_q{q}_gp_grid.npz"
    )
    for q in (56,)
}

circuit_list_paths = {
    56: Path(
        "Personal/FLE/Fancy_emulator/hardware_q56_nominal/nominal_q56_h21_circuit_list.json"
    ),
}

output_dir = Path(
    "Personal/Data/Paper_plots/optuna_3"
)


# -------------------------------------------------------------------------
# Extract p=0.5 contour
# -------------------------------------------------------------------------

def contour_points(grid_path, target=0.5):

    with np.load(grid_path) as grid:
        ratio = grid["ratio_grid"]
        gates = grid["gates_grid"]
        mean = grid["latent_mean"]

    level = ndtri(target)

    if not np.nanmin(mean) <= level <= np.nanmax(mean):
        return []

    fig, ax = plt.subplots()

    contours = ax.contour(
        ratio,
        gates,
        mean,
        levels=[level],
    )

    segments = [
        points.tolist()
        for points in contours.allsegs[0]
        if len(points) > 1
    ]

    plt.close(fig)

    return segments


# -------------------------------------------------------------------------
# Prepare a contour for interpolation
# -------------------------------------------------------------------------

def prepare_line(data):

    data = np.asarray(data, dtype=float)

    # Sort by two-qubit gate ratio
    data = data[np.argsort(data[:, 0])]

    x = data[:, 0]
    y = data[:, 1]

    # Remove duplicate x values
    x_unique, indices = np.unique(x, return_index=True)

    y_unique = y[indices]

    return x_unique, y_unique


# -------------------------------------------------------------------------
# Calculate the three shape losses
# -------------------------------------------------------------------------

def contour_losses(ref_data, test_data, n_points=200):

    x_ref, y_ref = prepare_line(ref_data)
    x_test, y_test = prepare_line(test_data)

    # Common x-range only
    xmin = max(x_ref.min(), x_test.min())
    xmax = min(x_ref.max(), x_test.max())

    if xmin >= xmax:
        return {
            "shape_mse": float("nan"),
            "slope_mse": float("nan"),
            "minimum_loss": float("nan"),
            "total_loss": float("nan"),
            "rmin_ref": float("nan"),
            "rmin_test": float("nan"),
        }

    x_common = np.linspace(
        xmin,
        xmax,
        n_points,
    )

    # Interpolate onto same x-grid
    y_ref_interp = np.interp(
        x_common,
        x_ref,
        y_ref,
    )

    y_test_interp = np.interp(
        x_common,
        x_test,
        y_test,
    )

    # Work in log(gate count)
    log_ref = np.log(y_ref_interp)
    log_test = np.log(y_test_interp)

    # ------------------------------------------------------------------
    # Standardize each line separately.
    #
    # This removes:
    #   1. absolute vertical position
    #   2. overall vertical scale
    #
    # leaving mainly the shape.
    # ------------------------------------------------------------------

    ref_std = np.std(log_ref)
    test_std = np.std(log_test)

    if ref_std == 0.0 or test_std == 0.0:
        return {
            "shape_mse": float("nan"),
            "slope_mse": float("nan"),
            "minimum_loss": float("nan"),
            "total_loss": float("nan"),
            "rmin_ref": float("nan"),
            "rmin_test": float("nan"),
        }

    norm_ref = (
        log_ref - np.mean(log_ref)
    ) / ref_std

    norm_test = (
        log_test - np.mean(log_test)
    ) / test_std

    # ------------------------------------------------------------------
    # Loss 1:
    # direct normalized shape mismatch
    # ------------------------------------------------------------------

    shape_mse = np.mean(
        (norm_ref - norm_test) ** 2
    )

    # ------------------------------------------------------------------
    # Loss 2:
    # mismatch between slopes
    #
    # This asks whether the two curves rise/fall in the same way.
    # ------------------------------------------------------------------

    slope_ref = np.gradient(
        norm_ref,
        x_common,
    )

    slope_test = np.gradient(
        norm_test,
        x_common,
    )

    slope_mse = np.mean(
        (slope_ref - slope_test) ** 2
    )

    # ------------------------------------------------------------------
    # Loss 3:
    # location of minimum
    # ------------------------------------------------------------------

    rmin_ref = x_common[
        np.argmin(y_ref_interp)
    ]

    rmin_test = x_common[
        np.argmin(y_test_interp)
    ]

    minimum_loss = (
        rmin_ref - rmin_test
    ) ** 2

    # ------------------------------------------------------------------
    # Total shape-driven objective
    # ------------------------------------------------------------------

    total_loss = (
        shape_mse
        + LAMBDA_SLOPE * slope_mse
        + LAMBDA_MINIMUM * minimum_loss
    )

    return {
        "shape_mse": float(shape_mse),
        "slope_mse": float(slope_mse),
        "minimum_loss": float(minimum_loss),
        "total_loss": float(total_loss),
        "rmin_ref": float(rmin_ref),
        "rmin_test": float(rmin_test),
    }


def load_circuits(circuit_list_path, n_qubits):
    manifest = json.loads(circuit_list_path.read_text(encoding="utf-8"))
    sources = {}
    circuits = []
    for item in manifest["circuits"]:
        if item["source_device"] != "H2-1" or item["pytket_n_qubits"] != n_qubits:
            continue
        path = item["source_json"]
        if path not in sources:
            sources[path] = json.loads(Path(path).read_text(encoding="utf-8"))["data"]
        record = sources[path][item["record_index"]]
        aliases = {"ZZP": "ZZMax", "ZZP_inv": "ZZMax_inv"}
        config = (
            RMBConfig.default()
            .with_n_qubits(int(record["n_qubits"]))
            .with_n_1qb_gates(int(record["n_1qb_gates"]))
            .with_n_2qb_gates(int(record["n_2qb_gates"]))
            .with_random_elimination(float(record["random_elimination"]))
            .with_use_scrambler(bool(record["use_scrambler"]))
            .with_gates_set(tuple(
                getattr(GATES, aliases.get(name, name)) for name in record["gates_set"]
            ))
        )
        rng = measurement_rng(item["seed"], config, item["shot_index"])
        circuit = config.random_circuit(rng=rng)
        if to_pytket_circuit(circuit).n_gates != item["pytket_n_gates"]:
            raise ValueError(f"Circuit {item['index']} does not match the emulator list")
        n_1q = sum(g.n_qudits == 1 and g.name != "Id" for g in circuit.gates)
        n_2q = sum(g.n_qudits == 2 and g.name != "Id" for g in circuit.gates)
        circuits.append((item, config, circuit, n_1q, n_2q, rng.bit_generator.state))
    return circuits


def simulate_grid(circuits, n_qubits, one_q_multiplier, two_q_multiplier, run_dir):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(42)
    device = torch.device("cpu")
    kwargs = control_panel_settings_kwargs()
    kwargs.update(n_qubits=n_qubits, rng_seed=42, use_gpu=False)
    settings = FantasySettings(**kwargs)
    backend = SympleqBackend(
        noise_model=DephasingNoise(one_q_multiplier * ONE_Q_NOISE),
        two_qubit_noise_model=DephasingNoise(two_q_multiplier * TWO_Q_NOISE),
    )
    strategy = build_strategy(settings)
    observations = []
    seed_fake_corners(strategy, settings, observations, [], device=device)
    results = []
    for item, config, circuit, n_1q, n_2q, rng_state in circuits:
        rng = np.random.default_rng()
        rng.bit_generator.state = rng_state
        initial_state = config.initial_state()
        backend.noise_model.rng = backend.two_qubit_noise_model.rng = rng
        circuit.with_noise(backend.noise_model).with_two_qudit_noise(backend.two_qubit_noise_model)
        success = int(circuit.act(initial_state) == initial_state)
        obs = Observation(raw_point(n_1q + n_2q, n_2q / (n_1q + n_2q), device=device), success, "fixed_circuit")
        add_observation_to_strategy(strategy, obs, device=device)
        observations.append(obs)
        results.append({"index": item["index"], "n_1q": n_1q, "n_2q": n_2q, "success": success})
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / f"fixed_q{n_qubits}.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    prediction = refreshed_strategy_for_prediction(strategy, settings, observations, device=device)
    grid_settings = replace(settings, n_gates_bounds=(200, 2000), ratio_bounds=(0.1, 0.97))
    return save_gp_prediction_grid(prediction, grid_settings, device=device, json_path=results_path)


def save_contour_plot(ref_points, segments, grid_path, reference_grid_path, path,
                      n_qubits, one_q_multiplier, two_q_multiplier):
    fig, ax = plt.subplots(figsize=(9, 6))
    for band_path, color, label in (
        (reference_grid_path, "navy", "H2-1"),
        (grid_path, "firebrick", "SympleQ"),
    ):
        with np.load(band_path) as grid:
            band = np.abs(grid["latent_mean"] - ndtri(float(grid["target"].item()))) - np.sqrt(
                np.maximum(grid["latent_variance"], 0.0)
            )
            finite_band = band[np.isfinite(band)]
            if finite_band.size and finite_band.min() < 0.0:
                ax.contourf(grid["ratio_grid"], grid["gates_grid"], band,
                            levels=[float(finite_band.min()), 0.0],
                            colors=[color], alpha=0.18)
                ax.plot([], [], color=color, linewidth=8, alpha=0.18,
                        label=rf"{label} $\mu \pm 1\sigma$")
    ax.plot(ref_points[:, 0], ref_points[:, 1], color="navy", label=f"H2-1 q{n_qubits}")
    for i, segment in enumerate(segments):
        points = np.asarray(segment)
        ax.plot(points[:, 0], points[:, 1], color="firebrick", linestyle="--",
                label="SympleQ dephasing" if i == 0 else None)
    ax.set(xlabel="Two-qubit gate ratio", ylabel="Total gates",
           title=f"1q={one_q_multiplier:.3f}x, 2q={two_q_multiplier:.3f}x")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 0.97)
    ax.set_ylim(200, 2000)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    circuits = {q: load_circuits(path, q) for q, path in circuit_list_paths.items()}
    for q, items in circuits.items():
        print(f"Loaded {len(items)} fixed H2-1 q{q} circuits")

    # ------------------------------------------------------------------
    # Load reference contour
    # ------------------------------------------------------------------

    references = {}
    for q, path in reference_paths.items():
        reference = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
        segment = max(
            (reference[reference["segment"] == index] for index in np.unique(reference["segment"])),
            key=len,
        )
        references[q] = np.column_stack((segment["two_qubit_gate_ratio"], segment["total_gates"]))

    # ------------------------------------------------------------------
    # Output CSV
    # ------------------------------------------------------------------

    output_path = output_dir / (
        f"dephasing_shape_optuna_"
        f"{datetime.now():%Y%m%d_%H%M%S}.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.writer(handle)

        writer.writerow(
            [
                "trial",
                "one_q_multiplier",
                "two_q_multiplier",
                *[f"q{q}_{name}" for q in (56,) for name in (
                    "contour", "shape_mse", "slope_mse", "minimum_loss",
                    "rmin_ref", "rmin_test", "total_loss",
                )],
            ]
        )

        # -----------------------------------------------------------------
        # Optuna objective
        # -----------------------------------------------------------------

        def objective(trial):

            one_q_multiplier = trial.suggest_float(
                "one_q_multiplier",
                ONE_Q_MULTIPLIER_MIN,
                ONE_Q_MULTIPLIER_MAX,
            )

            two_q_multiplier = trial.suggest_float(
                "two_q_multiplier",
                TWO_Q_MULTIPLIER_MIN,
                TWO_Q_MULTIPLIER_MAX,
            )

            run_dir = output_dir / (
                f"trial_{trial.number:03d}_"
                f"oneq_{one_q_multiplier:.5f}_"
                f"twoq_{two_q_multiplier:.5f}"
            )

            q = 56
            q_dir = run_dir / f"q{q}"
            grid_path = simulate_grid(
                circuits[q], q, one_q_multiplier, two_q_multiplier, q_dir
            )
            segments = contour_points(grid_path)
            plot_path = q_dir / "contour.png"
            save_contour_plot(
                references[q], segments, grid_path, reference_grid_paths[q], plot_path, q,
                one_q_multiplier, two_q_multiplier,
            )
            losses = contour_losses(references[q], max(segments, key=len)) if segments else {
                "shape_mse": float("nan"), "slope_mse": float("nan"),
                "minimum_loss": float("nan"), "rmin_ref": float("nan"),
                "rmin_test": float("nan"), "total_loss": 1.0e9,
            }
            if not np.isfinite(losses["total_loss"]):
                losses["total_loss"] = 1.0e9
            for name, value in losses.items():
                trial.set_user_attr(f"q{q}_{name}", value)
            trial.set_user_attr(f"q{q}_plot_path", str(plot_path))
            writer.writerow([
                trial.number, one_q_multiplier, two_q_multiplier,
                json.dumps(segments),
                *(losses[name] for name in (
                    "shape_mse", "slope_mse", "minimum_loss",
                    "rmin_ref", "rmin_test", "total_loss",
                )),
            ])
            handle.flush()
            print(
                f"Trial {trial.number}: 1q={one_q_multiplier:.6f}x "
                f"2q={two_q_multiplier:.6f}x q56_loss={losses['total_loss']:.8f}"
            )
            return losses["total_loss"]

        # -----------------------------------------------------------------
        # Optuna
        # -----------------------------------------------------------------

        sampler = optuna.samplers.TPESampler(
            seed=42
        )

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
        )

        # Start with the region already known to be sensible.
        study.enqueue_trial(
            {
                "one_q_multiplier": 1.0,
                "two_q_multiplier": 3.0,
            }
        )

        study.optimize(
            objective,
            n_trials=N_TRIALS,
        )

    # ---------------------------------------------------------------------
    # Final result
    # ---------------------------------------------------------------------

    best = study.best_trial
    print(f"Best trial: {best.number}")
    print(f"1q scale: {best.params['one_q_multiplier']:.8f}x")
    print(f"2q scale: {best.params['two_q_multiplier']:.8f}x")
    print(f"q56 loss: {best.value:.8f}")
    print(f"q56 plot: {best.user_attrs['q56_plot_path']}")
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    main()
