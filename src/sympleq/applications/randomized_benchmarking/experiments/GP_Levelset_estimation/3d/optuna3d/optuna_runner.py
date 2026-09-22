import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
from numpy.random import default_rng

from sympleq.applications.randomized_benchmarking.experiments.common import (
    mixed_sympleq_backend_factory,
)
from circlist_sympleq import build_sympleq_circuit_list, run_sympleq_3d


MANIFEST_PATH = Path(
    f"Personal/Data/accumulated/all_3d_nominal_circuit_list_H2_1.json"
)
OUTPUT_DIR = Path("Personal/Data/accumulated/Sympleq/Optuna3d")
SUMMARY_CSV = OUTPUT_DIR / "optuna_3d_trials.csv"
csv_path_ref = Path(
    "Personal/Data/accumulated/uniform_10_5000_grids/H2-1/Qubitwise_contours.csv"
)


def extract_contour_qubitwise(grid_path, csv_path, target=0.5):
    with np.load(grid_path) as grid:
        qubits_grid = np.asarray(grid["qubits_grid"], dtype=float)
        ratio_grid = np.asarray(grid["ratio_grid"], dtype=float)
        gates_grid = np.asarray(grid["gates_grid"], dtype=float)
        probabilities = grid["probabilities"]
    csv_path = Path(csv_path).with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["qubit", "contour"])
        available_q = np.unique(qubits_grid)
        for selected_q in available_q:
            # print(selected_q)
            mask = np.isclose(qubits_grid, selected_q)

            fig, ax = plt.subplots()
            contour = ax.tricontour(
                ratio_grid[mask],
                gates_grid[mask],
                probabilities[mask],
                levels=[target],
            )
            segments = contour.allsegs[0]
            plt.close(fig)

            segment = max(segments, key=len)
            x = segment[:, 0]
            y = segment[:, 1]
            log_y = np.log10(y)

            contour1 = np.column_stack((x, log_y))
            writer.writerow([selected_q, contour1])

    print(f"{csv_path}")
    return csv_path


def dataloader_contours(csv_path, selected_qubit):
    csv_path = Path(csv_path)

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            selected_q = float(row["qubit"])

            if not np.isclose(selected_q, float(selected_qubit)):
                continue

            values = np.fromstring(
                row["contour"]
                .replace("[", " ")
                .replace("]", " ")
                .replace("\n", " "),
                sep=" ",
            )

            if values.size % 2:
                raise ValueError(
                    "The saved contour does not contain complete x/log_y pairs"
                )

            return values.reshape(-1, 2)

    print(f"No contour found for q={selected_qubit}")
    return None


def chamfer_loss(ref, target):
    distances_squared = np.sum(
        (ref[:, None, :] - target[None, :, :]) ** 2,
        axis=2,
    )
    ref_to_target = distances_squared.min(axis=1).sum()
    target_to_ref = distances_squared.min(axis=0).sum()
    return ref_to_target + target_to_ref


def contour_3d_loss(ref_path, target_path):
    losses = []
    for q in range(26, 58, 2):
        ref_contour = dataloader_contours(ref_path, q)
        target_contour = dataloader_contours(target_path, q)
        if ref_contour is None or target_contour is None:
            raise ValueError(f"Missing contour for q={q}")
        losses.append(chamfer_loss(ref_contour, target_contour))

    return float(np.sum(losses))


def objective(trial, prepared):
    one_q_depol = trial.suggest_float("one_q_depol", 0.1, 10)
    two_q_depol = trial.suggest_float("two_q_depol", 0.1, 10)

    one_q_dephase = trial.suggest_float("one_q_dephase", 0.1, 10)
    two_q_dephase =  trial.suggest_float("two_q_dephase", 0.1, 10)

    run_dir = OUTPUT_DIR / f"trial_{trial.number:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "3d_sympleq.json"
    backend = mixed_sympleq_backend_factory(
        settings=None,
        rng=default_rng(42),
        alpha_1q_depol=one_q_depol,
        alpha_1q_dephase=one_q_dephase,
        alpha_2q_depol=two_q_depol,
        alpha_2q_dephase=two_q_dephase,
    )

    grid_path_target = run_sympleq_3d(prepared, output_path, backend)
    csv_path_target = extract_contour_qubitwise(
        grid_path_target,
        run_dir / "qubitwise_contours.csv",
        target=0.5,
    )
    trial.set_user_attr("contour_csv", str(csv_path_target))

    return contour_3d_loss(csv_path_ref, csv_path_target)


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
            trial.value,
            trial.user_attrs["contour_csv"],
        ])
    print(
        f"[optuna] trial={trial.number} loss={trial.value:.8g} "
        f"best_loss={study.best_value:.8g}"
    )
    print(f"[saved] trial {trial.number} -> {SUMMARY_CSV}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow([
            "trial", "one_q_depol", "two_q_depol",
            "one_q_dephase", "two_q_dephase", "loss", "contour_csv",
        ])

    prepared = build_sympleq_circuit_list(MANIFEST_PATH)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: objective(trial, prepared),
        n_trials=200,
        callbacks=[save_trial],
    )
