"""Optimize pure dephasing noise against the H2-1 q26 contour."""

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna

from opt_optuna import (
    contour_losses,
    contour_points,
    load_circuits,
    save_contour_plot,
    simulate_grid,
)


N_TRIALS = 1000
OUTPUT_DIR = Path("Personal/Data/Paper_plots/optuna_4")
CIRCUIT_LIST = Path(
    "Personal/FLE/Fancy_emulator/hardware_q26_nominal/"
    "nominal_q26_20260827_134207/nominal_q26_circuit_list.json"
)
REFERENCE_CSV = Path(
    "Personal/Data/Paper_plots/plot_2/plot2_q26_without_fake_h_h2_1_contour.csv"
)
REFERENCE_GRID = Path(
    "Personal/Data/accumulated/H2-1/"
    "accumulated_actualgr_H2-1_fle_costaware_20260813_175523_q_slices/"
    "accumulated_actualgr_H2-1_fle_costaware_20260813_175523_q26_gp_grid.npz"
)


def main():
    circuits = load_circuits(CIRCUIT_LIST, 26)
    reference = np.atleast_1d(np.genfromtxt(REFERENCE_CSV, delimiter=",", names=True))
    segment = max(
        (reference[reference["segment"] == index] for index in np.unique(reference["segment"])),
        key=len,
    )
    reference_points = np.column_stack((segment["two_qubit_gate_ratio"], segment["total_gates"]))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"dephasing_shape_optuna_{datetime.now():%Y%m%d_%H%M%S}.csv"
    print(f"Loaded {len(circuits)} fixed H2-1 q26 circuits")

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "trial", "one_q_multiplier", "two_q_multiplier", "q26_contour",
            "q26_shape_mse", "q26_slope_mse", "q26_minimum_loss",
            "q26_rmin_ref", "q26_rmin_test", "q26_total_loss",
        ])

        def objective(trial):
            one_q = trial.suggest_float("one_q_multiplier", 0.5, 2.5)
            two_q = trial.suggest_float("two_q_multiplier", 2.5, 3.4)
            run_dir = OUTPUT_DIR / f"trial_{trial.number:04d}_oneq_{one_q:.5f}_twoq_{two_q:.5f}" / "q26"
            grid_path = simulate_grid(circuits, 26, one_q, two_q, run_dir)
            segments = contour_points(grid_path)
            plot_path = run_dir / "contour.png"
            save_contour_plot(reference_points, segments, grid_path, REFERENCE_GRID,
                              plot_path, 26, one_q, two_q)
            losses = contour_losses(reference_points, max(segments, key=len)) if segments else {
                "shape_mse": float("nan"), "slope_mse": float("nan"),
                "minimum_loss": float("nan"), "rmin_ref": float("nan"),
                "rmin_test": float("nan"), "total_loss": 1.0e9,
            }
            if not np.isfinite(losses["total_loss"]):
                losses["total_loss"] = 1.0e9
            writer.writerow([
                trial.number, one_q, two_q, json.dumps(segments),
                *(losses[name] for name in (
                    "shape_mse", "slope_mse", "minimum_loss",
                    "rmin_ref", "rmin_test", "total_loss",
                )),
            ])
            handle.flush()
            trial.set_user_attr("plot_path", str(plot_path))
            print(f"Trial {trial.number}: 1q={one_q:.6f}x 2q={two_q:.6f}x "
                  f"q26_loss={losses['total_loss']:.8f} plot={plot_path}")
            return losses["total_loss"]

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.enqueue_trial({"one_q_multiplier": 1.4, "two_q_multiplier": 2.94})
        study.optimize(objective, n_trials=N_TRIALS)

    best = study.best_trial
    print(f"Best trial: {best.number} 1q={best.params['one_q_multiplier']:.8f}x "
          f"2q={best.params['two_q_multiplier']:.8f}x loss={best.value:.8f}")
    print(f"Best plot: {best.user_attrs['plot_path']}")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
