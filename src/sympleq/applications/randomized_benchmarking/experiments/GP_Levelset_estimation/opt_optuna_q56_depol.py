"""Fit depolarizing noise to fixed H2-1 H-wrapper q56 circuits."""

from opt_optuna_q26_depol import run_sweep


if __name__ == "__main__":
    run_sweep(56)
