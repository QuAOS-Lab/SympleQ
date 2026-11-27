"""Run AQUIRE on the plaquette under one noise model, or a grid of them, saving every run.

Each run: AQUIRE as in the paper (H^4 diagnostics with random computational input states, App. D;
one diagnostic shot per measurement shot; systematic error of Eq. 8b), with measurement and
diagnostic outcomes drawn from the exact noisy outcome probabilities of
noise_models.GateNoiseSimulator. diagnostic_mode="zero" starts every diagnostic from |0...0>
instead. Seeds depend only on the commutation mode and the repeat, so all models, strengths and
diagnostic modes share the same random allocation.
"""
import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")       # reproducible MCMC; must be set before numba is imported

import json  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402

import numba  # noqa: E402
import numpy as np  # noqa: E402
from sympleq.applications.measurement.aquire import Aquire  # noqa: E402

from noise_models import MODELS, GateNoiseSimulator  # noqa: E402
from plaquette import plaquette_hamiltonian, product_state  # noqa: E402

MODES = ("general", "quditwise")


@numba.njit(cache=False)
def _seed_numba(seed):
    np.random.seed(seed)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    _seed_numba(seed)                                   # AQUIRE's covariance MCMC runs in numba


def run_one(model, strength, mode, repeat, shots=(100, 1000, 10000), diagnostic_mode="random", d=3, g2=0.01):
    """One AQUIRE run. Returns a dict with the exact value and, per update step, estimate and errors."""
    t0 = time.time()
    H, psi = plaquette_hamiltonian(d, g2), product_state(d)
    sim = GateNoiseSimulator(H.dimensions, [MODELS[model](strength=strength, d=d)])
    seed = int(np.random.SeedSequence([repeat, MODES.index(mode)]).generate_state(1)[0] % (2**31 - 1))
    seed_all(seed)
    aquire = Aquire(H=H, psi=psi, commutation_mode=mode, enable_diagnostics=True, diagnostic_mode=diagnostic_mode,
                    outcome_distribution_function=sim)
    aquire.simulate_observable(update_steps=list(shots))
    stat = np.asarray(aquire.statistical_variance, dtype=float)
    syst = np.asarray(aquire.systematic_variance, dtype=float)
    return {"model": model, "strength": strength, "mode": mode, "repeat": repeat, "diagnostic_mode": diagnostic_mode,
            "seed": seed, "parameters": sim.models[0].describe(), "exact": float(np.real(aquire.true_mean_value)),
            "shots": list(shots), "estimate": [float(np.real(x)) for x in aquire.estimated_mean],
            "statistical_error": np.sqrt(stat).tolist(), "systematic_error": np.sqrt(syst).tolist(),
            "error_bar": np.sqrt(stat + syst).tolist(), "seconds": round(time.time() - t0, 1)}


def _run(args):
    return run_one(*args)


def run_grid(models, strengths, repeats, shots=(100, 1000, 10000), modes=MODES, path="results.jsonl", workers=2,
             diagnostic_mode="random"):
    """Run every (model, strength, mode, repeat) not already in path, in parallel; returns the saved runs of this
    grid."""
    def key(r):
        # runs saved before diagnostic_mode was a setting used zero-mode diagnostics
        return (r["model"], float(r["strength"]), r["mode"], int(r["repeat"]), tuple(r["shots"]),
                r.get("diagnostic_mode", "zero"))
    grid = [(m, float(s), mode, rep, tuple(shots), diagnostic_mode) for m in models for s in strengths
            for mode in modes for rep in range(repeats)]
    done = set()
    if os.path.exists(path):
        done = {key(json.loads(line)) for line in open(path) if line.strip()}
    todo = [run for run in grid if run not in done]
    print(f"{len(todo)} runs to do, {len(grid) - len(todo)} already saved in {path}")
    t0 = time.time()
    with open(path, "a") as f, get_context("spawn").Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_run, todo), 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if k % max(1, len(todo) // 20) == 0 or k == len(todo):
                print(f"  {k}/{len(todo)} done ({time.time() - t0:.0f} s)", flush=True)
    grid = set(grid)
    return [r for r in (json.loads(line) for line in open(path) if line.strip()) if key(r) in grid]
