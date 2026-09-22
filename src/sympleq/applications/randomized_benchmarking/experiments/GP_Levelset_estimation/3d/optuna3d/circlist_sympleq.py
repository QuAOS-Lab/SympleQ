import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path


from numpy.random import default_rng
from sympleq.applications.randomized_benchmarking.config import RMBConfig
from sympleq.applications.randomized_benchmarking.experiments.common import (
    measurement_rng,
    mixed_sympleq_backend_factory,
)

from sympleq.applications.randomized_benchmarking.experiments.GP_Levelset_estimation.make_h2_1_uniform_10_5000_grids import (
    save_uniform_grid,
)
from sympleq.integrations.quantinuum.utils import NATIVE_GATES_SET


def build_sympleq_circuit_list(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_records = {}
    circuits = []
    total = len(manifest["circuits"])

    for position, item in enumerate(manifest["circuits"], start=1):
        source_path = Path(item["source_json"])
        if source_path not in source_records:
            source_records[source_path] = json.loads(
                source_path.read_text(encoding="utf-8")
            )["data"]

        record = source_records[source_path][item["record_index"]]
        config = (
            RMBConfig.default()
            .with_n_qubits(int(record["n_qubits"]))
            .with_n_1qb_gates(int(record["n_1qb_gates"]))
            .with_n_2qb_gates(int(record["n_2qb_gates"]))
            .with_random_elimination(float(record["random_elimination"]))
            .with_use_scrambler(bool(record["use_scrambler"]))
            .with_gates_set(tuple(NATIVE_GATES_SET))
        )

        rng = measurement_rng(item["seed"], config, item["shot_index"])
        circuit = config.random_circuit(rng=rng)
        actual_n_1q = sum(
            gate.n_qudits == 1 and gate.name != "Id"
            for gate in circuit.gates
        )
        actual_n_2q = sum(
            gate.n_qudits == 2 and gate.name != "Id"
            for gate in circuit.gates
        )

        if (
            actual_n_1q != item["actual_n_1qb_gates"]
            or actual_n_2q != item["actual_n_2qb_gates"]
        ):
            raise ValueError(f"Circuit {item['index']} does not match the manifest")

        circuits.append({
            "item": item,
            "config": config,
            "circuit": circuit,
            "actual_n_1q": actual_n_1q,
            "actual_n_2q": actual_n_2q,
            "rng_state": deepcopy(rng.bit_generator.state),
        })

        if position % 200 == 0 or position == total:
            print(f"[build] circuits={position}/{total}", flush=True)

    return {"manifest_path": manifest_path, "circuits": circuits}


def run_sympleq_3d(prepared, output_path, backend):
    output_path = Path(output_path)
    circuits = prepared["circuits"]
    results = []
    counts = defaultdict(lambda: [0, 0])

    for position, prepared_circuit in enumerate(circuits, start=1):
        item = prepared_circuit["item"]
        config = prepared_circuit["config"]
        circuit = prepared_circuit["circuit"]
        actual_n_1q = prepared_circuit["actual_n_1q"]
        actual_n_2q = prepared_circuit["actual_n_2q"]
        rng = default_rng()
        rng.bit_generator.state = deepcopy(prepared_circuit["rng_state"])

        backend.noise_model.rng = rng
        backend.two_qubit_noise_model.rng = rng

        initial_state = config.initial_state()
        circuit.with_noise(backend.noise_model)
        circuit.with_two_qudit_noise(backend.two_qubit_noise_model)
        success = bool(circuit.act(initial_state) == initial_state)
        counts[(actual_n_1q, actual_n_2q, config.n_qubits)][int(success)] += 1

        results.append({
            "index": item["index"],
            "source_method": item["source_method"],
            "seed": item["seed"],
            "n_qubits": config.n_qubits,
            "n_1q": actual_n_1q,
            "n_2q": actual_n_2q,
            "n_gates": actual_n_1q + actual_n_2q,
            "ratio_2q": actual_n_2q / (actual_n_1q + actual_n_2q),
            "hardware_success": item["hardware_success"],
            "sympleq_success": success,
        })

        if position % 200 == 0 or position == len(circuits):
            print(
                f"[progress] circuits={position}/{len(circuits)}",
                flush=True,
            )

    data = []
    for (n_1q, n_2q, n_qubits), (failures, successes) in counts.items():
        data.append({
            "n_1qb_gates": n_1q,
            "n_2qb_gates": n_2q,
            "n_qubits": n_qubits,
            "random_elimination": 0.0,
            "use_scrambler": True,
            "gates_set": [gate.name for gate in NATIVE_GATES_SET],
            "results": [[False, failures], [True, successes]],
        })

    output_path.write_text(
        json.dumps({
            "source_manifest": str(prepared["manifest_path"]),
            "coordinate_system": "actual_gates",
            "backend": backend.to_dict(),
            "data": data,
            "circuit_results": results,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"[saved] results JSON: {output_path}")

    grid_path = save_uniform_grid(
        output_path,
        source_root=output_path.parent,
        output_root=output_path.parent,
    )
    print(f"[saved] GP grid: {grid_path}")

    return grid_path


if __name__ == "__main__":
    DEVICE = "H2_1"
    MANIFEST_PATH = Path(
        f"Personal/Data/accumulated/all_3d_nominal_circuit_list_{DEVICE}.json"
    )
    OUTPUT_PATH = Path(
        f"Personal/Data/accumulated/all_3d_sympleq_actual_gates_{DEVICE}.json"
    )
    # Standard SympleQ Backend
    backend = mixed_sympleq_backend_factory(
        settings=None,
        rng=default_rng(42),
        alpha_1q_depol=0.5,
        alpha_1q_dephase=0.5,
        alpha_2q_depol=0.5,
        alpha_2q_dephase=0.5,
    )
    prepared = build_sympleq_circuit_list(MANIFEST_PATH)
    run_sympleq_3d(prepared, OUTPUT_PATH, backend)
