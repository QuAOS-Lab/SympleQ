"""Rewrite Personal/Data result JSONs into per-seed ActualGR folders.

The rewritten JSONs use post-random-elimination gate counts:

    n_1qb_gates, n_2qb_gates, n_gates, ratio_2qb_gates

Original JSONs are not modified.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sympleq.applications.randomized_benchmarking.config import RMBConfig
from sympleq.applications.randomized_benchmarking.experiments.common import (
    measurement_rng,
)
from sympleq.core.circuits.gates import DEFAULT_GATES_SET
from sympleq.integrations.quantinuum.utils import NATIVE_GATES_SET


DEFAULT_ROOT = Path("Personal") / "Data"
ACTUALGR = "ActualGR"

GATE_BY_NAME = {
    gate.name: gate
    for gate in [*DEFAULT_GATES_SET, *NATIVE_GATES_SET]
}


@dataclass
class RewriteReport:
    source: Path
    output: Path
    input_records: int
    output_records: int
    observations: int
    output_observations: int | None = None
    source_kind: str = ""
    seed: int | None = None
    actualgr_existed: bool = False
    output_existed: bool = False
    ambiguous_fle_records: int = 0
    dropped_fle_records: int = 0
    dropped_fle_observations: int = 0
    source_observations: int | None = None
    checkpoint_observations: int | None = None
    skipped: bool = False
    reason: str = ""


def seed_from_path(path: Path) -> int:
    match = re.search(r"seed_(\d+)", str(path))
    if match is None:
        raise ValueError(f"Could not infer seed from path: {path}")
    return int(match.group(1))


def config_from_record(record: dict) -> RMBConfig:
    gate_names = record.get("gates_set") or [gate.name for gate in NATIVE_GATES_SET]
    return RMBConfig(
        n_1qb_gates=int(record["n_1qb_gates"]),
        n_2qb_gates=int(record["n_2qb_gates"]),
        n_qubits=int(record["n_qubits"]),
        random_elimination=float(record.get("random_elimination", 0.0)),
        use_scrambler=bool(record.get("use_scrambler", True)),
        gates_set=tuple(GATE_BY_NAME[name] for name in gate_names),
    )


def non_identity_counts(config: RMBConfig, seed: int, shot_index: int) -> tuple[int, int]:
    circuit = config.random_circuit(rng=measurement_rng(seed, config, shot_index))
    n_1q = sum(
        1
        for gate in circuit.gates
        if gate.n_qudits == 1 and gate.name != "Id"
    )
    n_2q = sum(
        1
        for gate in circuit.gates
        if gate.n_qudits == 2 and gate.name != "Id"
    )
    return n_1q, n_2q


def outcome_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "success"}:
        return True
    if text in {"0", "false", "failure"}:
        return False
    raise ValueError(f"Could not parse outcome value: {value!r}")


def result_counts(record: dict) -> Counter[bool]:
    counts: Counter[bool] = Counter()
    for outcome, count in record.get("results", []):
        count = int(count)
        if count > 0:
            counts[outcome_bool(outcome)] += count
    return counts


def results_from_counts(counts: Counter[bool]) -> list[list[bool | int]]:
    return [
        [outcome, int(counts[outcome])]
        for outcome in (False, True)
        if counts[outcome] > 0
    ]


def total_observations(records: list[dict]) -> int:
    return sum(sum(result_counts(record).values()) for record in records)


def actual_template(record: dict, actual_1q: int, actual_2q: int) -> dict:
    rewritten = copy.deepcopy(record)
    total = int(actual_1q + actual_2q)
    rewritten["n_1qb_gates"] = int(actual_1q)
    rewritten["n_2qb_gates"] = int(actual_2q)
    rewritten["n_gates"] = total
    ratio = float(actual_2q / total) if total else 0.0
    rewritten["ratio_2qb_gates"] = ratio
    if "ratio_2_qb_gates" in rewritten:
        rewritten["ratio_2_qb_gates"] = ratio
    return rewritten


def record_key(record: dict) -> tuple:
    return (
        int(record["n_1qb_gates"]),
        int(record["n_2qb_gates"]),
        int(record["n_qubits"]),
        float(record.get("random_elimination", 0.0)),
        bool(record.get("use_scrambler", True)),
        tuple(record.get("gates_set", [])),
    )


def add_result(
    templates: dict[tuple, dict],
    counts: dict[tuple, Counter[bool]],
    template: dict,
    outcome: bool,
    count: int = 1,
) -> None:
    key = record_key(template)
    templates.setdefault(key, template)
    counts.setdefault(key, Counter())[outcome] += int(count)


def output_records_from_groups(
    templates: dict[tuple, dict],
    counts: dict[tuple, Counter[bool]],
) -> list[dict]:
    records = []
    for key, template in templates.items():
        record = copy.deepcopy(template)
        record["results"] = results_from_counts(counts[key])
        records.append(record)
    return records


def latest_measurement_json(run_folder: Path) -> Path | None:
    candidates = [
        path
        for path in run_folder.glob("measurement_*.json")
        if "actual_gates" not in path.stem
    ]
    if candidates:
        return sorted(candidates, key=measurement_sort_key)[-1]

    recoveries = [
        path
        for path in run_folder.glob("FLE_recovery_*.json")
        if "actual_gates" not in path.stem
    ]
    if recoveries:
        return sorted(recoveries)[-1]
    return None


def measurement_sort_key(path: Path) -> tuple[int, str]:
    match = re.match(r"measurement_(\d+)_", path.name)
    step = int(match.group(1)) if match else -1
    return step, path.name


def fle_sources(
    root: Path,
    devices: set[str],
    q_folders: set[str] | None,
    *,
    all_checkpoints: bool = False,
) -> list[tuple[Path, Path, int]]:
    sources: list[tuple[Path, Path, int]] = []
    fle_root = root / "FLE"
    for device in sorted(devices):
        device_root = fle_root / device
        if not device_root.exists():
            print(f"[skip] missing FLE device folder: {device_root}")
            continue
        for q_folder in sorted(path for path in device_root.iterdir() if path.is_dir()):
            if q_folders is not None and q_folder.name not in q_folders:
                continue
            for seed_folder in sorted(q_folder.glob("seed_*")):
                run_folders = sorted(
                    path
                    for path in seed_folder.iterdir()
                    if path.is_dir() and path.name.startswith("FLE_")
                )
                for run_folder in run_folders:
                    if all_checkpoints:
                        run_sources = sorted(
                            (
                                path
                                for path in run_folder.glob("measurement_*.json")
                                if "actual_gates" not in path.stem
                            ),
                            key=measurement_sort_key,
                        )
                    else:
                        source = latest_measurement_json(run_folder)
                        run_sources = [] if source is None else [source]
                    if not run_sources:
                        print(f"[skip] no FLE measurement JSON in {run_folder}")
                        continue
                    sources.extend(
                        (source, seed_folder, seed_from_path(seed_folder))
                        for source in run_sources
                    )
    return sources


def rewrite_fle_json(
    source: Path,
    output: Path,
    seed: int,
    *,
    write: bool,
    overwrite: bool,
    allow_ambiguous: bool,
    drop_ambiguous: bool,
) -> RewriteReport:
    payload = json.loads(source.read_text(encoding="utf-8"))
    source_records = payload.get("data", [])

    templates: dict[tuple, dict] = {}
    grouped_counts: dict[tuple, Counter[bool]] = {}
    ambiguous = 0
    dropped_records = 0
    dropped_observations = 0
    total_records = len(source_records)

    print(f"[rewrite][FLE] seed={seed} source={source}")
    for record_index, record in enumerate(source_records, start=1):
        config = config_from_record(record)
        counts_by_outcome = result_counts(record)
        shots = sum(counts_by_outcome.values())
        if shots <= 0:
            print_fle_progress(seed, record_index, total_records)
            continue

        actual_counts = [
            non_identity_counts(config, seed, shot_index)
            for shot_index in range(shots)
        ]
        unique_actual_counts = set(actual_counts)
        nonzero_outcomes = [outcome for outcome, count in counts_by_outcome.items() if count]

        if len(unique_actual_counts) == 1:
            actual_1q, actual_2q = actual_counts[0]
            template = actual_template(record, actual_1q, actual_2q)
            for outcome, count in counts_by_outcome.items():
                add_result(templates, grouped_counts, template, outcome, count)
            print_fle_progress(seed, record_index, total_records)
            continue

        if len(nonzero_outcomes) == 1:
            outcome = nonzero_outcomes[0]
            for actual_1q, actual_2q in actual_counts:
                template = actual_template(record, actual_1q, actual_2q)
                add_result(templates, grouped_counts, template, outcome)
            print_fle_progress(seed, record_index, total_records)
            continue

        ambiguous += 1
        if drop_ambiguous:
            dropped_records += 1
            dropped_observations += shots
            print(
                f"[drop] ambiguous FLE record: {source} "
                f"record={record_index} shots={shots}"
            )
            print_fle_progress(seed, record_index, total_records)
            continue

        if not allow_ambiguous:
            print_fle_progress(seed, record_index, total_records)
            continue

        expanded_outcomes = [
            outcome
            for outcome in (False, True)
            for _ in range(counts_by_outcome[outcome])
        ]
        for (actual_1q, actual_2q), outcome in zip(actual_counts, expanded_outcomes):
            template = actual_template(record, actual_1q, actual_2q)
            add_result(templates, grouped_counts, template, outcome)

        print(
            f"[warning] ambiguous FLE record assigned by aggregate outcome order: "
            f"{source} record={record_index} shots={shots}"
        )

        print_fle_progress(seed, record_index, total_records)

    output_records = output_records_from_groups(templates, grouped_counts)
    source_observations = total_observations(source_records)
    output_observations = total_observations(output_records)
    report = RewriteReport(
        source=source,
        output=output,
        input_records=len(source_records),
        output_records=len(output_records),
        observations=source_observations,
        output_observations=output_observations,
        source_kind="FLE",
        seed=int(seed),
        actualgr_existed=output.parent.exists(),
        output_existed=output.exists(),
        ambiguous_fle_records=ambiguous,
        dropped_fle_records=dropped_records,
        dropped_fle_observations=dropped_observations,
    )

    if ambiguous and not allow_ambiguous and not drop_ambiguous:
        report.skipped = True
        report.reason = (
            "ambiguous repeated FLE records; rerun with --allow-ambiguous-fle "
            "only if aggregate outcome assignment is acceptable, or "
            "rerun with --drop-ambiguous-fle to omit those records"
        )
        return report

    rewritten = copy.deepcopy(payload)
    rewritten["data"] = output_records
    rewritten["actual_gates_rewrite"] = {
        "source_json": str(source),
        "output_json": str(output),
        "seed": int(seed),
        "source_kind": "FLE",
        "coordinate_system": "actual_after_random_elimination",
        "input_records": len(source_records),
        "output_records": len(output_records),
        "observations": report.observations,
        "output_observations": report.output_observations,
        "ambiguous_fle_records": ambiguous,
        "ambiguous_fle_records_allowed": bool(allow_ambiguous),
        "ambiguous_fle_records_dropped": int(dropped_records),
        "ambiguous_fle_observations_dropped": int(dropped_observations),
    }

    write_payload(output, rewritten, write=write, overwrite=overwrite)
    return report


def print_fle_progress(seed: int, record_index: int, total_records: int) -> None:
    if record_index % 50 == 0 or record_index == total_records:
        print(
            f"[progress][FLE] seed={seed} "
            f"records={record_index}/{total_records}"
        )


def print_ca_progress(seed: int, observations: int, next_progress: int) -> int:
    while observations >= next_progress:
        print(f"[progress][CA] seed={seed} observations={next_progress}")
        next_progress += 50
    return next_progress


def ca_seed_folders(root: Path, devices: set[str]) -> list[Path]:
    folders: list[Path] = []
    ca_root = root / "Cost_aware"
    for device in sorted(devices):
        device_root = ca_root / device
        if not device_root.exists():
            print(f"[skip] missing cost-aware device folder: {device_root}")
            continue
        folders.extend(sorted(path for path in device_root.glob("seed_*") if path.is_dir()))
    return folders


def rewrite_ca_seed(
    seed_folder: Path,
    *,
    write: bool,
    overwrite: bool,
    ca_mismatch: str,
) -> RewriteReport:
    data_path = seed_folder / "CostAware_restartable.json"
    checkpoint_path = seed_folder / "CostAware_restartable_checkpoint.json"
    output_path = seed_folder / ACTUALGR / "CostAware_restartable_actual_gates.json"

    if not data_path.exists() or not checkpoint_path.exists():
        return RewriteReport(
            source=data_path,
            output=output_path,
            input_records=0,
            output_records=0,
            observations=0,
            output_observations=0,
            source_kind="Cost_aware",
            seed=seed_from_path(seed_folder),
            actualgr_existed=output_path.parent.exists(),
            output_existed=output_path.exists(),
            source_observations=0,
            checkpoint_observations=0,
            skipped=True,
            reason="missing CostAware_restartable.json or checkpoint",
        )

    seed = seed_from_path(seed_folder)
    source_payload = json.loads(data_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))

    templates: dict[tuple, dict] = {}
    grouped_counts: dict[tuple, Counter[bool]] = {}
    nominal_1q = nominal_2q = actual_1q_total = actual_2q_total = 0
    observations = 0
    next_progress = 50

    print(f"[rewrite][CA] seed={seed} folder={seed_folder}")
    for batch in checkpoint.get("batch_history", []):
        for request in batch.get("requests", []):
            config_record = request.get("config")
            if not isinstance(config_record, dict):
                continue
            config = config_from_record(config_record)
            first_shot_index = int(request.get("first_shot_index", 0))
            for index, measurement in enumerate(request.get("measurements", [])):
                shot_index = int(measurement.get("shot_index", first_shot_index + index))
                outcome = outcome_bool(measurement["outcome"])
                actual_1q, actual_2q = non_identity_counts(config, seed, shot_index)
                template = actual_template(config_record, actual_1q, actual_2q)
                add_result(templates, grouped_counts, template, outcome)

                nominal_1q += int(config_record["n_1qb_gates"])
                nominal_2q += int(config_record["n_2qb_gates"])
                actual_1q_total += actual_1q
                actual_2q_total += actual_2q
                observations += 1
                next_progress = print_ca_progress(seed, observations, next_progress)
    print(f"[progress][CA] seed={seed} observations={observations} final")

    source_records = source_payload.get("data", [])
    source_observations = total_observations(source_records)
    if source_observations != observations:
        message = (
            f"source observations ({source_observations}) do not match "
            f"checkpoint observations ({observations})"
        )
        if ca_mismatch == "skip":
            return RewriteReport(
                source=data_path,
                output=output_path,
                input_records=len(source_records),
                output_records=0,
                observations=source_observations,
                output_observations=0,
                source_kind="Cost_aware",
                seed=int(seed),
                actualgr_existed=output_path.parent.exists(),
                output_existed=output_path.exists(),
                source_observations=source_observations,
                checkpoint_observations=observations,
                skipped=True,
                reason=message,
            )
        if ca_mismatch != "checkpoint":
            raise RuntimeError(f"{seed_folder}: {message}.")
        print(f"[warning][CA] {seed_folder}: {message}; using checkpoint")

    source_warning = None
    if source_observations != observations:
        source_warning = (
            "Output records were reconstructed from checkpoint measurements; "
            "source JSON data count was stale or incomplete."
        )

    if source_observations != observations and ca_mismatch not in {"checkpoint", "skip"}:
        raise RuntimeError(
            f"{seed_folder}: source observations ({source_observations}) do not "
            f"match checkpoint observations ({observations})."
        )

    output_records = output_records_from_groups(templates, grouped_counts)
    rewritten = copy.deepcopy(source_payload)
    rewritten["data"] = output_records
    rewritten["actual_gates_rewrite"] = {
        "source_json": str(data_path),
        "checkpoint_json": str(checkpoint_path),
        "output_json": str(output_path),
        "seed": int(seed),
        "source_kind": "Cost_aware",
        "coordinate_system": "actual_after_random_elimination",
        "input_records": len(source_records),
        "output_records": len(output_records),
        "observations": observations,
        "source_observations": source_observations,
        "checkpoint_observations": observations,
        "source_checkpoint_mismatch": source_observations != observations,
        "source_checkpoint_mismatch_policy": ca_mismatch,
        "source_checkpoint_mismatch_warning": source_warning,
        "nominal_1qb_gates": nominal_1q,
        "nominal_2qb_gates": nominal_2q,
        "actual_1qb_gates": actual_1q_total,
        "actual_2qb_gates": actual_2q_total,
    }

    actualgr_existed = output_path.parent.exists()
    output_existed = output_path.exists()
    write_payload(output_path, rewritten, write=write, overwrite=overwrite)
    return RewriteReport(
        source=data_path,
        output=output_path,
        input_records=len(source_records),
        output_records=len(output_records),
        observations=observations,
        output_observations=observations,
        source_kind="Cost_aware",
        seed=int(seed),
        actualgr_existed=actualgr_existed,
        output_existed=output_existed,
        source_observations=source_observations,
        checkpoint_observations=observations,
        reason=source_warning or "",
    )


def write_payload(output: Path, payload: dict, *, write: bool, overwrite: bool) -> None:
    if not write:
        return
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_report(report: RewriteReport, *, write: bool) -> None:
    status = "skip" if report.skipped else ("write" if write else "dry-run")
    existed = "existing-output" if report.output_existed else "new-output"
    actualgr = "existing-ActualGR" if report.actualgr_existed else "new-ActualGR"
    output_observations = (
        report.observations
        if report.output_observations is None
        else report.output_observations
    )
    print(
        f"[{status}] {report.source_kind} seed={report.seed} "
        f"{actualgr} {existed} | folder={report.source.parent} | "
        f"{report.source} -> {report.output} | "
        f"records {report.input_records}->{report.output_records} | "
        f"observations {report.observations}->{output_observations}"
    )
    if report.ambiguous_fle_records:
        print(f"  ambiguous_fle_records={report.ambiguous_fle_records}")
    if report.dropped_fle_records:
        print(
            f"  dropped_fle_records={report.dropped_fle_records} "
            f"dropped_observations={report.dropped_fle_observations}"
        )
    if (
        report.source_observations is not None
        and report.checkpoint_observations is not None
        and report.source_observations != report.checkpoint_observations
    ):
        print(
            f"  source_observations={report.source_observations} "
            f"checkpoint_observations={report.checkpoint_observations}"
        )
    if report.reason:
        print(f"  reason={report.reason}")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def print_status_report(
    *,
    root: Path,
    targets: list[str],
    fle_devices: set[str],
    ca_devices: set[str],
    q_folders: set[str] | None,
    all_fle_checkpoints: bool,
    fle_plan: list[tuple[Path, Path, int]],
    ca_plan: list[Path],
    write: bool,
    overwrite: bool,
    skip_existing: bool,
    allow_ambiguous_fle: bool,
    drop_ambiguous_fle: bool,
    ca_mismatch: str,
) -> None:
    mode = "WRITE" if write else "DRY-RUN"
    q_label = "all" if q_folders is None else ", ".join(sorted(q_folders))
    print("=" * 88)
    print("[status] ActualGR actual-gates rewrite")
    print(f"[status] root                  = {display_path(root)}")
    print(f"[status] targets               = {', '.join(targets)}")
    print(f"[status] mode                  = {mode}")
    print(f"[status] overwrite             = {overwrite}")
    print(f"[status] skip existing         = {skip_existing}")
    print(f"[status] allow_ambiguous_fle   = {allow_ambiguous_fle}")
    print(f"[status] drop_ambiguous_fle    = {drop_ambiguous_fle}")
    print(f"[status] ca_mismatch           = {ca_mismatch}")
    print(f"[status] FLE devices           = {', '.join(sorted(fle_devices)) or 'none'}")
    print(f"[status] Cost-aware devices    = {', '.join(sorted(ca_devices)) or 'none'}")
    print(f"[status] FLE q folders         = {q_label}")
    print(f"[status] all FLE checkpoints   = {all_fle_checkpoints}")
    print(f"[status] planned FLE rewrites  = {len(fle_plan)}")
    print(f"[status] planned CA rewrites   = {len(ca_plan)}")
    print("-" * 88)

    for source, seed_folder, seed in fle_plan:
        output = seed_folder / ACTUALGR / f"{source.stem}_actual_gates.json"
        actualgr = "yes" if output.parent.exists() else "no"
        output_state = "exists" if output.exists() else "new"
        print(
            f"[status][FLE] seed={seed} ActualGR={actualgr} output={output_state}"
        )
        print(f"  folder: {display_path(source.parent)}")
        print(f"  seed folder: {display_path(seed_folder)}")
        print(f"  source: {display_path(source)}")
        print(f"  output: {display_path(output)}")

    for seed_folder in ca_plan:
        seed = seed_from_path(seed_folder)
        source = seed_folder / "CostAware_restartable.json"
        checkpoint = seed_folder / "CostAware_restartable_checkpoint.json"
        output = seed_folder / ACTUALGR / "CostAware_restartable_actual_gates.json"
        actualgr = "yes" if output.parent.exists() else "no"
        output_state = "exists" if output.exists() else "new"
        ready = source.exists() and checkpoint.exists()
        print(
            f"[status][CA] seed={seed} ActualGR={actualgr} output={output_state} "
            f"ready={ready}"
        )
        print(f"  folder:     {display_path(seed_folder)}")
        print(f"  source:     {display_path(source)}")
        print(f"  checkpoint: {display_path(checkpoint)}")
        print(f"  output:     {display_path(output)}")

    print("=" * 88)


def print_final_summary(reports: list[RewriteReport], *, write: bool) -> None:
    skipped = [report for report in reports if report.skipped]
    blocked = [report for report in skipped if report.ambiguous_fle_records]
    writes = [report for report in reports if not report.skipped]
    total_observations = sum(report.observations for report in writes)
    total_output_observations = sum(
        report.observations
        if report.output_observations is None
        else report.output_observations
        for report in writes
    )
    total_input_records = sum(report.input_records for report in writes)
    total_output_records = sum(report.output_records for report in writes)
    total_dropped_records = sum(report.dropped_fle_records for report in reports)
    total_dropped_observations = sum(report.dropped_fle_observations for report in reports)
    mode = "write" if write else "dry-run"

    print("-" * 88)
    print(f"[summary] mode                  = {mode}")
    print(f"[summary] successful plans      = {len(writes)}")
    print(f"[summary] skipped plans         = {len(skipped)}")
    print(f"[summary] blocked ambiguous FLE = {len(blocked)}")
    print(f"[summary] input records         = {total_input_records}")
    print(f"[summary] output records        = {total_output_records}")
    print(f"[summary] observations          = {total_observations}->{total_output_observations}")
    print(f"[summary] dropped FLE records   = {total_dropped_records}")
    print(f"[summary] dropped FLE obs       = {total_dropped_observations}")
    print("-" * 88)


def parse_targets(values: list[str]) -> tuple[set[str], set[str]]:
    if "all" in values:
        return {"H2_1", "H2_2"}, {"H2-2"}

    fle_devices: set[str] = set()
    ca_devices: set[str] = set()
    for value in values:
        if value == "fle-h2-1":
            fle_devices.add("H2_1")
        elif value == "fle-h2-2":
            fle_devices.add("H2_2")
        elif value == "ca-h2-2":
            ca_devices.add("H2-2")
        else:
            raise ValueError(f"Unknown target: {value}")
    return fle_devices, ca_devices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["all"],
        choices=["all", "fle-h2-1", "fle-h2-2", "ca-h2-2"],
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually write JSONs. Without this, only prints what would happen.",
    )
    parser.add_argument(
        "--compute-dry-run",
        action="store_true",
        help=(
            "In dry-run mode, also reconstruct circuits and compute output record "
            "counts. Default dry-run is status-only."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip FLE rewrites whose ActualGR output JSON already exists.",
    )
    parser.add_argument(
        "--allow-ambiguous-fle",
        action="store_true",
        help=(
            "Allow FLE records with repeated shots, varied actual counts, and mixed "
            "outcomes to be split by aggregate outcome order."
        ),
    )
    parser.add_argument(
        "--drop-ambiguous-fle",
        action="store_true",
        help="Drop ambiguous repeated FLE records instead of skipping the whole seed.",
    )
    parser.add_argument(
        "--ca-mismatch",
        choices=["error", "skip", "checkpoint"],
        default="checkpoint",
        help=(
            "How to handle cost-aware source/checkpoint observation mismatches. "
            "Default 'checkpoint' writes from checkpoint measurements."
        ),
    )
    parser.add_argument(
        "--q-folders",
        nargs="+",
        default=None,
        help="Optional FLE q folders to include, for example: --q-folders qband_4.",
    )
    parser.add_argument(
        "--all-fle-checkpoints",
        action="store_true",
        help="Rewrite every FLE measurement checkpoint instead of only the latest.",
    )
    args = parser.parse_args(argv)
    if args.allow_ambiguous_fle and args.drop_ambiguous_fle:
        parser.error("--allow-ambiguous-fle and --drop-ambiguous-fle are mutually exclusive")
    if args.overwrite and args.skip_existing:
        parser.error("--overwrite and --skip-existing are mutually exclusive")

    fle_devices, ca_devices = parse_targets(args.targets)
    reports: list[RewriteReport] = []

    q_folders = None if args.q_folders is None else set(args.q_folders)
    fle_plan = fle_sources(
        args.root,
        fle_devices,
        q_folders,
        all_checkpoints=args.all_fle_checkpoints,
    )
    if args.skip_existing:
        pending_fle_plan = []
        for source, seed_folder, seed in fle_plan:
            output = seed_folder / ACTUALGR / f"{source.stem}_actual_gates.json"
            if output.exists():
                print(f"[skip-existing] {output}")
                continue
            pending_fle_plan.append((source, seed_folder, seed))
        fle_plan = pending_fle_plan
    ca_plan = ca_seed_folders(args.root, ca_devices)
    print_status_report(
        root=args.root,
        targets=args.targets,
        fle_devices=fle_devices,
        ca_devices=ca_devices,
        q_folders=q_folders,
        all_fle_checkpoints=args.all_fle_checkpoints,
        fle_plan=fle_plan,
        ca_plan=ca_plan,
        write=args.write,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        allow_ambiguous_fle=args.allow_ambiguous_fle,
        drop_ambiguous_fle=args.drop_ambiguous_fle,
        ca_mismatch=args.ca_mismatch,
    )
    if not args.write and not args.compute_dry_run:
        print("[done] status-only dry-run; no circuits reconstructed and no files written")
        print("[hint] add --compute-dry-run to validate output counts without writing")
        return 0

    for source, seed_folder, seed in fle_plan:
        output = seed_folder / ACTUALGR / f"{source.stem}_actual_gates.json"
        reports.append(
            rewrite_fle_json(
                source,
                output,
                seed,
                write=args.write,
                overwrite=args.overwrite,
                allow_ambiguous=args.allow_ambiguous_fle,
                drop_ambiguous=args.drop_ambiguous_fle,
            )
        )

    for seed_folder in ca_plan:
        reports.append(
            rewrite_ca_seed(
                seed_folder,
                write=args.write,
                overwrite=args.overwrite,
                ca_mismatch=args.ca_mismatch,
            )
        )

    for report in reports:
        print_report(report, write=args.write)

    blocked = [report for report in reports if report.skipped and report.ambiguous_fle_records]
    print_final_summary(reports, write=args.write)
    print(
        f"[done] targets={args.targets} mode={'write' if args.write else 'dry-run'} "
        f"reports={len(reports)} blocked={len(blocked)}"
    )
    return 2 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
