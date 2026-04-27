"""
macro_macro_runner.py
=====================
Top-level orchestrator for the full PnP-OVSS experimental pipeline.

Execution order per (dataset × transformer) pair:
  1  Baseline    — paper defaults (L=8 H=10 for BLIP), 200 samples
  3  Head/Layer  — grid-search L and H,                  30 samples
  2  Patch/Drop  — sweep spatial × granularity × dropout × threshold
                   using best L/H from phase 3,           30 samples
  4  Final Eval  — best overall settings,               200 samples

Each phase writes to outputs/{dataset}/{transformer}/phase_{N}/.
State (best L/H, best patch combo) passes via JSON files.

Usage
-----
  # All datasets, all models
  set_slot 0 .venv/bin/python macro_macro_runner.py

  # Specific subset
  set_slot 0 .venv/bin/python macro_macro_runner.py \\
      --datasets voc chest_xray --models blip bridgetower

  # Dry-run
  python macro_macro_runner.py --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

ROOT   = Path(__file__).resolve().parent
WORKER = str(ROOT / "scripts" / "run_macro_sweep.py")

DATASETS     = ["voc", "chest_xray"]
TRANSFORMERS = ["blip", "bridgetower"]

# Execution order: 1 → 3 → 2 → 4
PHASE_ORDER = [1, 3, 2, 4]

LOG_FILE = "macro_macro_run.log"


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("macro")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


log = setup_logging()


# ── Result tracking ───────────────────────────────────────────────────────────

@dataclass
class PhaseResult:
    phase:   int
    success: bool
    elapsed: float


@dataclass
class PairResult:
    dataset: str
    model:   str
    phases:  list[PhaseResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return all(p.success for p in self.phases)

    @property
    def elapsed(self) -> float:
        return sum(p.elapsed for p in self.phases)


# ── Subprocess helper ─────────────────────────────────────────────────────────

def _run_phase(
    phase: int,
    dataset: str,
    model: str,
    state_dir: Path,
    dry_run: bool,
) -> PhaseResult:
    """Build and run the subprocess call for a single phase."""

    phase_dir   = state_dir / f"phase_{phase}"
    best_pl_json = state_dir / "best_pipeline.json"
    best_co_json = state_dir / "best_combo.json"

    PHASE_LABEL = {
        1: "baseline",
        3: "head-tuning",
        2: "patch-tuning",
        4: "final-eval",
    }
    only = PHASE_LABEL[phase]

    cmd = [
        sys.executable, WORKER,
        "--dataset",         dataset,
        "--transformer",     model,
        "--only-phase",      only,
        "--output-phase-dir", str(phase_dir),
    ]

    # Sample-cap flags
    if phase in (1, 4):
        cmd += ["--max-eval-samples", "200"]
    else:
        cmd += ["--max-tune-samples", "30"]

    # State-passing flags
    if phase == 3:
        cmd += ["--best-pipeline-out", str(best_pl_json)]
    if phase == 2:
        cmd += ["--best-pipeline-in", str(best_pl_json),
                "--best-combo-out",   str(best_co_json)]
    if phase == 4:
        cmd += ["--best-pipeline-in", str(best_pl_json),
                "--best-combo-in",    str(best_co_json)]

    label = f"phase{phase} ({only}) | {dataset} | {model}"

    if dry_run:
        log.info("  [DRY-RUN] %s", label)
        log.info("            %s", " ".join(cmd))
        return PhaseResult(phase, True, 0.0)

    log.info("  ▶ %s", label)
    t0 = time.perf_counter()
    try:
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        elapsed = time.perf_counter() - t0
        log.info("  ✓ %s  (%.1fs)", label, elapsed)
        return PhaseResult(phase, True, elapsed)
    except subprocess.CalledProcessError as exc:
        elapsed = time.perf_counter() - t0
        log.error("  ✗ %s  exit=%d  (%.1fs)", label, exc.returncode, elapsed)
        return PhaseResult(phase, False, elapsed)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        log.error("  ✗ %s  error=%s  (%.1fs)", label, exc, elapsed)
        return PhaseResult(phase, False, elapsed)


def run_pair(dataset: str, model: str, phases: list[int], dry_run: bool) -> PairResult:
    """Run all requested phases for one (dataset, model) pair in dependency order."""
    state_dir = ROOT / "outputs" / dataset / model
    if not dry_run:
        state_dir.mkdir(parents=True, exist_ok=True)

    result = PairResult(dataset=dataset, model=model)

    for phase in phases:
        pr = _run_phase(phase, dataset, model, state_dir, dry_run)
        result.phases.append(pr)
        if not pr.success:
            log.warning("  Phase %d failed — continuing with remaining phases", phase)

    return result


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(results: list[PairResult]) -> None:
    log.info("")
    log.info("=" * 72)
    log.info("  MACRO-MACRO RUN — FINAL SUMMARY")
    log.info("=" * 72)

    for r in results:
        status = "✓" if r.success else "✗"
        phase_summary = "  ".join(
            f"P{p.phase}:{'ok' if p.success else 'FAIL'}" for p in r.phases
        )
        log.info("  %s  %-12s | %-12s  [%s]  (%.1f min)",
                 status, r.dataset, r.model, phase_summary, r.elapsed / 60)

    n_ok   = sum(1 for r in results if r.success)
    n_fail = sum(1 for r in results if not r.success)
    log.info("-" * 72)
    log.info("  Pairs: %d   succeeded: %d   failed: %d", len(results), n_ok, n_fail)
    if n_fail:
        log.info("  Failed pairs:")
        for r in results:
            if not r.success:
                failed_phases = [p.phase for p in r.phases if not p.success]
                log.info("    • %s | %s  (phases %s)", r.dataset, r.model, failed_phases)
    log.info("=" * 72)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Macro-macro orchestrator for PnP-OVSS experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=DATASETS,
        help="Datasets to run.",
    )
    p.add_argument(
        "--models", nargs="+", choices=TRANSFORMERS, default=TRANSFORMERS,
        help="Transformer models to run.",
    )
    p.add_argument(
        "--phases", nargs="+", type=int, choices=[1, 2, 3, 4], default=PHASE_ORDER,
        help="Phases to execute. Default order enforces 1→3→2→4 dependency chain.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them.",
    )
    return p.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    # Preserve dependency order: if user requests [1,2,3,4] sort into [1,3,2,4]
    requested = set(args.phases)
    phases = [p for p in PHASE_ORDER if p in requested]

    n_pairs = len(args.datasets) * len(args.models)
    log.info("=" * 72)
    log.info("  MACRO-MACRO RUNNER  START")
    log.info("  datasets  : %s", args.datasets)
    log.info("  models    : %s", args.models)
    log.info("  phases    : %s  (exec order: %s)", sorted(requested), phases)
    log.info("  pairs     : %d", n_pairs)
    log.info("  dry-run   : %s", args.dry_run)
    log.info("  outputs   : outputs/{dataset}/{model}/phase_{N}/")
    log.info("=" * 72)

    results: list[PairResult] = []
    t_start = time.perf_counter()

    for dataset, model in itertools.product(args.datasets, args.models):
        log.info("━" * 72)
        log.info("  PAIR  dataset=%-12s  model=%s", dataset, model)
        log.info("━" * 72)
        result = run_pair(dataset, model, phases=phases, dry_run=args.dry_run)
        results.append(result)
        ok = sum(1 for p in result.phases if p.success)
        fail = len(result.phases) - ok
        log.info("  Pair done  ok=%d  fail=%d  (%.1f min)", ok, fail, result.elapsed / 60)

    total_elapsed = time.perf_counter() - t_start
    log.info("")
    log.info("Total wall time: %.1fs (%.1f min)", total_elapsed, total_elapsed / 60)
    _print_summary(results)


if __name__ == "__main__":
    main()
