"""Re-run the chaos matchups after re-tuning chaos.py's eval weights.

The original exp01 results for any matchup involving the chaos universe are
invalid (chaos's eval was broken — see chaos.py history). This script:

  1. Moves the chaos-contaminated JSON checkpoints out of the live results
     directory and into a backup, so the resumable runner will re-play them.
  2. Re-invokes ``python -m experiments.run_exp01``. The runner skips any
     matchup whose JSON checkpoint still exists, so only the moved-out
     matchups are re-played.

Contaminated matchups:
  - round_robin/: every JSON whose filename contains "chaos" (6 files, since
    chaos is paired against each of the other 6 single universes).
  - multiverse/: only the ``shared-7`` and ``indep-7`` variants — those use
    SINGLE_NAMES (which includes chaos) per experiments/run_exp01.py. The
    ``shared-5`` and ``indep-5`` variants use FIVE_UNIVERSE_NAMES (no chaos)
    and stay put.

Run from the repo root with the project's uv environment, e.g.::

    uv run python experiments/rerun_chaos.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "experiments" / "results" / "exp01"
RR_DIR = RESULTS_ROOT / "round_robin"
MV_DIR = RESULTS_ROOT / "multiverse"
BACKUP_ROOT = REPO_ROOT / "experiments" / "results" / "exp01_broken_chaos_backup"

# Multiverse labels whose universe list includes "chaos". Verified against
# SINGLE_NAMES vs FIVE_UNIVERSE_NAMES in experiments/run_exp01.py.
CONTAMINATED_MV_LABELS = ("shared-7", "indep-7")


def collect_chaos_round_robin() -> list[Path]:
    if not RR_DIR.is_dir():
        return []
    return sorted(p for p in RR_DIR.glob("*.json") if "chaos" in p.name)


def collect_chaos_multiverse() -> list[Path]:
    if not MV_DIR.is_dir():
        return []
    return sorted(
        p
        for p in MV_DIR.glob("*.json")
        if any(p.name.startswith(label + "__vs__") for label in CONTAMINATED_MV_LABELS)
    )


def print_plan(rr_files: list[Path], mv_files: list[Path]) -> None:
    print(f"Backup destination: {BACKUP_ROOT}")
    print()
    print(f"Round-robin JSONs to move ({len(rr_files)}):")
    for p in rr_files:
        print(f"  {p.name}")
    if not rr_files:
        print("  (none)")
    print()
    print(f"Multiverse JSONs to move ({len(mv_files)}):")
    for p in mv_files:
        print(f"  {p.name}")
    if not mv_files:
        print("  (none)")
    print()
    print(
        f"Then will invoke: {sys.executable} -m experiments.run_exp01\n"
        "  (cwd = repo root; the runner is resumable and will only re-play\n"
        "   the matchups whose JSONs we just moved out.)"
    )
    print()


def move_to_backup(files: list[Path]) -> None:
    if not files:
        return
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    for src in files:
        target = BACKUP_ROOT / src.name
        if target.exists():
            print(f"  SKIP (already in backup): {src.name}")
            continue
        shutil.move(str(src), str(target))
        print(f"  moved: {src.name}")


def main() -> int:
    rr_files = collect_chaos_round_robin()
    mv_files = collect_chaos_multiverse()

    print_plan(rr_files, mv_files)

    if not rr_files and not mv_files:
        print("Nothing contaminated to move. Exiting without re-running the experiment.")
        return 0

    ans = input("Type 'yes' to proceed: ").strip()
    if ans.lower() != "yes":
        print("Aborted — no files moved.")
        return 1

    print()
    print("Moving files...")
    move_to_backup(rr_files)
    move_to_backup(mv_files)
    print()
    print("Invoking experiment runner...")
    return subprocess.call(
        [sys.executable, "-m", "experiments.run_exp01"],
        cwd=str(REPO_ROOT),
    )


if __name__ == "__main__":
    sys.exit(main())
