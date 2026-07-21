"""Run every week-5 experiment in sequence.

Quick pass (default, ~15-30 min):   python src/rlc_model/run_all.py
Final report quality (hours):       python src/rlc_model/run_all.py --full
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import subprocess

SCRIPTS = [
    "exp1_circuit_verification.py",
    "exp2_convergence.py",
    "exp3_rate_vs_distance.py",
    "exp4_rate_vs_N.py",
    "exp5_component_analysis.py",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="report-quality settings (1000 realizations)")
    args = parser.parse_args()

    here = pathlib.Path(__file__).resolve().parent
    for script in SCRIPTS:
        cmd = [sys.executable, "-u", str(here / script)]
        if args.full and script.startswith(("exp3", "exp4", "exp5")):
            cmd += ["--realizations", "500"]
        print(f"\n=== {' '.join(cmd)} ===")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
