#!/usr/bin/env python3
"""Run the three data-generation stages over one row window, in order."""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

STAGES = (
    ("precompute_esm", "ESM-2 embeddings"),
    ("tm_data", "TM-align"),
    ("fgw_data", "local FGW scores"),
)


def log(msg):
    print(msg, flush=True)


def run_stage(script, description, args):
    command = [
        sys.executable,
        os.path.join(HERE, f"{script}.py"),
        "--start-row",
        str(args.start_row),
        "--end-row",
        "none" if args.end_row is None else str(args.end_row),
    ]
    if args.no_resume:
        command.append("--no-resume")
    if args.output_suffix and script in ("tm_data", "fgw_data"):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            script, os.path.join(HERE, f"{script}.py")
        )
        base = None
        with open(os.path.join(HERE, f"{script}.py")) as handle:
            for line in handle:
                if line.startswith("OUTPUT_CSV"):
                    base = line.split('"')[1]
                    break
        if base:
            root, ext = os.path.splitext(base)
            command += ["--output", f"{root}{args.output_suffix}{ext}"]

    log("")
    log("=" * 66)
    log(f"  {script}  ({description})")
    log("=" * 66)

    if args.dry_run:
        log("  would run: " + " ".join(command))
        return 0

    return subprocess.call(command)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument(
        "--end-row",
        type=lambda v: None if v.lower() in ("none", "eof", "") else int(v),
        default=None,
        help="exclusive; omit or 'none' for end of file",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="reprocess entries already present in each output",
    )
    parser.add_argument(
        "--stages",
        default="all",
        help="comma-separated subset, e.g. 'tm_data,fgw_data' (default: all)",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="suffix each stage's CSV, e.g. '.shard3'. Use this for array jobs: "
             "concurrent appends to one CSV interleave rows. Concatenate after.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    wanted = (
        [name for name, _ in STAGES]
        if args.stages == "all"
        else [s.strip() for s in args.stages.split(",") if s.strip()]
    )
    unknown = set(wanted) - {name for name, _ in STAGES}
    if unknown:
        parser.error(f"unknown stage(s): {sorted(unknown)}")

    end = "EOF" if args.end_row is None else args.end_row
    log(f"Parquet row window: [{args.start_row}, {end})")
    log(f"Stages: {', '.join(wanted)}")
    log(f"Resume: {'off' if args.no_resume else 'on'}")

    for script, description in STAGES:
        if script not in wanted:
            continue
        code = run_stage(script, description, args)
        if code != 0:
            log("")
            log(f"STOPPING: {script} exited with code {code}")
            log("Later stages consume its output, so they were not run.")
            return code

    log("")
    log("=" * 66)
    log("  pipeline complete")
    log("=" * 66)
    log("Next: python data_scripts/audit_fgw_data.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
