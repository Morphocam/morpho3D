"""Command-line companion to the GUI.

    python tools_cli.py doctor                 # what is installed, what is not
    python tools_cli.py setup <backend|all>    # provision backend env(s)
    python tools_cli.py mark <backend>         # record an already-working backend
    python tools_cli.py run <backend> <img>... # generate without opening the GUI
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from morpho3d import config                                  # noqa: E402
from morpho3d.backends import ALL_BACKENDS, JobSpec, Mode, registry  # noqa: E402
from morpho3d.setup.provision import provision               # noqa: E402


def cmd_doctor(_args) -> int:
    print(f"conda : {config.CONDA.root if config.CONDA else 'NOT FOUND'}")
    print(f"GPU   : {config.gpu_summary()}")
    print(f"MSVC  : {config.msvc_summary()}")
    print(f"cache : {config.HF_HOME}")
    print()
    width = max(len(b.label) for b in ALL_BACKENDS)
    for b in ALL_BACKENDS:
        avail = b.availability()
        modes = "+".join("img" if m is Mode.IMAGE else "mv" for m in b.modes)
        state = "READY" if avail.ready else avail.reason
        print(f"  {b.label:<{width}}  [{modes:>7}]  {state}")
    return 0


def cmd_setup(args) -> int:
    targets = ALL_BACKENDS if args.backend == "all" else [registry.get(args.backend)]
    failed = []
    for b in targets:
        if b.availability().ready:
            print(f"-- {b.label}: already set up")
            continue
        try:
            provision(b, print)
        except Exception as exc:  # noqa: BLE001
            print(f"!! {b.label} failed: {exc}")
            failed.append(b.label)
    if failed:
        print(f"\nFailed: {', '.join(failed)}")
        return 1
    return 0


def cmd_mark(args) -> int:
    b = registry.get(args.backend)
    b.mark_provisioned()
    print(f"marked {b.label} as provisioned ({b._provision_marker()})")
    return 0


def cmd_run(args) -> int:
    b = registry.get(args.backend)
    images = [Path(p).resolve() for p in args.images]
    missing = [p for p in images if not p.exists()]
    if missing:
        print(f"missing input: {missing[0]}")
        return 1

    mode = Mode.MULTIVIEW if len(images) > 1 else Mode.IMAGE
    if not b.supports(mode):
        print(f"{b.label} does not support {mode.label}")
        return 1

    avail = b.availability()
    if not avail.ready:
        print(f"{b.label} is not ready: {avail.reason}")
        return 1

    out = config.OUTPUT_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{b.key}"
    job = JobSpec(mode=mode, images=images, out_dir=out, seed=args.seed,
                  steps=args.steps, remove_background=not args.keep_background)
    result = b.run(job, print)
    print(f"\nmesh: {result.mesh_path}")
    print(f"time: {result.elapsed_s:.1f}s")
    if result.notes:
        print(f"note: {result.notes}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor").set_defaults(func=cmd_doctor)

    p_setup = sub.add_parser("setup")
    p_setup.add_argument("backend", choices=[b.key for b in ALL_BACKENDS] + ["all"])
    p_setup.set_defaults(func=cmd_setup)

    p_mark = sub.add_parser("mark")
    p_mark.add_argument("backend", choices=[b.key for b in ALL_BACKENDS])
    p_mark.set_defaults(func=cmd_mark)

    p_run = sub.add_parser("run")
    p_run.add_argument("backend", choices=[b.key for b in ALL_BACKENDS])
    p_run.add_argument("images", nargs="+")
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--steps", type=int, default=30)
    p_run.add_argument("--keep-background", action="store_true")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
