"""Resolve an eLog tag to its run list.

Thin CLI wrapper around `lute.io.elog.get_elog_runs_by_tag`, for submission
scripts that need to know a --tag's run list before/without submitting
anything - e.g. to pre-create per-run output directories, which LUTE does
not do itself (see `references/cctbx-sfx-workflow.md` §4 in the ask-lute
skill). `launch_scripts/tagged_launch.py` resolves the same tag internally
when actually submitting; this is the same lookup exposed standalone.
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print every run number tagged with a given eLog tag."
    )
    parser.add_argument("-e", "--experiment", type=str, required=True)
    parser.add_argument("-t", "--tag", type=str, required=True)
    parser.add_argument(
        "--separator",
        type=str,
        default=" ",
        help="Separator between printed run numbers (default: a single space).",
    )
    args = parser.parse_args()

    from lute.io.elog import get_elog_runs_by_tag

    runs = sorted(get_elog_runs_by_tag(args.experiment, args.tag))
    if not runs:
        print(
            f"No runs found for tag '{args.tag}' in experiment "
            f"'{args.experiment}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(args.separator.join(str(r) for r in runs))


if __name__ == "__main__":
    main()
