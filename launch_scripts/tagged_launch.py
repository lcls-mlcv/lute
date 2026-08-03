"""Tag-driven multi-run workflow submission.

Entry point used by `launch_maestro.py::main()` when `--tag` is passed
instead of (or alongside) `-r`/`--run`. Resolves every run carrying the
given eLog tag, splits the requested workflow DAG into a run-dependent
subgraph and a non-run-dependent subgraph (see `dag_partition.py`), submits
the run-dependent subgraph once per resolved run, and - once every one of
those has completed successfully - submits the non-run-dependent subgraph
exactly once, with `TAG`/`EXPERIMENT` exported into its environment for
that subgraph's own Tasks to use (e.g. `MergeCCTBXXFELParameters.phil_
parameters.tag`, see `lute/io/models/sfx_merge.py`).

Deliberately re-invokes the real, unmodified `launch_slurm` binary as a
subprocess for every stage/run rather than calling `load_lute_dag`/
`run_workflow` repeatedly in-process: the DAG execution engine is a compiled
extension (`maestro._maestro._maestro`) whose safety under repeated
in-process invocation within one Python process was not verified, whereas
re-running the already-tested single-run entry point as a fresh OS process
per invocation carries no such risk.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

logger = logging.getLogger(__name__)


class TaggedLaunchError(Exception):
    """Raised when a --tag workflow submission cannot proceed."""


def _common_args(args: argparse.Namespace, extra_args: List[str]) -> List[str]:
    """Rebuild the CLI args every child `launch_slurm` invocation should get,
    minus -W/-r/--tag (each stage supplies its own -W; -r varies per run for
    the dependent stage and is fixed to a representative run for the
    non-dependent stage; --tag is intentionally NOT forwarded, so the child
    process takes the normal single-run code path)."""
    common: List[str] = ["-c", args.config]
    if args.experiment:
        common += ["-e", args.experiment]
    if args.type:
        common += ["--type", args.type]
    if args.debug:
        common += ["-d"]
    if getattr(args, "num_server_threads", None):
        common += ["--num_server_threads", str(args.num_server_threads)]
    if getattr(args, "unbuffered", False):
        common += ["--unbuffered"]
    common += extra_args
    return common


def _submit_stage(launch_slurm_bin: str, dag_path: str, run: int, common: List[str]) -> None:
    cmd = [launch_slurm_bin, "-W", dag_path, "-r", str(run), *common]
    logger.info("Submitting tagged stage for run %s: %s", run, " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_tagged_workflow(
    args: argparse.Namespace,
    extra_args: List[str],
    bin_dir: str,
    lute_location: str,
) -> None:
    experiment: Optional[str] = os.getenv("EXPERIMENT") or args.experiment
    if not experiment:
        raise TaggedLaunchError(
            "--tag requires -e/--experiment (or the EXPERIMENT env var) to "
            "resolve which runs carry the tag."
        )

    from lute.io.elog import get_elog_runs_by_tag

    runs: List[int] = sorted(get_elog_runs_by_tag(experiment, args.tag))
    if not runs:
        raise TaggedLaunchError(
            f"No runs found for tag '{args.tag}' in experiment '{experiment}'. "
            "(Note: this is indistinguishable today from an eLog auth/network "
            "failure - get_elog_runs_by_tag collapses both cases to an empty "
            "list; check your Kerberos ticket / the tag spelling.)"
        )
    logger.info("Tag '%s' resolved to runs: %s", args.tag, runs)

    from launch_scripts.dag_partition import partition_workflow_yaml

    dep_yaml, non_dep_yaml = partition_workflow_yaml(args.workflow_defn)

    tmp_dir = tempfile.mkdtemp(prefix="lute_tag_", dir=os.path.dirname(os.path.abspath(args.workflow_defn)))
    logger.info("Writing partitioned DAG(s) for this --tag submission to %s", tmp_dir)

    launch_slurm_bin = f"{bin_dir}/launch_slurm"
    common = _common_args(args, extra_args)

    if dep_yaml is not None:
        dep_path = os.path.join(tmp_dir, "run_dependent.dag")
        with open(dep_path, "w") as f:
            f.write(dep_yaml)

        max_workers = args.max_concurrent_runs or len(runs)
        logger.info(
            "Submitting run-dependent stage for %d run(s), up to %d concurrently",
            len(runs),
            max_workers,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                run: pool.submit(_submit_stage, launch_slurm_bin, dep_path, run, common)
                for run in runs
            }
            failed: List[int] = []
            for run, future in futures.items():
                try:
                    future.result()
                except subprocess.CalledProcessError as e:
                    logger.error("Run-dependent stage failed for run %s: %s", run, e)
                    failed.append(run)
        if failed:
            raise TaggedLaunchError(
                f"Run-dependent stage failed for runs {sorted(failed)} - not "
                "submitting the non-run-dependent stage on top of incomplete data."
            )

    if non_dep_yaml is not None:
        non_dep_path = os.path.join(tmp_dir, "non_run_dependent.dag")
        with open(non_dep_path, "w") as f:
            f.write(non_dep_yaml)

        os.environ["TAG"] = args.tag
        os.environ["EXPERIMENT"] = experiment
        representative_run = runs[0]
        logger.info(
            "Submitting non-run-dependent stage once (TAG=%s, representative run=%s)",
            args.tag,
            representative_run,
        )
        _submit_stage(launch_slurm_bin, non_dep_path, representative_run, common)

    logger.info("--tag submission for '%s' complete.", args.tag)
