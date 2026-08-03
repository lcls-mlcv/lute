"""Script to submit a single **managed** Task via SLURM."""

__author__ = "Gabriel Dorlhiac"

import argparse
import datetime
import os
import re
import secrets
import subprocess
import sys
from typing import List, Optional, Tuple


def get_parser() -> argparse.ArgumentParser:
    """Setup the submit_slurm command-line argument parser.

    Returns:
        parser (argparse.ArgumentParser): The command-line parser.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    # Immediately pop the group - takes out the help argument. We add to the group later
    optional_args: argparse._ArgumentGroup = parser._action_groups.pop()

    # Required arguments
    required_args: argparse._ArgumentGroup = parser.add_argument_group(
        "required arguments"
    )

    required_args.add_argument(
        "-c", "--config", type=str, help="Absolute path to config YAML file."
    )
    required_args.add_argument(
        "-t", "--taskname", help="Name of the LUTE **managed** Task to run."
    )

    # Arguments required for when running from command-line
    non_arp_required_args: argparse._ArgumentGroup = parser.add_argument_group(
        "required arguments when running without the ARP"
    )
    non_arp_required_args.add_argument(
        "-e",
        "--experiment",
        type=str,
        help="Provide an experiment if not running with ARP.",
        required=False,
    )
    non_arp_required_args.add_argument(
        "-r",
        "--run",
        type=str,
        help="Provide a run number if not running with ARP.",
        required=False,
    )

    # Optional Arguments
    optional_args.add_argument(
        "--tag",
        type=str,
        default=None,
        required=False,
        help=(
            "Submit against every run tagged with this eLog tag. If -r/--run "
            "is NOT also given, submits once per resolved run (for a "
            "run-dependent Task, e.g. indexing). If -r/--run IS also given, "
            "submits once as usual but also exports TAG into the "
            "environment (for a non-run-dependent Task, e.g. merging, whose "
            "own parameter validator is responsible for resolving the tag "
            "into the runs it should aggregate)."
        ),
    )

    optional_args.add_argument(
        "-d", "--debug", help="Run in debug mode.", action="store_true"
    )

    optional_args.add_argument(
        "-K", "--KERB", help="Kerberos cache file variable. Should NOT be set manually!"
    )

    parser.add_argument(
        "--psana2", help="Use the psana2 base environment.", action="store_true"
    )
    parser.add_argument(
        "--psana1",
        help=(
            "Use the psana1 base environment. NOTE: We allow both --psana1 and --psana2 "
            "to be passed. This is because the LUTE infrastructure now defaults to "
            "psana2 in the event that it cannot auto-determine the type of experiment. "
            "When both --psana2 and --psana1 are passed, **psana1 TAKES PRECEDENCE**. "
            "This allows forcing the use of the psana1 environment without changing "
            "default behaviour elsewhere in the infrastructure. "
        ),
        action="store_true",
    )

    parser._action_groups.append(optional_args)
    return parser


def parse_arguments(
    parser: argparse.ArgumentParser,
) -> Tuple[argparse.Namespace, List[str]]:
    """Check the command-line parsing.

    To avoid any issues with other parsing, it enforces that SLURM arguments are
    of the format: --long_arg=value. The one exception to this rule is if the
    --exclusive flag is passed, which does not take an argument value.

    Args:
        parser (argparse.ArgumentParser): The command-line parser.

    Returns:
        lute_args (argparse.Namespace): The main LUTE argument namespace.

        slurm_args (List[str]): A list of additional arguments. Assumed to be
            for SLURM.
    """
    args: argparse.Namespace
    extra_args: List[str]
    args, extra_args = parser.parse_known_args()

    # Double check that SLURM args were passed as --long_form=value
    # Not sure if numbers allowed. Nearly certain that no... but leave it in case...
    # Make an exception for --exclusive
    valid_slurm_form: re.Pattern = re.compile(r"^--[A-Za-z0-9][A-Za-z0-9_-]*=.+")

    invalid_args: List[str] = [
        arg
        for arg in extra_args
        if not valid_slurm_form.match(arg) and arg != "--exclusive"
    ]
    if invalid_args:
        parser.error(
            "SLURM arguments should be passed in the form: `--long_form=value`.\n"
            "E.g. --nodes=2 instead of -N 2."
        )

    return args, extra_args


def prepare_environment_variables(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> str:
    """Setup the LUTE environment variables.

    Args:
        parser (argparse.ArgumentParser): The command-line parser. This is used
            only to throw errors as needed.

        args (argparse.Namespace): Validated LUTE command-line arguments.

    Returns:
        bin_subdir (str): The sub-directory for LUTE executables.
    """
    experiment: Optional[str] = os.getenv("EXPERIMENT")
    run_num: Optional[str] = os.getenv("RUN_NUM")

    if experiment is None:
        experiment = args.experiment
        if experiment is None:
            parser.error(
                "The `EXPERIMENT` environment variable was not defined, so you "
                "have likely submitted this from the command-line. Please pass "
                "-e <EXPERIMENT> to provide an experiment!"
            )

    if run_num is None:
        run_num = args.run
        if run_num is None:
            parser.error(
                "The `RUN_NUM` environment variable was not defined, so you "
                "have likely submitted this from the command-line. Please pass "
                "-r <RUN_NUM> to provide a run number!"
            )

    # NOTE: When submitted from the ARP, the RUN_NUM env var is actually: RUN_DATETIME
    # So we clip off the date time portion
    run_num = run_num.split("_")[0]

    # Re-export all required environment variables
    os.environ["RUN"] = run_num
    os.environ["RUN_NUM"] = run_num  # Both forms of RUN are used by different things...
    os.environ["EXPERIMENT"] = experiment

    if args.KERB:
        os.environ["KRB5CCNAME"] = args.KERB

    # Check TCP vs Unix socket. Default is TCP if unset
    if (env_tcp := os.getenv("LUTE_USE_TCP")) is not None and env_tcp == "0":
        print("Using Unix sockets")
        # Emulating bash $RANDOM - could make this better now... but should be fine
        os.environ["LUTE_SOCKET"] = f"/tmp/lute_{secrets.randbelow(32768)}.sock"
        if "LUTE_USE_TCP" in os.environ:
            os.environ.pop("LUTE_USE_TCP")
    else:
        os.environ["LUTE_USE_TCP"] = "1"

    script_dir: str = os.path.dirname(os.path.realpath(sys.argv[0]))
    lute_location: str = os.path.abspath(f"{script_dir}/..")

    bin_subdir: str = script_dir.split("/")[-1]

    os.environ["LUTE_PATH"] = lute_location

    return bin_subdir


def fill_in_batch_script(
    args: argparse.Namespace, slurm_args: List[str], bin_subdir: str
) -> str:
    """Prepare a temporary batch script.

    Args:
        args (argparse.Namespace): Validated LUTE command-line arguments.

        slurm_args (List[str]): Validated SLURM command-line arguments.

        bin_subdir (str): The sub-directory for LUTE executables.

    Returns:
        batch_script (str): A completed batch script which can be run with
            sbatch once written to a file.
    """
    experiment: Optional[str] = os.getenv("EXPERIMENT")
    run_num: Optional[str] = os.getenv("RUN_NUM")

    if experiment is None or run_num is None:
        raise RuntimeError("Experiment and/or run number are not defined!")

    batch_script: str = "#!/bin/bash\n"

    # Setup log file name - task_exp_run_time_job, the %J will be filled in by SLURM
    # This will be used for both stdout and stderr log files
    curr_time: str = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    log_file: str = (
        f"{args.taskname}_{experiment}_r{int(run_num):04d}_{curr_time}_%J.out"
    )

    for slurm_arg in slurm_args:
        batch_script += f"#SBATCH {slurm_arg}\n"

    batch_script += f"#SBATCH --output={log_file}\n"
    batch_script += f"#SBATCH --error={log_file}\n"

    if args.psana1:
        print("Using a Psana1 base environment")
        batch_script += (
            "source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh\n"
        )
    elif args.psana2:
        print("Using a Psana2 base environment")
        batch_script += (
            "source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh\n"
        )
    else:
        print("Using a Psana1 base environment")
        batch_script += (
            "source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh\n"
        )

    lute_location: Optional[str] = os.getenv("LUTE_PATH")
    if lute_location is None:
        raise RuntimeError("LUTE path env var is empty! Check the launch scripts!")

    full_bindir: str = f"{lute_location}/{bin_subdir}"
    lute_executable: str
    if "launch_scripts" in bin_subdir:
        lute_executable = f"{lute_location}/run_task.py"
    else:
        lute_executable = f"{full_bindir}/run_task"

        py_ver: str = f"{sys.version_info.major}.{sys.version_info.minor}"
        # Need to redo the LUTE_PATH env var here... not great...
        # Note that in this case, the LUTE_PATH included the /install/ dir
        os.environ["LUTE_PATH"] = f"{lute_location}/lib/python{py_ver}/site-packages"

        batch_script += f"source {full_bindir}/activate_installation\n\n"

    cmd: str
    if args.debug:
        cmd = f"python -B {lute_executable} -c {args.config} -t {args.taskname}"
    else:
        cmd = f"python -OB {lute_executable} -c {args.config} -t {args.taskname}"

    batch_script += f"{cmd}\n"

    return batch_script


def _submit_batch_script(taskname: str, batch_script: str, debug: bool, slurm_args: List[str]) -> None:
    """Write a batch script to a temp file, sbatch it, and clean up.

    Extracted from `main()` so it can be called once (plain single-run
    submission) or in a loop (one call per resolved run in a --tag,
    non-run-dependent submission), unchanged either way.
    """
    temp_filename: str = f"submit_{taskname}_{secrets.token_hex(4)}.sh"
    with open(temp_filename, "w") as f:
        f.write(batch_script)

    print(f"Submitting task {taskname}")
    if debug:
        print(f"Running {taskname} with SLURM arguments: {slurm_args}")
        print(f"Full script:\n{batch_script}")

    try:
        result: subprocess.CompletedProcess = subprocess.run(
            ["sbatch", temp_filename], check=True
        )
        print(result.stdout)
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e}")
        sys.exit(1)
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


def main() -> None:
    """Parse LUTE and SLURM command-line arguments and submit a batch job."""
    parser: argparse.ArgumentParser = get_parser()

    args, slurm_args = parse_arguments(parser)

    if args.tag and not args.run:
        # Run-dependent case: no explicit -r, resolve every run tagged
        # `args.tag` and submit once per run, reusing the existing
        # single-run functions completely unmodified in a loop.
        experiment: Optional[str] = os.getenv("EXPERIMENT") or args.experiment
        if experiment is None:
            parser.error(
                "--tag without -r/--run requires -e/--experiment (or the "
                "EXPERIMENT env var) to resolve which runs carry the tag."
            )

        from lute.io.elog import get_elog_runs_by_tag

        runs: List[int] = sorted(get_elog_runs_by_tag(experiment, args.tag))
        if not runs:
            parser.error(
                f"No runs found for tag '{args.tag}' in experiment "
                f"'{experiment}'."
            )

        os.environ["TAG"] = args.tag
        for run in runs:
            os.environ["RUN_NUM"] = str(run)
            bin_subdir: str = prepare_environment_variables(parser=parser, args=args)
            batch_script: str = fill_in_batch_script(
                args=args, slurm_args=slurm_args, bin_subdir=bin_subdir
            )
            _submit_batch_script(args.taskname, batch_script, args.debug, slurm_args)
        return

    if args.tag:
        # Non-run-dependent case: -r given alongside --tag. Submit once, as
        # usual, but export TAG so the Task's own parameter validator can
        # resolve which runs to aggregate (e.g. MergeCCTBXXFELParameters).
        os.environ["TAG"] = args.tag

    bin_subdir = prepare_environment_variables(parser=parser, args=args)

    batch_script = fill_in_batch_script(
        args=args, slurm_args=slurm_args, bin_subdir=bin_subdir
    )

    _submit_batch_script(args.taskname, batch_script, args.debug, slurm_args)


if __name__ == "__main__":
    main()
