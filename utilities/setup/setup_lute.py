"""Script to setup LUTE and workflow definitions."""

__author__ = "Gabriel Dorlhiac"

import argparse
import logging
import os
import requests
import shutil
import subprocess
import sys
from typing import List, Dict, Any, Optional

from krtc import KerberosTicket  # type: ignore

DEFAULT_CONFIG = {
    "SmallDataProducer": {
        "nodes": 4,
        "ntasks_per_node": 50,
    },
    "SmallDataProducer2": {
        "nodes": 4,
        "ntasks_per_node": 50,
    },
    "BayFAIOptimizer": {
        "nodes": 1,
        "ntasks_per_node": 120,
    },
    "BayFAIOptimizer2": {
        "nodes": 1,
        "ntasks_per_node": 120,
    },
}

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


def _run_subprocess_log(cmd: List[str], env: Optional[Dict[str, str]] = None) -> None:
    """Run a subprocess with logging."""
    global logger

    out: str
    err: str
    out, err = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        env=env,
    ).communicate()
    if out:
        logger.info(out)
    if err:
        logger.info(err)


def touch(full_path: str) -> None:
    """Touch a file.

    Args:
        full_path (str): The full path to the file to touch.
    """
    cmd: List[str] = ["touch", full_path]
    _run_subprocess_log(cmd)


def database_setup(full_path: str) -> None:
    """Touch a file.

    Args:
        full_path (str): The full path to the file to touch.
    """
    if os.path.exists(full_path):
        logger.error(
            f"LUTE database already exists at: {full_path}. Will not overwrite, exiting."
        )
        sys.exit(-1)
    touch(full_path)
    os.chmod(full_path, 0o664)


def git_clone(repo: str, location: str, tag: str) -> None:
    """Clone a git repository.

    Will not overwrite a directory of there is already a folder at the specified
    location.
    Args:
        repo (str): Name of the repository to clone. Should be specified as:
            "<user_or_organization>/<repository_name>"

        location (str): Path to the location to clone to.
    """
    global logger

    repo_only: str = repo.split("/")[1]
    if os.path.exists(f"{location}/{repo_only}"):
        logger.debug(
            f"Repository {repo} already exists at {location}. Will not overwrite."
        )
        return
    cmd: List[str] = [
        "git",
        "clone",
        f"https://github.com/{repo}.git",
        location,
    ]
    _run_subprocess_log(cmd)

    cwd: str = os.getcwd()
    os.chdir(location)
    cmd = ["git", "checkout", tag]
    _run_subprocess_log(cmd)
    os.chdir(cwd)


def run_build_script(lute_path: str) -> None:
    """Run the LUTE build script.

    Args:
        lute_path (str): The path to the LUTE installation to build.
    """

    cwd: str = os.getcwd()
    os.chdir(lute_path)
    cmd: List[str] = ["./build.sh"]
    _run_subprocess_log(cmd)
    os.chdir(cwd)


def pip_install(src_dir: str, install_dir: Optional[str] = None) -> None:
    """Install from a source directory to an optionally specified directory.

    If no prefix (`install_dir`) is provided, this will install to the current
    packages directory determined by, e.g., conda environment, etc.

    If a prefix is provided this command will also create the directory (or
    multiple directories) if necessary.

    Args:
        src_dir (str): Directory with the source code and setup.py.

        install_dir (Optional[str]): Optionally provide a directory to install
            install to.
    """
    cmd: List[str] = [
        "pip",
        "install",
        "--no-deps",
        src_dir,
        f'--prefix="{install_dir}"',
    ]
    logging.info(f"Attempting to install from: {src_dir} to: {install_dir}")
    env: Dict[str, str] = os.environ.copy()
    env["PATH"] = (
        f"/sdf/group/lcls/ds/ana/sw/conda1/inst/envs/ana-4.0.63-py3/bin:{env['PATH']}"
    )
    if install_dir is not None:
        cmd.append(f"--prefix={install_dir}")
        if not os.path.exists(install_dir):
            mkdir_cmd: List[str] = ["mkdir", "-p", install_dir]
            _run_subprocess_log(mkdir_cmd)
    _run_subprocess_log(cmd, env)


def inplace_sed(in_file: str, pattern: str) -> None:
    """Perform an in-place operation on a file using sed.

    Args:
        in_file (str): Path to the file to perform the substitution on.

        pattern (str): Operation. E.g. substitute with "s/old_text/new_text/g"
    """
    cmd: List[str] = ["sed", "-i", pattern, in_file]
    _run_subprocess_log(cmd)


def modify_permissions(lute_path: str):
    """Recursively set permissions for a LUTE installation."""
    os.chmod(lute_path, 0o765)
    for root, dirs, files in os.walk(lute_path):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o765)

        for f in files:
            os.chmod(os.path.join(root, f), 0o765)

def update_dag_params(dag_path: str, partition: str, account: str, nodes: int, ntasks_per_node: int) -> None:
    """Update slurm_params in a DAG file in place.

    For tasks listed in DEFAULT_CONFIG, use the task-specific nodes/ntasks_per_node.
    For all other tasks, use the user-provided values.

    Args:
        dag_path (str): Path to the DAG file.

        partition (str): SLURM partition.

        account (str): SLURM account.

        nodes (int): Number of nodes (used for tasks not in DEFAULT_CONFIG).

        ntasks_per_node (int): Tasks per node (used for tasks not in DEFAULT_CONFIG).
    """
    with open(dag_path, "r") as f:
        lines: List[str] = f.readlines()

    result: List[str] = []
    current_task: Optional[str] = None

    for line in lines:
        stripped = line.lstrip()
        # Track the most recent task_name we've seen
        if stripped.startswith("task_name:"):
            # Extract task name (handles both quoted and unquoted)
            task_value = stripped.split(":", 1)[1].strip().strip("\"'")
            current_task = task_value

        if stripped.startswith("slurm_params:"):
            indent = line[: len(line) - len(stripped)]
            if current_task and current_task in DEFAULT_CONFIG:
                cfg = DEFAULT_CONFIG[current_task]
                params = (
                    f"--account={account} --partition={partition} "
                    f"--ntasks-per-node={cfg['ntasks_per_node']} "
                    f"--nodes={cfg['nodes']} --exclusive"
                )
            else:
                params = (
                    f"--account={account} --partition={partition} "
                    f"--ntasks-per-node={ntasks_per_node} "
                    f"--nodes={nodes}"
                )
            result.append(f"{indent}slurm_params: '{params}'\n")
        else:
            result.append(line)

    with open(dag_path, "w") as f:
        f.writelines(result)

    logger.info(f"Updated slurm_params in DAG file: {dag_path}")

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="setup_lute",
        description="Setup LUTE work space and eLog workflows for an experiment.",
        epilog="Refer to https://github.com/slac-lcls/lute for more information.",
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Turn on verbose logging."
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        help="Experiment to perform setup for.",
        required=True,
    )
    parser.add_argument(
        "-f",
        "--fresh_install",
        help=(
            "Install a new version of LUTE in the experiment folder. This allows "
            "for local modifications of code. Otherwise, the central installation "
            "will be used which cannot be modified."
        ),
        action="store_true",
    )
    parser.add_argument(
        "--directory",
        type=str,
        help=(
            "Subdirectory name within the experiment results folder to use for "
            "LUTE output and LUTE fresh install if specified. If not specified, "
            "will use results folder directly."
        ),
        default="",
    )
    parser.add_argument(
        "--test", help="Use test Airflow instance.", action="store_true"
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        help=(
            "Version of LUTE to use. Corresponds to release tag or `dev`. "
            "Defaults to `dev-mfx`."
        ),
        default="dev-mfx",
    )
    parser.add_argument(
        "-W",
        "--workflow",
        type=str,
        help=("Which analysis workflow to run. Defaults to smd_summaries."),
        default="smd",
    )
    args: argparse.Namespace
    extra_args: List[str]  # May have additional SLURM arguments
    args, extra_args = parser.parse_known_args()

    hutch: str = args.experiment[:3]

    results_dir: str = f"/sdf/data/lcls/ds/{hutch}/{args.experiment}/results"
    if args.directory != "":
        results_dir = f"{results_dir}/{args.directory}"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, mode=0o777)
            os.chmod(results_dir, 0o777)
    lute_path: str
    arp_executable: str
    launch_executable: str
    std_hutch_config: str
    std_test_config: str
    if args.fresh_install:
        lute_path = f"{results_dir}/lute"
        git_clone("lcls-mlcv/lute", lute_path, args.version)
        run_build_script(lute_path)
        modify_permissions(lute_path)
        arp_executable = f"{lute_path}/install/bin/submit_launch_slurm.sh"
        launch_executable = f"{lute_path}/install/bin/launch_slurm"
        py_ver: str = f"python{sys.version_info.major}.{sys.version_info.minor}"
        std_hutch_config = (
            f"{lute_path}/install/lib/{py_ver}/site-packages/config/{hutch}.yaml"
        )
        std_test_config = (
            f"{lute_path}/install/lib/{py_ver}/site-packages/config/test.yaml"
        )
    else:
        lute_path = f"/sdf/group/lcls/ds/tools/lute/{args.version}/lute"
        arp_executable = f"{lute_path}/install/bin/submit_launch_slurm.sh"
        launch_executable = f"{lute_path}/install/bin/launch_slurm"
        std_hutch_config = f"{lute_path}/config/{hutch}.yaml"
        std_test_config = f"{lute_path}/config/test.yaml"

    lute_output_dir: str = f"{results_dir}/lute_output"
    if not os.path.exists(lute_output_dir):
        os.makedirs(lute_output_dir, mode=0o777)
        os.chmod(lute_output_dir, 0o777)
    config_path: str = f"{lute_output_dir}/{hutch}_lute.yaml"
    if os.path.exists(config_path):
        logger.error(
            f"Configuration YAML already exists at: {config_path}. Will not overwrite, exiting."
        )
        sys.exit(-1)
    if not os.path.exists(std_hutch_config):
        shutil.copy(std_test_config, config_path)
    else:
        shutil.copy(std_hutch_config, config_path)
    os.chmod(config_path, 0o666)
    # Substitute the work_dir in LUTE's config to the experiment results folder.
    sed_pattern: str = f's|work_dir:\(.*\)|work_dir: \\"{lute_output_dir}\\"|g'
    inplace_sed(config_path, sed_pattern)

    database_setup(f"{lute_output_dir}/lute.db")  # Setup permissions on database
    full_workflow_path: str = f"{lute_output_dir}/{args.workflow}.dag"
    if not os.path.exists(full_workflow_path):
        included_wf_defn: str = f"{lute_path}/workflows/common/{args.workflow}.dag"
        shutil.copy(included_wf_defn, full_workflow_path)
    os.chmod(full_workflow_path, 0o666)

    param_string: str = f"{launch_executable} -c {config_path} -W {full_workflow_path}"

    if args.debug:
        param_string = f"{param_string} --debug"
    if args.test:
        param_string = f"{param_string} --test"

    extra_args_str: str = " ".join(extra_args)
    # Check for partition, account and ntasks. ntasks has defaults by workflow
    partition: str = "milano"
    if "partition" not in extra_args_str:
        logger.warning(
            f"No queue/partition provided. Defaulting to {partition}. Any key to continue. "
            "Ctrl-C to exit."
        )
        try:
            _: str = input()
            extra_args_str = f"{extra_args_str} --partition={partition}"
        except KeyboardInterrupt:
            logger.info("Exiting.")
            sys.exit(0)

    account: str = f"lcls:{args.experiment}"
    if "account" not in extra_args_str:
        logger.warning(
            f"No account provided. Defaulting to {account}. Any key to continue. "
            "Ctrl-C to exit."
        )
        try:
            _ = input()
            extra_args_str = f"{extra_args_str} --account={account}"
        except KeyboardInterrupt:
            logger.info("Exiting.")
            sys.exit(0)

    nodes: int = 1
    if "nodes" not in extra_args_str:
        logger.warning(
            f"No nodes provided. Defaulting to {nodes}. Any key to continue. "
            "Ctrl-C to exit."
        )
        try:
            _ = input()
            extra_args_str = f"{extra_args_str} --nodes={nodes}"
        except KeyboardInterrupt:
            logger.info("Exiting.")
            sys.exit(0)

    ntasks_per_node: int = 1
    if "ntasks-per-node" not in extra_args_str:
        logger.warning(
            f"No ntasks-per-node provided. Defaulting to {ntasks_per_node}. Any key to continue. "
            "Ctrl-C to exit."
        )
        try:
            _ = input()
            extra_args_str = f"{extra_args_str} --ntasks-per-node={ntasks_per_node}"
        except KeyboardInterrupt:
            logger.info("Exiting.")
            sys.exit(0)
    
    if "exclusive" not in extra_args_str:
        logger.warning(
            f"No exclusivity provided. Defaulting to no exclusive access. Any key to continue. "
            "Ctrl-C to exit."
        )
        try:
            _ = input()
        except KeyboardInterrupt:
            logger.info("Exiting.")
            sys.exit(0)

    param_string = f"{param_string} {extra_args_str}"

    # Update the DAG file in place with collected SLURM params
    update_dag_params(full_workflow_path, partition, account, nodes, ntasks_per_node)

    main_workflow: Dict[str, str]
    # if args.workflow in ("smd_summaries", "smd_xss", "smd_xes", "smd_xss"):
    if args.workflow in ("smd_summaries", "smd_xss", "smd_xes", "smd_xss"):
        main_workflow = {
            "name": f"lute_{args.workflow}",
            "executable": arp_executable,
            "trigger": "RUN_PARAM_IS_VALUE",
            "run_param_name": "SmallData",
            "run_param_value": "done",
            "location": "S3DF",
            "parameters": param_string,
        }
    elif args.workflow == "bayfai":
        main_workflow = {
            "name": f"lute_{args.workflow}",
            "executable": arp_executable,
            "trigger": "MANUAL",
            "location": "S3DF",
            "parameters": param_string,
        }
    elif 0:
        # Replace eventually with workflows which use START_OF_RUN
        main_workflow = {
            "name": f"lute_{args.workflow}",
            "executable": arp_executable,
            "trigger": "START_OF_RUN",
            "location": "S3DF",
            "parameters": param_string,
        }
    else:
        main_workflow = {
            "name": f"lute_{args.workflow}",
            "executable": arp_executable,
            "trigger": "END_OF_RUN",
            "location": "S3DF",
            "parameters": param_string,
        }

    workflows: List[Dict[str, str]] = []
    workflows.append(main_workflow)
    # Will want to append additional auxiliary workflows eventually

    for workflow in workflows:
        logger.info(
            f"Creating eLog workflow named {workflow['name']} with parameters: {workflow['parameters']}"
        )
        krbticket: Any = KerberosTicket("HTTP@pswww.slac.stanford.edu")
        krbheaders: dict = krbticket.getAuthHeaders()
        url: str = (
            f"https://pswww.slac.stanford.edu/ws-kerb/lgbk/lgbk/{args.experiment}/ws"
            "/create_update_workflow_def"
        )
        post_params: Dict[str, Any] = {
            "url": url,
            "headers": krbheaders,
            "json": workflow,
        }
        resp: requests.models.Response = requests.post(**post_params)
        resp.raise_for_status()
        # Extra logging and such...


if __name__ == "__main__":
    main()
