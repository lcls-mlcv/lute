"""Launch a LUTE workflow using Maestro as the workflow manager."""

__all__ = []
__author__ = "Gabriel Dorlhiac"

import logging
import os
import random
import socket
import sys
from typing import List, Optional, Dict

from lute.execution.launch import (
    get_base_launch_parser,
    setup_launch_env,
    retrieve_run_info,
    get_concurrent_job_steps,
)
from maestro._maestro import _maestro
from maestro.parser import load_lute_dag

logger: logging.Logger = logging.getLogger("PyMaestro")
handler: logging.Handler = logging.StreamHandler()
formatter: logging.Formatter = logging.Formatter(logging.BASIC_FORMAT)
handler.setFormatter(formatter)
logger.addHandler(handler)


def find_random_port(
    min_port: int = 41923, max_port: int = 64324, max_tries: int = 20
) -> Optional[int]:
    ports: List[int] = random.choices(range(min_port, max_port), k=max_tries)

    for port in ports:
        sock: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            sock.bind(("", port))
            sock.close()
            del sock
            return port
        except Exception:
            continue
    return None


def main():
    parser = get_base_launch_parser(
        "A light-weight workflow manager which executes LUTE Managed Tasks."
    )
    parser.add_argument(
        "--num_server_threads",
        type=int,
        help="Number of threads to use for the HTTP server.",
        default=2,
    )
    parser.add_argument(
        "--unbuffered",
        help=(
            "Flush logs immediately. Warning: This can make output confusing "
            "when running multiple managed Tasks are running in parallel."
        ),
        action="store_true",
    )

    args, extra_args = parser.parse_known_args()

    launch_info = setup_launch_env(args)
    experiment = launch_info["experiment"]
    run_num = launch_info["run_num"]
    jid_authorization = launch_info["authorization"]

    run_type, is_daq2 = retrieve_run_info(
        experiment, run_num, jid_authorization, args.type
    )
    branch_conditions: Dict[str, bool] = {
        "daq2": is_daq2 if is_daq2 else True,
        run_type: True,
    }

    bin_dir: str = os.path.dirname(os.path.realpath(sys.argv[0]))
    lute_location: str = os.path.abspath(f"{bin_dir}/..")

    wf_defn: List[_maestro.JobStep] = load_lute_dag(
        workflow_path=args.workflow_defn,
        lute_location=lute_location,
        executable_subdir=bin_dir.split("/")[-1],
        config_file=args.config,
        debug=args.debug,
        default_slurm_params=" ".join(extra_args),
        branch_conditions=branch_conditions,
    )

    num_concurrent_steps: int = get_concurrent_job_steps(wf_defn)
    manager_host: str = socket.gethostname()
    manager_port: Optional[int] = find_random_port()
    if manager_port is None:
        raise RuntimeError("Cannot bind a port!")
    os.environ["LUTE_MANAGER_URL"] = f"{manager_host}:{manager_port}"
    # fmt: off
    manager_params: _maestro.ManagerParameters = _maestro.ManagerParameters(
        num_concurrent_steps,                    # Manager threads
        args.num_server_threads,                 # Server threads
        args.unbuffered,                         # Unbuffered logs
        "0.0.0.0",                               # Server IP
        manager_port,                            # Server port
        _maestro.LauncherType.SlurmLauncherType, # Launch mechanism
        is_daq2,                                 # Is daq2?
        run_type,                                # Run type
    )
    # fmt: on

    status: str = _maestro.run_workflow(wf_defn, manager_params)
    if status == "FAILED":
        raise RuntimeError("Workflow Failed!")


if __name__ == "__main__":
    main()
