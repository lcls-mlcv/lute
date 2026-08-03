"""Common utilities for LUTE launch scripts."""

import argparse
import datetime
import getpass
import logging
import os
import sys
import uuid
from typing import cast, Any, Dict, List, Optional, Tuple, TypedDict

import requests

logger = logging.getLogger(__name__)


class LuteParams(TypedDict):
    config_file: str
    debug: bool


class LuteLaunchConfig(TypedDict):
    """Complete description of config options required for workflow execution."""

    experiment: str
    run_id: str
    JID_UPDATE_COUNTERS: Optional[str]
    ARP_ROOT_JOB_ID: str
    ARP_LOCATION: str
    Authorization: str
    user: str
    lute_location: str
    executable_subdir: str
    kerb_file: Optional[str]
    lute_params: LuteParams
    slurm_params: List[str]
    workflow: Dict[str, Any]
    run_type: Optional[str]
    is_daq2: Optional[bool]


class EnvLaunchInfo(TypedDict):
    """Subset of workflow info retrievable via eLog or environment variables."""

    experiment: str
    run_num: str
    authorization: str
    arp_job_id: str
    kerb_file: Optional[str]


def request_arp_token(exp: str, lifetime: int = 300) -> str:
    """Request an ARP token via Kerberos endpoint.

    A token is required for job submission.

    Args:
        exp (str): The experiment to request the token for. All tokens are
            scoped to a single experiment.

        lifetime (int): The lifetime, in minutes, of the token. After the token
            expires, it can no longer be used for job submission. The maximum
            time you can request is 480 minutes (i.e. 8 hours). NOTE: since this
            token is used for the entirety of a workflow, it must have a lifetime
            equal or longer than the duration of the workflow's execution time.

    Returns:
        formatted_token (str): The formated ARP token.
    """
    try:
        from kerberos import GSSError  # type: ignore
        from krtc import KerberosTicket  # type: ignore
    except ImportError:
        logger.error("Kerberos/KRTC not installed. Cannot request ARP token.")
        raise

    try:
        krbheaders: Dict[str, str] = KerberosTicket(
            "HTTP@pswww.slac.stanford.edu"
        ).getAuthHeaders()
    except GSSError:
        logger.info(
            "Cannot proceed without credentials. Try running `kinit` from the command-line."
        )
        raise
    base_url: str = "https://pswww.slac.stanford.edu/ws-kerb/lgbk/lgbk"
    token_endpoint: str = (
        f"{base_url}/{exp}/ws/generate_arp_token?token_lifetime={lifetime}"
    )
    resp: requests.models.Response = requests.get(token_endpoint, headers=krbheaders)
    resp.raise_for_status()
    token: str = resp.json()["value"]
    formatted_token: str = f"Bearer {token}"
    return formatted_token


def get_base_launch_parser(description: str) -> argparse.ArgumentParser:
    """Create a base ArgumentParser with common arguments.

    Args:
        description (str): A string to go in the help message for the launch
            script that will be built with the parser.

    Returns:
        parser (argparse.ArgumentParser): The script command-line parser.
    """
    parser = argparse.ArgumentParser(
        description=description,
        epilog="Refer to https://github.com/slac-lcls/lute for more information.",
    )
    parser.add_argument("-c", "--config", type=str, help="Path to config YAML file.")
    parser.add_argument("-d", "--debug", help="Run in debug mode.", action="store_true")
    parser.add_argument(
        "-W",
        "--workflow_defn",
        type=str,
        help="Path to a YAML file with workflow.",
        default="",
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        help="Provide an experiment if not running with ARP.",
        required=False,
    )
    parser.add_argument(
        "-r",
        "--run",
        type=str,
        help="Provide a run number if not running with ARP.",
        required=False,
    )
    parser.add_argument(
        "--type",
        help=(
            "Provide a run type to describe this workflow submission. "
            "Overrides the eLog type provided by the DAQ (if present)."
        ),
        type=str,
        default="",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        required=False,
        help=(
            "Submit against every run tagged with this eLog tag instead of a "
            "single -r/--run. Workflow DAG nodes marked `run_dependent: false` "
            "run once, over all tagged runs; nodes marked `run_dependent: true` "
            "(the default) run once per tagged run. Takes precedence over "
            "-r/--run and RUN_NUM if both are provided."
        ),
    )
    parser.add_argument(
        "--max-concurrent-runs",
        type=int,
        default=None,
        required=False,
        help=(
            "With --tag, the maximum number of tagged runs to submit "
            "concurrently for the run-dependent stage. Default is unbounded "
            "(all resolved runs submitted at once) - set this from the "
            "experiment's actual node/core allocation budget."
        ),
    )
    return parser


def setup_launch_env(args: argparse.Namespace) -> EnvLaunchInfo:
    """Setup experiment, run, and authorization information.

    Checks environment variables first (ARP submission), then falls back to
    arguments and manual token request.

    Args:
        args (argparse.Namespace): The arguments parsed via command-line.

    Returns:
        env_cmd_vars (EnvLaunchInfo): A dictionary of variables which will
            be passed either via environment or command-line to the job submission
            scripts (submit_slurm) during workflow execution.
    """
    cache_file: Optional[str] = os.getenv("KRB5CCNAME")

    experiment: Optional[str] = os.getenv("EXPERIMENT")
    run_num: Optional[str] = os.getenv("RUN_NUM")
    jid_authorization: Optional[str] = os.getenv("Authorization")
    arp_job_id: Optional[str] = os.getenv("ARP_JOB_ID")

    if jid_authorization is None or experiment is None or run_num is None:
        if cache_file is None:
            logger.info("No Kerberos cache. Try running `kinit` and resubmitting.")
            sys.exit(-1)

        if args.experiment is None or args.run is None:
            logger.info(
                (
                    "You must provide a `-e ${EXPERIMENT}` and `-r ${RUN_NUM}` "
                    "if not running with the ARP!\n"
                    "If you submitted this from the eLog and are seeing this error "
                    "please contact the maintainers."
                )
            )
            sys.exit(-1)

        experiment = args.experiment
        run_num = args.run
        assert isinstance(experiment, str)
        assert isinstance(run_num, str)
        os.environ["EXPERIMENT"] = experiment
        os.environ["RUN_NUM"] = run_num

        jid_authorization = request_arp_token(experiment)
        os.environ["Authorization"] = jid_authorization

        if arp_job_id is None:
            arp_job_id = uuid.uuid4().hex[:24]
            os.environ["ARP_JOB_ID"] = arp_job_id

    return {
        "experiment": experiment,
        "run_num": run_num,
        "authorization": jid_authorization,
        "arp_job_id": cast(str, arp_job_id),
        "kerb_file": cache_file,
    }


def clean_python_path(is_daq2: Optional[bool]) -> None:
    """Cleanup PYTHONPATH environment leakage.

    Args:
        is_daq2 (Optional[bool]): Whether running a job for a psana2 experiment. If not,
            we can make sure to remove the PYTHONPATH modifications from the
            environment.
    """
    curr_pythonpath: str = os.environ.get("PYTHONPATH", "")
    if curr_pythonpath:
        if is_daq2 is not None and not is_daq2:
            logger.info("Cleaning PYTHONPATH of LCLS2 environment modifications.")
            os.environ["PYTHONPATH"] = ":".join(
                [
                    pp
                    for pp in curr_pythonpath.split(":")
                    if "conda2" not in pp and "lcls2" not in pp
                ]
            )


def retrieve_run_info(
    experiment: str, run_num: str, authorization: str, override_type: str = ""
) -> Tuple[str, Optional[bool]]:
    """Retrieve run type and DAQ version from eLog API.

    Args:
        experiment (str): The experiment.

        run_num (str): The run number.

        authorization (str): The JWT token for making the API requests to get run
            documents from the eLog.

        override_type (str): An optional override type to use instead of the actual
            run type. (This is only used to maintain the structure of older launch
            scripts.)

    Returns:
        run_type (str): The run type (potentially overriden by override_type).

        is_daq2 (Optional[bool]): Whether this is an LCLS1 or LCLS2 experiment if
            it could be determined.
    """
    headers: Dict[str, str] = {
        "Authorization": authorization,
    }
    base_url: str = "https://pswww.slac.stanford.edu/ws-jwt/lgbk/lgbk"
    run_doc_url: str = f"{base_url}/{experiment}/ws/runs/{run_num}"

    run_type: str = "UNKNOWN"
    is_daq2: Optional[bool] = None

    try:
        resp = requests.get(run_doc_url, headers=headers)
        if resp.status_code != 200:
            logger.warning(
                "Unable to retrieve run document! No `run_type` information will be used! "
                "No information about psana1/psana2 can be retrieved."
            )
        else:
            data = resp.json()["value"]
            run_type = override_type if override_type else data.get("type", "UNKNOWN")

            # Try checking for "psana1" vs "psana2" by searching for "drp" in detector names
            params = data.get("params", {})
            for key in params.keys():
                if "/drp/" in key:
                    is_daq2 = True
                    break
            else:
                is_daq2 = False
    except Exception as e:
        logger.error(f"Error retrieving run info: {e}")

    clean_python_path(is_daq2=is_daq2)

    return run_type, is_daq2


def get_lute_launch_config(
    launch_info: EnvLaunchInfo,
    run_type: str,
    is_daq2: Optional[bool],
    lute_params: LuteParams,
    slurm_params: List[str],
    workflow_defn: Dict[str, Any] = {},
    lute_location: Optional[str] = None,
    executable_subdir: Optional[str] = None,
) -> LuteLaunchConfig:
    """Construct the standardized LUTE launch configuration dictionary.

    Args:
        launch_info (EnvLaunchInfo): The set of launch information retrievable from
            command-line arguments or environment variables. See `setup_launch_env`.

        run_type (str): The run type.

        is_daq2 (Optional[bool]): Whether it is an LCLS1 or LCLS2 experiment.

        lute_params (LuteParams): The config file and debug options for LUTE.

        slurm_params (List[str]): The list of SLURM parameters (if any).

        workflow_defn (Dict[str, Any]): The parsed workflow definition. Empty dict
            if not using YAML DAGs.

        lute_location (Optional[str]): The path to the LUTE installation. If not
            provided, it is assumed to be the parent directory of the current
            working directory.

        executable_subdir (Optional[str]): The subdirectory where the launch
            executable is located. If not provided, it is assumed to be the
            basename of the current working directory.
    """
    if lute_location is None:
        lute_location = os.path.abspath(os.path.join(os.getcwd(), ".."))
    if executable_subdir is None:
        executable_subdir = os.path.basename(os.getcwd())

    return {
        "experiment": launch_info["experiment"],
        "run_id": f"{launch_info['run_num']}_{datetime.datetime.utcnow().isoformat()}",
        "JID_UPDATE_COUNTERS": os.getenv("JID_UPDATE_COUNTERS"),
        "ARP_ROOT_JOB_ID": launch_info["arp_job_id"],
        "ARP_LOCATION": os.getenv("ARP_LOCATION", "S3DF"),
        "Authorization": launch_info["authorization"],
        "user": getpass.getuser(),
        "lute_location": lute_location,
        "executable_subdir": executable_subdir,
        "kerb_file": launch_info["kerb_file"],
        "lute_params": lute_params,
        "slurm_params": slurm_params,
        "workflow": workflow_defn,
        "run_type": run_type,
        "is_daq2": is_daq2,
    }


def get_concurrent_job_steps(wf: List[Any]) -> int:
    """Return the maximum number of concurrent JobSteps.

    This can be used to determine how many threads to add to the threadpool for the
    workflow manager.

    NOTE: This is a very basic calculation - if you have complicated branch structures
    it may undershoot the number of concurrent jobs. For safety you can add one to
    the returned value - this will likely cover 99% of all workflow cases.

    Args:
        wf (List[Any]): The workflow (list of JobSteps).

    Returns:
        max_concurrent_jobs (int): The maximum number of jobs found to run in
            parallel at any given time.
    """
    num_concurrent_steps: int = len(wf)
    for step in wf:
        # Assuming step has a 'next' attribute which is a list of JobSteps
        next_concurrent_steps: int = get_concurrent_job_steps(getattr(step, "next", []))
        num_concurrent_steps = max(num_concurrent_steps, next_concurrent_steps)

    return num_concurrent_steps
