"""Script submitted by Automated Run Processor (ARP) to trigger a Prefect flow.

This script is submitted by the ARP to the batch nodes. It triggers Prefect to
begin running the tasks of the specified deployment of a flow.
"""

__author__ = "Gabriel Dorlhiac"

import logging
import os
import sys
import time
import yaml
from typing import Any, Dict, Literal, Tuple, cast
from typing_extensions import TypedDict

import requests
from requests.auth import HTTPBasicAuth

from lute.execution.launch import (
    get_base_launch_parser,
    setup_launch_env,
    retrieve_run_info,
    get_lute_launch_config,
    LuteParams,
    LuteLaunchConfig,
)

# Requests, urllib have lots of debug statements. Only set level for this logger
logger: logging.Logger = logging.getLogger("Launch_Prefect")
handler: logging.Handler = logging.StreamHandler()
formatter: logging.Formatter = logging.Formatter(logging.BASIC_FORMAT)
handler.setFormatter(formatter)
logger.addHandler(handler)

if __debug__:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)


class FlowRequestDict(TypedDict):
    parameters: Dict[Literal["flow_conf"], LuteLaunchConfig]


def _retrieve_creds_and_url(instance: str = "experimental") -> Tuple[str, str, str]:
    path: str = "/sdf/group/lcls/ds/tools/lute/prefect_{instance}.txt"
    if instance == "experimental":
        path = path.format(instance=instance)
    else:
        raise ValueError('`instance` must be "experimental"')
    with open(path, "r") as f:
        user_pw: str = f.readline().strip()
        url: str = f.readline().strip()
    user: str
    pw: str
    user, pw = user_pw.split(":")
    return user, pw, url


def main() -> None:
    parser = get_base_launch_parser("Trigger Prefect to begin executing a LUTE flow.")
    args, extra_args = parser.parse_known_args()

    launch_info = setup_launch_env(args)
    experiment = launch_info["experiment"]
    run_num = launch_info["run_num"]
    jid_authorization = launch_info["authorization"]

    user: str
    pw: str
    PREFECT_API_URL: str
    user, pw, PREFECT_API_URL = _retrieve_creds_and_url()
    auth: HTTPBasicAuth = HTTPBasicAuth(user, pw)

    flow_name: str = "lute_dynamic"
    deployment_name: str = "dev"

    csrf_endpoint: str = f"{PREFECT_API_URL}/csrf-token"
    name_endpoint: str = (
        f"{PREFECT_API_URL}/deployments/name/{flow_name}/{deployment_name}"
    )

    if not os.path.exists(args.workflow_defn):
        logger.error("Workflow definition path does not exist! Exiting!")
        sys.exit(-1)

    wf_defn: Dict[str, Any]
    with open(args.workflow_defn, "r") as f:
        wf_defn = yaml.load(stream=f, Loader=yaml.FullLoader)

    run_type, is_daq2 = retrieve_run_info(
        experiment, run_num, jid_authorization, args.type
    )

    lute_params: LuteParams = {"config_file": args.config, "debug": args.debug}

    conf: LuteLaunchConfig = get_lute_launch_config(
        launch_info=launch_info,
        run_type=run_type,
        is_daq2=is_daq2,
        lute_params=lute_params,
        slurm_params=extra_args,
        workflow_defn=wf_defn,
    )

    # Get CSRF
    ##############################################
    resp = requests.get(csrf_endpoint, auth=auth, params={"client": user})

    token: str = resp.json()["token"]
    client: str = resp.json()["client"]

    # Get ID from name
    ##############################################
    resp = requests.get(name_endpoint, auth=auth)

    deployment_id: str = resp.json()["id"]

    # Launch flow_run
    ##############################################
    launch_endpoint: str = (
        f"{PREFECT_API_URL}/deployments/{deployment_id}/create_flow_run"
    )

    data: FlowRequestDict = {"parameters": {"flow_conf": conf}}
    headers: Dict[str, str] = {
        "Prefect-Csrf-Token": token,
        "Prefect-Csrf-Client": client,
    }

    resp = requests.post(
        launch_endpoint, headers=headers, json=cast(dict, data), auth=auth
    )

    flow_run_id: str = resp.json()["id"]

    # Run loop, gather logs, etc.
    ##############################################
    flow_run_state_endpoint: str = f"{PREFECT_API_URL}/flow_runs/{flow_run_id}"
    log_endpoint: str = f"{PREFECT_API_URL}/logs/filter"

    log_payload: Dict[str, Any] = {
        "logs": {
            "level": {
                "ge_": 20,
            },
            "flow_run_id": {"any_": [flow_run_id]},
        },
        "sort": "TIMESTAMP_ASC",
    }

    resp = requests.get(flow_run_state_endpoint, auth=auth)
    state: str = resp.json()["state_type"]

    last_log_idx: int = 0
    while state in ("SCHEDULED", "PENDING", "RUNNING"):
        time.sleep(5)
        # Retrieve logs for flow run, printing new ones... Bit wasteful...
        resp = requests.post(log_endpoint, headers=headers, auth=auth, json=log_payload)
        current_len_logs: int = len(resp.json())
        log_dict: Dict[str, Any]
        for log_dict in resp.json()[last_log_idx:current_len_logs]:
            print(log_dict["message"])
        last_log_idx = current_len_logs

        resp = requests.get(flow_run_state_endpoint, auth=auth)
        state = resp.json()["state_type"]


if __name__ == "__main__":
    main()
