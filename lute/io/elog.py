"""Provides utilities for communicating with the LCLS eLog.

Make use of various eLog API endpoint to retrieve information or post results.

Functions:
    get_elog_opr_auth(exp: str): Return an authorization object to interact with
        eLog API as an opr account for the hutch where `exp` was conducted.

    get_elog_kerberos_auth(): Return the authorization headers for the user account
        submitting the job.

    elog_http_request(url: str, request_type: str, **params): Make an HTTP
        request to the API endpoint at `url`.

    format_file_for_post(in_file: Union[str, tuple, list]): Prepare files
        according to the specification needed to add them as attachments to eLog
        posts.

    post_elog_message(exp: str, msg: str, tag: Optional[str],
                      title: Optional[str],
                      in_files: List[Union[str, tuple, list]],
                      auth: Optional[Union[HTTPBasicAuth, Dict]] = None)
        Post a message to the eLog.

    post_elog_run_status(data: Dict[str, Union[str, int, float]],
                         update_url: Optional[str] = None)
        Post a run status to the summary section on the Workflows>Control tab.

    post_elog_run_table(exp: str, run: int, data: Dict[str, Any],
                       auth: Optional[Union[HTTPBasicAuth, Dict]] = None)
        Update run table in the eLog.

    get_elog_runs_by_tag(exp: str, tag: str,
                         auth: Optional[Union[HTTPBasicAuth, Dict]] = None)
        Return a list of runs with a specific tag.

    get_elog_params_by_run(exp: str, params: List[str], runs: Optional[List[int]])
        Retrieve the requested parameters by run. If no run is provided, retrieve
        the requested parameters for all runs.
"""

__all__ = [
    "post_elog_message",
    "post_elog_run_table",
    "get_elog_runs_by_tag",
    "post_elog_run_status",
    "get_elog_params_by_run",
]
__author__ = "Gabriel Dorlhiac"

import requests
from requests.auth import HTTPBasicAuth
import mimetypes
import os
import logging
from typing import Any, Dict, Optional, List, Union, Tuple, Mapping, cast
from io import BufferedReader

from lute.io.exceptions import ElogFileFormatError
from lute.execution.logging import get_logger

if __debug__:
    logging.basicConfig(level=logging.DEBUG)
    logging.captureWarnings(True)
else:
    logging.basicConfig(level=logging.INFO)

logger: logging.Logger = get_logger(__name__, is_task=False)


def get_elog_workflows(exp: str) -> Dict[str, str]:
    """Get the current workflow definitions for an experiment.

    Returns:
        defns (Dict[str, str]): A dictionary of workflow definitions.
    """
    raise NotImplementedError


def post_elog_workflow(
    exp: str,
    name: str,
    executable: str,
    wf_params: str,
    *,
    trigger: str = "run_end",
    location: str = "S3DF",
    **trig_args: str,
) -> None:
    """Create a new eLog workflow, or update an existing one.

    The workflow will run a specific executable as a batch job when the
    specified trigger occurs. The precise arguments may vary depending on the
    selected trigger type.

    Args:
        name (str): An identifying name for the workflow. E.g. "process data"

        executable (str): Full path to the executable to be run.

        wf_params (str): All command-line parameters for the executable as a string.

        trigger (str): When to trigger execution of the specified executable.
            One of:
                - 'manual': Must be manually triggered. No automatic processing.
                - 'run_start': Execute immediately if a new run begins.
                - 'run_end': As soon as a run ends.
                - 'param_is': As soon as a parameter has a specific value for a run.

        location (str): Where to submit the job. S3DF or NERSC.

        **trig_args (str): Arguments required for a specific trigger type.
            trigger='param_is' - 2 Arguments
                trig_param (str): Name of the parameter to watch for.
                trig_param_val (str): Value the parameter should have to trigger.
    """
    endpoint: str = f"{exp}/ws/create_update_workflow_def"
    trig_map: Dict[str, str] = {
        "manual": "MANUAL",
        "run_start": "START_OF_RUN",
        "run_end": "END_OF_RUN",
        "param_is": "RUN_PARAM_IS_VALUE",
    }
    if trigger not in trig_map.keys():
        raise NotImplementedError(
            f"Cannot create workflow with trigger type: {trigger}"
        )
    wf_defn: Dict[str, str] = {
        "name": name,
        "executable": executable,
        "parameters": wf_params,
        "trigger": trig_map[trigger],
        "location": location,
    }
    if trigger == "param_is":
        if "trig_param" not in trig_args or "trig_param_val" not in trig_args:
            raise RuntimeError(
                "Trigger type 'param_is' requires: 'trig_param' and 'trig_param_val' arguments"
            )
        wf_defn.update(
            {
                "run_param_name": trig_args["trig_param"],
                "run_param_val": trig_args["trig_param_val"],
            }
        )
    post_params: Dict[str, Dict[str, str]] = {"json": wf_defn}
    status_code, resp_msg, _ = elog_http_request(
        exp, endpoint=endpoint, request_type="POST", **post_params
    )


def get_elog_active_expmt(hutch: str, *, endstation: int = 0) -> str:
    """Get the current active experiment for a hutch.

    This function is one of two functions to manage the HTTP request independently.
    This is because it does not require an authorization object, and its result
    is needed for the generic function `elog_http_request` to work properly.

    Args:
        hutch (str): The hutch to get the active experiment for.

        endstation (int): The hutch endstation to get the experiment for. This
            should generally be 0.
    """

    base_url: str = "https://pswww.slac.stanford.edu/ws/lgbk/lgbk"
    endpoint: str = "ws/activeexperiment_for_instrument_station"
    url: str = f"{base_url}/{endpoint}"
    params: Dict[str, str] = {"instrument_name": hutch, "station": f"{endstation}"}
    resp: requests.models.Response = requests.get(url, params)
    if resp.status_code > 300:
        raise RuntimeError(
            f"Error getting current experiment!\n\t\tIncorrect hutch: '{hutch}'?"
        )
    if resp.json()["success"]:
        return resp.json()["value"]["name"]
    else:
        msg: str = resp.json()["error_msg"]
        raise RuntimeError(f"Error getting current experiment! Err: {msg}")


def get_elog_auth(exp: str) -> Union[HTTPBasicAuth, Dict[str, str]]:
    """Determine the appropriate auth method depending on experiment state.

    Returns:
        auth (HTTPBasicAuth | Dict[str, str]): Depending on whether an experiment
            is active/live, returns authorization for the hutch operator account
            or the current user submitting a job.
    """
    hutch: str = exp[:3]
    if exp.lower() == get_elog_active_expmt(hutch=hutch).lower():
        return get_elog_opr_auth(exp)
    else:
        return get_elog_kerberos_auth()


def get_elog_opr_auth(exp: str) -> HTTPBasicAuth:
    """Produce authentication for the "opr" user associated to an experiment.

    This method uses basic authentication using username and password.

    Args:
        exp (str): Name of the experiment to produce authentication for.

    Returns:
        auth (HTTPBasicAuth): HTTPBasicAuth for an active experiment based on
            username and password for the associated operator account.
    """
    opr: str = f"{exp[:3]}opr"
    with open("/sdf/group/lcls/ds/tools/forElogPost.txt", "r") as f:
        pw: str = f.readline()[:-1]
    return HTTPBasicAuth(opr, pw)


def get_elog_kerberos_auth() -> Dict[str, str]:
    """Returns Kerberos authorization key.

    This functions returns authorization for the USER account submitting jobs.
    It assumes that `kinit` has been run.

    Returns:
        auth (Dict[str, str]): Dictionary containing Kerberos authorization key.
    """
    from krtc import KerberosTicket  # type: ignore

    return KerberosTicket("HTTP@pswww.slac.stanford.edu").getAuthHeaders()


def elog_http_request(
    exp: str, endpoint: str, request_type: str, **params
) -> Tuple[int, str, Optional[Any]]:
    """Make an HTTP request to the eLog.

    This method will determine the proper authorization method and update the
    passed parameters appropriately. Functions implementing specific endpoint
    functionality and calling this function should only pass the necessary
    endpoint-specific parameters and not include the authorization objects.

    Args:
        exp (str): Experiment.

        endpoint (str): eLog API endpoint.

        request_type (str): Type of request to make. Recognized options: POST or
            GET.

        **params (Dict): Endpoint parameters to pass with the HTTP request!
            Differs depending on the API endpoint. Do not include auth objects.

    Returns:
        status_code (int): Response status code. Can be checked for errors.

        msg (str): An error message, or a message saying SUCCESS.

        value (Optional[Any]): For GET requests ONLY, return the requested
            information.
    """
    auth: Union[HTTPBasicAuth, Dict[str, str]] = get_elog_auth(exp)
    base_url: str
    if isinstance(auth, HTTPBasicAuth):
        params.update({"auth": auth})
        base_url = "https://pswww.slac.stanford.edu/ws-auth/lgbk/lgbk"
    elif isinstance(auth, dict):
        params.update({"headers": auth})
        base_url = "https://pswww.slac.stanford.edu/ws-kerb/lgbk/lgbk"

    url: str = f"{base_url}/{endpoint}"

    resp: requests.models.Response
    if request_type.upper() == "POST":
        resp = requests.post(url, **params)
    elif request_type.upper() == "GET":
        resp = requests.get(url, **params)
    else:
        return (-1, "Invalid request type!", None)

    status_code: int = resp.status_code
    msg: str = "SUCCESS"

    if resp.json()["success"] and request_type.upper() == "GET":
        return (status_code, msg, resp.json()["value"])

    if status_code >= 300:
        msg = f"Error when posting to eLog: Response {status_code}"

    if not resp.json()["success"]:
        err_msg = resp.json()["error_msg"]
        msg += f"\nInclude message: {err_msg}"
    return (resp.status_code, msg, None)


def format_file_for_post(
    in_file: Union[str, tuple, list],
) -> Tuple[str, Tuple[str, BufferedReader], Any]:
    """Format a file for attachment to an eLog post.

    The eLog API expects a specifically formatted tuple when adding file
    attachments. This function prepares the tuple to specification given a
    number of different input types.

    Args:
        in_file (str | tuple | list): File to include as an attachment in an
            eLog post.
    """
    description: str
    fptr: BufferedReader
    ftype: Optional[str]
    if isinstance(in_file, str):
        description = os.path.basename(in_file)
        fptr = open(in_file, "rb")
        ftype = mimetypes.guess_type(in_file)[0]
    elif isinstance(in_file, tuple) or isinstance(in_file, list):
        description = in_file[1]
        fptr = open(in_file[0], "rb")
        ftype = mimetypes.guess_type(in_file[0])[0]
    else:
        raise ElogFileFormatError(f"Unrecognized format: {in_file}")

    out_file: Tuple[str, Tuple[str, BufferedReader], Any] = (
        "files",
        (description, fptr),
        ftype,
    )
    return out_file


def _get_current_run_status(update_url: str) -> Dict[str, Union[str, int, float]]:
    """Retrieve the current 'counters' or status for a workflow.

    This function is intended to be called from the posting function to allow
    for incremental updates to the status. It will only work for currently
    running workflows, as it does not go back to the database, only the JID/ARP.

    Args:
        update_url (str): The JID_UPDATE_COUNTERS url.

    Returns:
        data (Dict[str, str]): A dictionary of key:value pairs of currently
            displayed data.
    """
    import getpass

    if os.getenv("ARP_ROOT_JOB_ID") is None or os.getenv("RUN_NUM") is None:
        raise RuntimeError(
            "Cannot call _get_current_run_status with no ROOT_JOB or RUN_NUM"
        )
    user: str = getpass.getuser()
    replace_counters_parts: List[str] = update_url.split("/")
    exp: str = replace_counters_parts[-2]
    get_url: str = "/".join(replace_counters_parts[:-3])
    get_url = f"{get_url}/{exp}/get_counters"
    job_doc: Dict[str, str] = {
        "_id": cast(str, os.environ.get("ARP_ROOT_JOB_ID")),
        "experiment": exp,
        "run_num": cast(str, os.environ.get("RUN_NUM")),
        "user": user,
    }
    resp: requests.models.Response = requests.post(
        get_url,
        json=job_doc,
        headers={"Authorization": os.environ.get("Authorization", "")},
    )
    current_status: Dict[str, Union[str, int, float]] = {
        d["key"]: d["value"] for d in resp.json()["value"]
    }
    return current_status


def post_elog_run_status(
    data: Mapping[str, Union[str, int, float]], update_url: Optional[str] = None
) -> None:
    """Post a summary to the status/report section of a specific run.

    In contrast to most eLog update/post mechanisms, this function searches
    for a specific environment variable which contains a specific URL for
    posting. This is updated every job/run as jobs are submitted by the JID.
    The URL can optionally be passed to this function if it is known.

    Args:
        data (Dict[str, Union[str, int, float]]): The data to post to the eLog
            report section. Formatted in key:value pairs.

        update_url (Optional[str]): Optional update URL. If not provided, the
            function searches for the corresponding environment variable. If
            neither is found, the function aborts
    """
    if update_url is None:
        update_url = os.environ.get("JID_UPDATE_COUNTERS")
        if update_url is None:
            logger.error("eLog Update Failed! JID_UPDATE_COUNTERS is not defined!")
            return
    current_status: Dict[str, Union[str, int, float]] = _get_current_run_status(
        update_url
    )
    current_status.update(data)
    post_list: List[Dict[str, str]] = [
        {"key": f"{key}", "value": f"{value}"} for key, value in current_status.items()
    ]
    _: requests.models.Response = requests.post(update_url, json=post_list)


def post_elog_message(
    exp: str,
    msg: str,
    *,
    tag: Optional[str],
    title: Optional[str],
    in_files: List[Union[str, tuple, list]] = [],
) -> Optional[str]:
    """Post a new message to the eLog. Inspired by the `elog` package.

    Args:
        exp (str): Experiment name.

        msg (str): BODY of the eLog post.

        tag (str | None): Optional "tag" to associate with the eLog post.

        title (str | None): Optional title to include in the eLog post.

        in_files (List[str | tuple | list]): Files to include as attachments in
            the eLog post.

    Returns:
        err_msg (str | None): If successful, nothing is returned, otherwise,
            return an error message.
    """
    # MOSTLY CORRECT
    out_files: list = []
    for f in in_files:
        try:
            out_files.append(format_file_for_post(in_file=f))
        except ElogFileFormatError as err:
            logger.error(f"ElogFileFormatError: {err}")
    post: Dict[str, str] = {}
    post["log_text"] = msg
    if tag:
        post["log_tags"] = tag
    if title:
        post["log_title"] = title

    endpoint: str = f"{exp}/ws/new_elog_entry"

    params: Dict[str, Any] = {"data": post}

    if out_files:
        params.update({"files": out_files})

    status_code, resp_msg, _ = elog_http_request(
        exp=exp, endpoint=endpoint, request_type="POST", **params
    )

    if resp_msg != "SUCCESS":
        return resp_msg
    # NEED to handle/propagate errors...
    return None


def post_elog_run_table(
    exp: str,
    run: int,
    data: Dict[str, Any],
) -> Optional[str]:
    """Post data for eLog run tables.

    Args:
        exp (str): Experiment name.

        run (int): Run number corresponding to the data being posted.

        data (Dict[str, Any]): Data to be posted in format
            data["column_header"] = value.

    Returns:
        err_msg (None | str): If successful, nothing is returned, otherwise,
            return an error message.
    """
    endpoint: str = f"run_control/{exp}/ws/add_run_params"

    params: Dict[str, Any] = {"params": {"run_num": run}, "json": data}

    status_code, resp_msg, _ = elog_http_request(
        exp=exp, endpoint=endpoint, request_type="POST", **params
    )

    if resp_msg != "SUCCESS":
        return resp_msg
    # NEED to handle/propagate errors....
    return None


def get_elog_runs_by_tag(
    exp: str, tag: str, auth: Optional[Union[HTTPBasicAuth, Dict]] = None
) -> List[int]:
    """Retrieve run numbers with a specified tag.

    Args:
        exp (str): Experiment name.

        tag (str): The tag to retrieve runs for.
    """
    endpoint: str = f"{exp}/ws/get_runs_with_tag?tag={tag}"
    params: Dict[str, Any] = {}

    status_code, resp_msg, tagged_runs = elog_http_request(
        exp=exp, endpoint=endpoint, request_type="GET", **params
    )

    if not tagged_runs:
        tagged_runs = []

    return tagged_runs


def get_elog_params_by_run(
    exp: str, params: List[str], runs: Optional[List[int]] = None
) -> Optional[Dict[str, str]]:
    """Retrieve requested parameters by run or for all runs.

    Args:
        exp (str): Experiment to retrieve parameters for.

        params (List[str]): A list of parameters to retrieve. These can be any
            parameter recorded in the eLog (PVs, parameters posted by other
            Tasks, etc.)
    """
    return None
