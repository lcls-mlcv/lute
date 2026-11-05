"""Functions containing more complex logic for Task environment configuration.

Functions:
    setup_smd2_env(): Sets up psana2 environment variables.
    setup_glint_env(): GLINT's torch environment (ana-4.0.58-py3-minipytorch) and checkout.
    setup_glint_env_ana59(): The same in ana-4.0.59-py3-minipytorch.
"""

__all__ = ["setup_smd2_env", "setup_glint_env", "setup_glint_env_ana59"]
__author__ = "Gabriel Dorlhiac"

import os
import re

# import subprocess
from typing import List, Dict, Optional

import requests


def setup_smd2_env() -> Dict[str, str]:
    """Setup environment variables smalldata_tools uses with psana2.

    Tries to setup psana2 environment variables controlling the distribution of
    SRV, BD, and EB ranks automatically based on the SLURM allocation. If run
    without SLURM, it sets all relevant variables to 1. If the environment variables
    were intentionally set it will return those values instead.

    It will also write a host file to specify mpi slots to make sure rank 0 is on the
    first node.

    Returns:
        psana_vars (Dict[str,str]): Dictionary of relevant psana environment variables.
    """
    # partition: str = ...
    psana_vars: Dict[str, str] = {}
    psana_vars["PS_PARALLEL"] = "mpi"
    exp: Optional[str] = os.getenv("EXPERIMENT")
    run: Optional[str] = os.getenv("RUN_NUM")
    if exp and run:
        base_url: str = "https://pswww.slac.stanford.edu/ws/lgbk/lgbk"
        endpoint: str = f"{exp}/ws/{run}/files_for_live_mode_at_location"
        full_url: str = f"{base_url}/{endpoint}"
        try:
            resp: requests.models.Response = requests.get(
                full_url, params={"location": "S3DF"}
            )
            resp.raise_for_status()
            data_dir = resp.json()["value"]["all_present"]

            if data_dir:
                psana_vars["SIT_PSDM_DATA"] = "/sdf/data/lcls/ds"
            else:
                psana_vars["SIT_PSDM_DATA"] = "/sdf/data/lcls/drpsrcf/ffb"
        except Exception as e:
            print(e)

    # These values are the requests - may not be defined if --nodes and
    # --ntasks-per-node were not passed.
    nodes: Optional[str] = os.getenv("SLURM_NNODES")
    cores_per_node: Optional[str] = os.getenv("SLURM_NTASKS_PER_NODE")

    mpi_slots: int
    # Can get the above information from other vars
    if nodes is None or cores_per_node is None:
        cpus_per_node_str: Optional[str] = os.getenv("SLURM_JOB_CPUS_PER_NODE")
        cpus_per_node: List[int] = []
        if cpus_per_node_str:
            # Check the compact version first: 8(x3),4 ...
            # Otherwise has format of 6,4,6,2,... for each node in allocation
            for part in cpus_per_node_str.split(","):
                m: Optional[re.Match] = re.match(r"(\d+)\(x(\d+)\)", part)
                if m:
                    cpus, count = map(int, m.groups())
                    cpus_per_node.extend([cpus] * count)
                else:
                    cpus_per_node.append(int(part))
            nodes = str(len(cpus_per_node))
            # Take average for cores_per_node??
            cores_per_node = str(sum(cpus_per_node) // len(cpus_per_node))
            mpi_slots = sum(cpus_per_node) - 1
            # cores_per_node: Optional[str] = os.getenv("SLURM_TASKS_PER_NODE")
        # else not running in SLURM
        else:
            psana_vars["PS_SRV_NODES"] = "1"
            psana_vars["PS_EB_NODES"] = "1"
            return psana_vars
    else:
        mpi_slots = int(cores_per_node) * int(nodes) - 1

    # default_srv_cores: int = 16 * int(nodes)
    # Try to convert above for the case where no nodes were specified explicitly
    default_srv_cores: int = (int(cores_per_node) // 8 + 1) * int(nodes)

    # Check if the environment has been overridden, otherwise use default value
    srv_cores: int
    if (env_srv_cores := os.getenv("PS_SRV_NODES")) is not None:
        srv_cores = int(env_srv_cores)
    else:
        srv_cores = default_srv_cores

    default_eb_cores: int = (mpi_slots - srv_cores) // 16
    if default_eb_cores == 0:
        default_eb_cores = 1
    eb_cores: str
    if (env_eb_cores := os.getenv("PS_EB_NODES")) is not None:
        eb_cores = env_eb_cores
    else:
        eb_cores = str(default_eb_cores)

    psana_vars["PS_SRV_NODES"] = str(srv_cores)
    psana_vars["PS_EB_NODES"] = eb_cores

    slurm_job_nodelist: Optional[str] = os.getenv("SLURM_JOB_NODELIST")
    if slurm_job_nodelist is None:
        return psana_vars
    # cmd: List[str] = ["scontrol", "show", "hostnames", slurm_job_nodelist]
    # host_list_bytes: bytes
    # host_list_bytes, _ = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()

    # host_list: List[str] = host_list_bytes.decode().split("\n")[:-1]

    slurm_job_id: Optional[str] = os.getenv("SLURM_JOB_ID")
    if slurm_job_id is None:
        return psana_vars
    # host_file: str = f"slurm_host_{slurm_job_id}"
    # with open(host_file, "w") as f:
    #     for i in range(len(host_list)):
    #         if i == 0:
    #             f.write(f"{host_list[i]} slots=1\n")
    #         else:
    #             f.write(f"{host_list[i]}\n")

    # This calculation may not work of --ntasks-per-node is not passed
    # But on the other hand, I cannot find PS_N_RANKS used in psana code.
    n_ranks: int = int(cores_per_node) * (int(nodes) - 1) + 1

    # psana_vars["PS_HOST_FILE"] = host_file
    psana_vars["PS_N_RANKS"] = str(n_ranks)

    return psana_vars


GLINT_CONDA1_ENVS: str = "/sdf/group/lcls/ds/ana/sw/conda1/inst/envs"
GLINT_ANA_ENV: str = "ana-4.0.58-py3-minipytorch"
"""Default GLINT environment: torch 2.1. Cannot parse Jungfrau.ConfigV4 (see GLINT_ANA_ENV_ALT)."""
GLINT_ANA_ENV_ALT: str = "ana-4.0.59-py3-minipytorch"
"""torch 1.11; reads the detector ConfigV versions 4.0.58 cannot. Same GLINT code runs on both."""


def _glint_env(conda_env: str) -> Dict[str, str]:
    """Environment of the GLINT ThirdPartyTask, on top of conda1's psconda.sh (shell_source).

    What `conda activate <conda_env>` adds to psconda.sh on S3DF (measured 2026-10-09) is PATH,
    CUDA_PATH (cupy's activate hook) and the two CONDA_ markers; SIT_* come from psconda.sh. PYTHONPATH
    is the GLINT root alone: the interpreter here is the environment's python 3.9 with torch, and must
    not inherit the Executor's own site-packages.

    Args:
        conda_env (str): Name of the conda1 environment.

    Returns:
        env (Dict[str, str]): Environment variables for the Task.
    """
    from lute.io.models.glint_index import glint_root

    prefix: str = f"{GLINT_CONDA1_ENVS}/{conda_env}"
    return {
        "PATH": f"{prefix}/bin:{os.environ.get('PATH', '')}",
        "PYTHONPATH": glint_root(),
        "CUDA_PATH": prefix,
        "CONDA_PREFIX": prefix,
        "CONDA_DEFAULT_ENV": conda_env,
    }


def setup_glint_env() -> Dict[str, str]:
    """GLINT in its default environment, GLINT_ANA_ENV. See `_glint_env`."""
    return _glint_env(GLINT_ANA_ENV)


def setup_glint_env_ana59() -> Dict[str, str]:
    """GLINT in GLINT_ANA_ENV_ALT, for detectors whose ConfigV the default cannot read."""
    return _glint_env(GLINT_ANA_ENV_ALT)
