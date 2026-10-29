import argparse
import json
import logging
import os
import re
import signal
import sys
import time
import types
from typing import Any, Dict, Optional, Tuple, Type, cast

import lute.execution.subprocess_utils
from lute.tasks.task import Task, ThirdPartyTask
from lute.execution.ipc import Message
from lute.io.config import parse_config
from lute.io.models.base import TaskParameters, ThirdPartyParameters
from lute.io.db import record_parameters_db
from lute.io.parameters import RowIds


def get_task() -> Optional[Task]:
    """Return the current Task."""
    objects: Dict[str, Any] = globals()
    for _, obj in objects.items():
        if isinstance(obj, Task):
            return obj
    return None


def timeout_handler(signum: int, frame: Optional[types.FrameType]) -> Any:
    """Log and exit gracefully on Task timeout."""
    task: Optional[Task] = get_task()
    if task:
        msg: Message = Message(contents="Timed out.", signal="TASK_FAILED")
        task._report_to_executor(msg)
        task.clean_up_timeout()
        sys.exit(-1)


def setup_env() -> bool:
    """Setup a new Task environment for first-party Tasks.

    Returns:
        setup_new_env (bool): Returns True if a new environment was requested.
    """
    setup_new_env: bool = False
    new_env: Dict[str, str] = {}
    for key, value in os.environ.items():
        if "LUTE_TENV_" in key:
            # Set if using a custom environment
            setup_new_env = True
            new_key: str = key[10:]
            new_env[new_key] = value
    if setup_new_env:
        os.environ.update(new_env)
    return setup_new_env


def is_mpi_job() -> Tuple[bool, int]:
    """Determine whether this is an MPI submission without initializing a context.

    Returns:
        is_mpi_job (bool): Whether it is an MPI submission.

        mpi_rank (int): The MPI rank for this process, or -1 if not an MPI job.
    """
    # MPI rank returned as negative if not an MPI submission
    mpi_rank: int = -1
    mpi_size: int = 1
    if "OMPI_COMM_WORLD_RANK" in os.environ:
        mpi_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        mpi_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))
    elif "PMI_RANK" in os.environ:
        mpi_rank = int(os.environ["PMI_RANK"])
        mpi_size = int(os.environ.get("PMI_SIZE", 1))
    elif "SLURM_PROCID" in os.environ:
        mpi_rank = int(os.environ["SLURM_PROCID"])
        mpi_size = int(os.environ.get("SLURM_NTASKS", 1))
        # When using SLURM to figure out MPI/non-MPI, subtract 1 for Executor core
        if mpi_size > 1:
            mpi_size -= 1
    # Need other env vars? PMIX_RANK, MV2_COMM_WORLD_RANK?
    return mpi_size > 1, mpi_rank


signal.signal(signal.SIGALRM, timeout_handler)

if __debug__:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

logger: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="run_subprocess_task",
        description="Analysis Task run as a subprocess managed by a LUTE Executor.",
        epilog="Refer to https://github.com/slac-lcls/lute for more information.",
    )
    parser.add_argument(
        "-c", "--config", type=str, help="Path to config file with Task parameters."
    )
    parser.add_argument(
        "-t", "--taskname", type=str, help="Name of the Task to run.", default="test"
    )

    args: argparse.Namespace = parser.parse_args()
    config: str = args.config
    task_name: str = args.taskname
    task_parameters: TaskParameters = parse_config(
        task_name=task_name, config_path=config
    )

    # In the event we are using parameter sweeps we now need to remove the
    # appended _XX number from the Task name
    task_name = re.sub(r"_\d+$", "", task_name)
    # For now, we will only use the exec with first-party Task's that require a new env.
    TaskType: Type[Task]
    if isinstance(task_parameters, ThirdPartyParameters) or not setup_env():
        # lute.execution.subprocess_utils.USE_PYDANTIC_MODELS has a bool
        # It defaults to True, but we set here in case anything changes in the future
        lute.execution.subprocess_utils.USE_PYDANTIC_MODELS = True
        if isinstance(task_parameters, ThirdPartyParameters):
            TaskType = ThirdPartyTask
        else:
            from lute.tasks import import_task, TaskNotFoundError

            try:
                TaskType = import_task(task_name=task_name)
            except TaskNotFoundError:
                logger.debug(
                    (
                        f"Task {task_name} not found! Things to double check:"
                        "\t - The spelling of the Task name."
                        "\t - Has the Task been registered in lute.tasks.import_task."
                    )
                )
                sys.exit(-1)
        task: Task = TaskType(params=task_parameters)
        task.run()
    else:
        exec_script_template: str = lute.execution.subprocess_utils.exec_script_template
        # `lute.execution.subprocess_utils.USE_PYDANTIC_MODELS` needs to be set to False
        # but this gets set in the `exec_script_template` and is only required by the
        # process after the exec

        # We are a first-party Task that needs a new environment
        # Record the parameters - but only once if using MPI
        use_mpi: bool
        rank: int
        use_mpi, rank = is_mpi_job()

        row_ids: Optional[RowIds]
        if use_mpi:
            # Use args.task_name for the files in case of parameter sweeps
            row_id_file: str = f"{args.taskname}_row_ids.out"
            temp_row_id_file: str = f"{args.taskname}_row_ids.inprogress"
            if rank == 0:
                print("Running a first-party Task in a new environment.", flush=True)
                row_ids = record_parameters_db(task_parameters)
                assert row_ids is not None
                with open(temp_row_id_file, "w") as f:
                    f.write("\n")

                with open(row_id_file, "w") as f:
                    json.dump(row_ids, f)
                os.remove(temp_row_id_file)
                # This allows rank 0 to delete it later - this is done internally
                # in Task. It occurs in _signal_start after a Barrier so rank 0 knows
                # other ranks have actually started before deleting anything
                os.environ["LUTE_BOOTSTRAP_FILE"] = row_id_file
            else:
                while not os.path.exists(row_id_file) or os.path.exists(
                    temp_row_id_file
                ):
                    time.sleep(0.01)

                with open(row_id_file, "r") as f:
                    row_ids = cast(RowIds, json.load(f))
        else:
            print("Running a first-party Task in a new environment.", flush=True)
            row_ids = record_parameters_db(task_parameters)
        work_dir: str = task_parameters.lute_config.work_dir

        exec_script: str = exec_script_template.format(
            work_dir=work_dir,
            task_name=task_name,
            row_ids=row_ids,
        )
        if __debug__:
            os.execlp("python", "python", "-B", "-c", exec_script)
        else:
            os.execlp("python", "python", "-O", "-c", exec_script)


if __name__ == "__main__":
    main()
