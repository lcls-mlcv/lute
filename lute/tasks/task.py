"""Base classes for implementing analysis tasks.

Classes:
    Task: Abstract base class from which all analysis tasks are derived.

    ThirdPartyTask: Class to run a third-party executable binary as a `Task`.
"""

__all__ = ["Task", "ThirdPartyTask"]
__author__ = "Gabriel Dorlhiac"

import os
import signal
import socket
import sys
import time
import warnings
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    TextIO,
    Type,
    Union,
    TYPE_CHECKING,
)

import lute.execution.subprocess_utils

if TYPE_CHECKING or lute.execution.subprocess_utils.USE_PYDANTIC_MODELS:
    from lute.io.models.base import (
        TaskParameters,
        ThirdPartyParameters,
        TemplateParameters,
        TemplateConfig,
        AnalysisHeader,
    )
else:
    from lute.io.parameters import (
        TaskParameters,
        ThirdPartyParameters,
        TemplateParameters,
        TemplateConfig,
        AnalysisHeader,
    )
from lute.execution.ipc import (
    Communicator,
    Message,
    MessageType,
    PipeCommunicator,
    SocketCommunicator,
    TaskRequest,
    TaskRequestMessage,
    TaskMetadataMessage,
)
from lute.execution.debug_utils import LUTE_DEBUG_EXIT
from lute.io.parameters import RowIds
from lute.tasks.dataclasses import TaskParametersDBReference, TaskResult, TaskStatus

if __debug__:
    warnings.simplefilter("default")
    os.environ["PYTHONWARNINGS"] = "default"

    def lute_warn(
        message: Union[str, Warning],
        category: Type[Warning],
        filename: str,
        lineno: int,
        file: Union[TextIO, None] = None,
        line: Union[str, None] = None,
    ) -> None:
        formatted_warning: str = warnings.formatwarning(
            message, category=category, filename=filename, lineno=lineno, line=line
        )
        msg: Message = Message(contents=formatted_warning)
        communicator: PipeCommunicator = PipeCommunicator()

        communicator.write(msg)

    warnings.showwarning = lute_warn
else:
    warnings.simplefilter("ignore")


class Task(ABC):
    """Abstract base class for analysis tasks.

    Attributes:
        name (str): The name of the Task.
    """

    def __init__(
        self,
        *,
        params: TaskParameters,
        use_mpi: bool = False,
        row_ids: Optional[RowIds] = None,
    ) -> None:
        """Initialize a Task.

        Args:
            params (TaskParameters): Parameters needed to properly configure
                the analysis task. These are NOT related to execution parameters
                (number of cores, etc), except, potentially, in case of binary
                executable sub-classes.

            use_mpi (bool): Whether this Task requires the use of MPI.
                This determines the behaviour and timing of certain signals
                and ensures appropriate barriers are placed to not end
                processing until all ranks have finished.

            row_ids (Optional[RowIds]): If provided the parameter object was stored
                by the Task layer. Instead of sending a `TaskParameters` object, a
                set of row_ids will be sent. This is because the Executor does not
                know how the `lute.io.parameters.TaskParameters` was constructed,
                so with the RowIds it will be reconstruct it itself.
        """
        self.name: str = str(type(self)).split("'")[1].split(".")[-1]
        self._result: TaskResult = TaskResult(
            task_name=self.name,
            task_status=TaskStatus.PENDING,
            summary="PENDING",
            payload="",
        )
        self._task_parameters: TaskParameters = params
        if (
            hasattr(self._task_parameters.Config, "result_from_params")
            and self._task_parameters.Config.result_from_params is not None
        ):
            object.__setattr__(
                self._task_parameters,
                "_result_from_params",
                self._task_parameters.Config.result_from_params,
            )
        timeout: int = self._task_parameters.lute_config.task_timeout
        signal.setitimer(signal.ITIMER_REAL, timeout)

        run_directory: Optional[str] = self._task_parameters.Config.run_directory
        if run_directory is not None:
            try:
                os.chdir(run_directory)
            except FileNotFoundError:
                warnings.warn(
                    (
                        f"Attempt to change to {run_directory}, but it is not found!\n"
                        f"Will attempt to run from {os.getcwd()}. It may fail!"
                    ),
                    category=UserWarning,
                )
        self._use_mpi: bool = use_mpi
        self._row_ids: Optional[RowIds] = row_ids

        affinity: Set[int] = os.sched_getaffinity(0)
        # By convention, the Executor takes the minimum core on this node.
        # Task gets everything else. If we only have 1 core here then out of luck
        # and cannot set new affinities without issues
        if len(affinity) > 1:
            executor_affinity: Set[int] = {min(affinity)}
            task_affinity: Set[int] = affinity - executor_affinity
            os.sched_setaffinity(0, task_affinity)

        # We can use `stdin` to receive Message's back from the Executor
        # Set non-blocking otherwise reads would hang
        os.set_blocking(sys.stdin.fileno(), False)

    def run(self) -> None:
        """Calls the analysis routines and any pre/post task functions.

        This method is part of the public API and should not need to be modified
        in any subclasses.
        """
        self._signal_start()
        self._pre_run()
        self._run()
        self._post_run()
        self._signal_result()

    @abstractmethod
    def _run(self) -> None:
        """Actual analysis to run. Overridden by subclasses.

        Separating the calling API from the implementation allows `run` to
        have pre and post task functionality embedded easily into a single
        function call.
        """
        ...

    def _pre_run(self) -> None:
        """Code to run BEFORE the main analysis takes place.

        This function may, or may not, be employed by subclasses.
        """
        ...

    def _post_run(self) -> None:
        """Code to run AFTER the main analysis takes place.

        This function may, or may not, be employed by subclasses.
        """
        ...

    @property
    def result(self) -> TaskResult:
        """TaskResult: Read-only Task Result information."""
        return self._result

    def __call__(self) -> None:
        self.run()

    def _signal_start(self) -> None:
        """Send the signal that the Task will begin shortly."""
        msg_contents: Union[TaskParameters, TaskParametersDBReference]
        if self._row_ids is not None:
            msg_contents = TaskParametersDBReference(
                db_dir=self._task_parameters.lute_config.work_dir, row_ids=self._row_ids
            )
        else:
            msg_contents = self._task_parameters
        start_msg: Message = Message(contents=msg_contents, signal="TASK_STARTED")
        self._result.task_status = TaskStatus.RUNNING
        if self._use_mpi:
            from mpi4py import MPI

            comm: MPI.Intracomm = MPI.COMM_WORLD
            rank: int = comm.Get_rank()
            comm.Barrier()
            if rank == 0:
                self._report_to_executor(start_msg)

                # In first-party, different environment Tasks the bootstrap process
                # will create a file to share RowIds. The path is stored in an env var
                # and it can now be deleted safely since we're past the Barrier above
                bootstrap_file: Optional[str] = os.getenv("LUTE_BOOTSTRAP_FILE")
                if bootstrap_file is not None and os.path.exists(bootstrap_file):
                    os.remove(bootstrap_file)
        else:
            self._report_to_executor(start_msg)

        # We stop process here so Executor can do any tasklet work if needed
        if os.getenv("LUTE_CONFIGPATH") is not None:
            # Guard w/ environment variable that is set only by Executor - don't
            # SIGSTOP if Task is running without Executor
            if self._use_mpi:
                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()
                if rank == 0:
                    os.kill(os.getppid(), signal.SIGSTOP)
                comm.Barrier()
            else:
                os.kill(os.getpid(), signal.SIGSTOP)

        # Upon resuming, we will also send some metadata that can be logged
        # for inter-Task communication later
        hostnames: List[str] = []
        metadata: Dict[str, List[str]] = {}
        hostname: str
        if self._use_mpi:
            comm = MPI.COMM_WORLD
            hostname = MPI.Get_processor_name()
            hostnames = comm.allgather(hostname)
        else:
            hostname = socket.gethostname()
            hostnames.append(hostname)
        metadata["task_hostnames"] = hostnames
        self.publish_metadata(metadata=metadata)

    def _signal_result(self) -> None:
        """Send the signal that results are ready along with the results."""
        signal: str = "TASK_RESULT"
        results_msg: Message = Message(contents=self.result, signal=signal)
        if self._use_mpi:
            from mpi4py import MPI

            comm: MPI.Intracomm = MPI.COMM_WORLD
            rank: int = comm.Get_rank()
            comm.Barrier()
            if rank == 0:
                self._report_to_executor(results_msg)
        else:
            self._report_to_executor(results_msg)
        time.sleep(0.1)

    def _report_to_executor(self, msg: MessageType) -> None:
        """Send a message to the Executor.

        Details of `Communicator` choice are hidden from the caller. This
        method may be overriden by subclasses with specialized functionality.

        Args:
            msg (Message): The message object to send.
        """
        communicator: Communicator
        if isinstance(msg.contents, str) or msg.contents is None:
            communicator = PipeCommunicator()
        else:
            communicator = SocketCommunicator()

        communicator.delayed_setup()
        communicator.write(msg)
        communicator.clear_communicator()

    def task_request(
        self, for_task: str, request: Any, wait_for_resp: bool = True
    ) -> None:
        """Send a message with a request for another Task.

        The other `Task` may or may not be running. The request is sent to the
        Executor that is managing this Task. A response can be waited for, or
        it can be checked for later.

        Args:
            for_task (str): The name of the `Task` the request should be routed to.
                This is the `Task` (NOT **managed** Task) name. It can be running in
                any environment etc. The details of how the request is routed are
                not relevant to the Task layer.

            request (Any): The request to be sent. It must be JSON serializable.

            wait_for_resp (bool): Whether to block on the response. If False, the
                Task can check at a later time using the `check_request_response`
                method.

        Returns:
            resp
        """
        req: TaskRequest = TaskRequest(
            request=request, for_task=for_task, for_manager=False
        )
        req_msg: TaskRequestMessage = TaskRequestMessage(contents=req)
        self._report_to_executor(msg=req_msg)
        if wait_for_resp:
            communicator: PipeCommunicator = PipeCommunicator()
            # Ignoring for now...
            # resp_msg: Message = communicator.read()
            communicator.read()

    def get_running_tasks(self) -> Message:
        """Send a message with a request to know any other running Tasks.

        Args:
            for_task (str): The name of the `Task` the request should be routed to.
                This is the `Task` (NOT **managed** Task) name. It can be running in
                any environment etc. The details of how the request is routed are
                not relevant to the Task layer.

            request (Any): The request to be sent. It must be JSON serializable.

            wait_for_resp (bool): Whether to block on the response. If False, the
                Task can check at a later time using the `check_request_response`
                method.

        Returns:
            resp (Message): Response from the workflow manager (if any).
        """
        req: TaskRequest = TaskRequest(
            request="RUNNING_TASKS", for_task=None, for_manager=True
        )
        req_msg: TaskRequestMessage = TaskRequestMessage(contents=req)
        self._report_to_executor(msg=req_msg)

        communicator: PipeCommunicator = PipeCommunicator()
        return communicator.read(wait=2)

    def publish_metadata(self, metadata: Dict[str, Any]) -> None:
        """Send a message containing metadata that can be accessed by other Tasks.

        Args:
            metadata (Dict[str, Any]): A dictionary of key/value pairs that contains
                data that other Tasks running in parallel could find useful.
        """
        meta_msg: TaskMetadataMessage = TaskMetadataMessage(contents=metadata)
        self._report_to_executor(msg=meta_msg)

    def clean_up_timeout(self) -> None:
        """Perform any necessary cleanup actions before exit if timing out."""
        ...


class ThirdPartyTask(Task):
    """A `Task` interface to analysis with binary executables."""

    def __init__(self, *, params: ThirdPartyParameters) -> None:
        """Initialize a Task.

        Args:
            params (TaskParameters): Parameters needed to properly configure
                the analysis task. `Task`s of this type MUST include the name
                of a binary to run and any arguments which should be passed to
                it (as would be done via command line). The binary is included
                with the parameter `executable`. All other parameter names are
                assumed to be the long/extended names of the flag passed on the
                command line by default:
                    * `arg_name = 3` is converted to `--arg_name 3`
                Positional arguments can be included with `p_argN` where `N` is
                any integer:
                    * `p_arg1 = 3` is converted to `3`

                Note that it is NOT recommended to rely on this default behaviour
                as command-line arguments can be passed in many ways. Refer to
                the dcoumentation at
                https://slac-lcls.github.io/lute/tutorial/new_task/
                under "Speciyfing a TaskParameters Model for your Task" for more
                information on how to control parameter parsing from within your
                TaskParameters model definition.
        """
        super().__init__(params=params)
        if not hasattr(self._task_parameters, "executable"):
            raise RuntimeError("ThirdPartyTask must have executable defined!")
        self._cmd = self._task_parameters.executable
        self._args_list: List[str] = [self._cmd]
        self._template_context: Dict[str, Any] = {}

    def _add_to_jinja_context(self, param_name: str, value: Any) -> None:
        """Store a parameter as a Jinja template variable.

        Variables are stored in a dictionary which is used to fill in a
        premade Jinja template for a third party configuration file.

        Args:
            param_name (str): Name to store the variable as. This should be
                the name defined in the corresponding pydantic model. This name
                MUST match the name used in the Jinja Template!
            value (Any): The value to store. If possible, large chunks of the
                template should be represented as a single dictionary for
                simplicity; however, any type can be stored as needed.
        """
        context_update: Dict[str, Any] = {param_name: value}
        if __debug__:
            msg: Message = Message(contents=f"TemplateParameters: {context_update}")
            self._report_to_executor(msg)
        self._template_context.update(context_update)

    def _template_to_config_file(self) -> None:
        """Convert a template file into a valid configuration file.

        Uses Jinja to fill in a provided template file with variables supplied
        through the LUTE config file. This facilitates parameter modification
        for third party tasks which use a separate configuration, in addition
        to, or instead of, command-line arguments.
        """
        from jinja2 import Environment, FileSystemLoader, Template

        if not hasattr(self._task_parameters, "lute_template_cfg"):
            raise RuntimeError("Missing lute_template_cfg! Cannot compile template!")

        out_file: str = self._task_parameters.lute_template_cfg.output_path
        template_name: str = self._task_parameters.lute_template_cfg.template_name

        lute_path: Optional[str] = os.getenv("LUTE_PATH")
        template_dir: str
        if lute_path is None:
            warnings.warn(
                "LUTE_PATH is None in Task process! Using relative path for templates!",
                category=UserWarning,
            )
            template_dir = "../../config/templates"
        else:
            py_ver: str = f"python{sys.version_info.major}.{sys.version_info.minor}"
            template_dir = f"{lute_path}/lib/{py_ver}/site-packages/config/templates"
            if not os.path.exists(template_dir):
                # Did not install and running from clone of repo
                template_dir = f"{lute_path}/config/templates"
        environment: Environment = Environment(loader=FileSystemLoader(template_dir))
        template: Template = environment.get_template(template_name)

        with open(out_file, "w", encoding="utf-8") as cfg_out:
            cfg_out.write(template.render(self._template_context))

    def _pre_run(self) -> None:
        """Parse the parameters into an appropriate argument list.

        Arguments are identified by a `flag_type` attribute, defined in the
        pydantic model, which indicates how to pass the parameter and its
        argument on the command-line. This method parses flag:value pairs
        into an appropriate list to be used to call the executable.

        Note:
        ThirdPartyParameter objects are returned by custom model validators.
        Objects of this type are assumed to be used for a templated config
        file used by the third party executable for configuration. The parsing
        of these parameters is performed separately by a template file used as
        an input to Jinja. This method solely identifies the necessary objects
        and passes them all along. Refer to the template files and pydantic
        models for more information on how these parameters are defined and
        identified.
        """
        super()._pre_run()
        full_schema: Dict[str, Any] = self._task_parameters.schema()
        short_flags_use_eq: bool
        long_flags_use_eq: bool
        if hasattr(self._task_parameters.Config, "short_flags_use_eq"):
            short_flags_use_eq = self._task_parameters.Config.short_flags_use_eq
        else:
            short_flags_use_eq = False

        if hasattr(self._task_parameters.Config, "long_flags_use_eq"):
            long_flags_use_eq = self._task_parameters.Config.long_flags_use_eq
        else:
            long_flags_use_eq = False
        for param, value in self._task_parameters.dict().items():
            # Clunky test with __dict__[param] because compound model-types are
            # converted to `dict`. E.g. type(value) = dict not AnalysisHeader
            if (
                param in ("executable", "_result_from_params")
                or value is None  # Cannot have empty values in argument list for execvp
                or value == ""  # But do want to include, e.g. 0
                or isinstance(self._task_parameters.__dict__[param], TemplateConfig)
                or isinstance(self._task_parameters.__dict__[param], AnalysisHeader)
            ):
                continue
            if isinstance(self._task_parameters.__dict__[param], TemplateParameters):
                # TemplateParameters objects have a single parameter `params`
                self._add_to_jinja_context(param_name=param, value=value.params)
                continue

            param_attributes: Dict[str, Any] = full_schema["properties"][param]
            # Some model params do not match the commnad-line parameter names
            param_repr: str
            if "rename_param" in param_attributes:
                param_repr = param_attributes["rename_param"]
            else:
                param_repr = param
            if "flag_type" in param_attributes:
                flag: str = param_attributes["flag_type"]
                if flag:
                    # "-" or "--" flags
                    if flag == "--" and isinstance(value, bool) and not value:
                        continue
                    constructed_flag: str = f"{flag}{param_repr}"
                    if flag == "--" and isinstance(value, bool) and value:
                        # On/off flag, e.g. something like --verbose: No Arg
                        self._args_list.append(f"{constructed_flag}")
                        continue
                    if (flag == "-" and short_flags_use_eq) or (
                        flag == "--" and long_flags_use_eq
                    ):  # Must come after above check! Otherwise you get --param=True
                        # Flags following --param=value or -param=value
                        constructed_flag = f"{constructed_flag}={value}"
                        self._args_list.append(f"{constructed_flag}")
                        continue
                    self._args_list.append(f"{constructed_flag}")
            else:
                warnings.warn(
                    (
                        f"Model parameters should be defined using Field(...,flag_type='')"
                        f" in the future.  Parameter: {param}"
                    ),
                    category=PendingDeprecationWarning,
                )
                if len(param) == 1:  # Single-dash flags
                    if short_flags_use_eq:
                        self._args_list.append(f"-{param_repr}={value}")
                        continue
                    self._args_list.append(f"-{param_repr}")
                elif "p_arg" in param:  # Positional arguments
                    pass
                else:  # Double-dash flags
                    if isinstance(value, bool) and not value:
                        continue
                    if long_flags_use_eq:
                        self._args_list.append(f"--{param_repr}={value}")
                        continue
                    self._args_list.append(f"--{param_repr}")
                    if isinstance(value, bool) and value:
                        continue
            if isinstance(value, str) and " " in value:
                for val in value.split():
                    self._args_list.append(f"{val}")
            else:
                self._args_list.append(f"{value}")
        if (
            hasattr(self._task_parameters, "lute_template_cfg")
            and self._template_context
        ):
            self._template_to_config_file()

    def _run(self) -> None:
        """Execute the new program by replacing the current process."""
        if __debug__:
            time.sleep(0.1)
            msg: Message = Message(contents=self._formatted_command())
            self._report_to_executor(msg)
        LUTE_DEBUG_EXIT("LUTE_DEBUG_BEFORE_TPP_EXEC")
        task_env: Dict[str, str] = self._setup_env()
        task_env.update(self._non_slurm_mpi_mods(task_env=task_env))
        sys.stdout.flush()
        sys.stderr.flush()
        os.execvpe(file=self._cmd, args=self._args_list, env=task_env)

    def _formatted_command(self) -> str:
        """Returns the command as it would passed on the command-line."""
        formatted_cmd: str = "".join(f"{arg} " for arg in self._args_list)
        return formatted_cmd

    def _signal_start(self) -> None:
        """Override start signal method to switch communication methods."""
        super()._signal_start()
        time.sleep(0.05)
        signal: str = "NO_PICKLE_MODE"
        msg: Message = Message(signal=signal)
        self._report_to_executor(msg)

    def _non_slurm_mpi_mods(self, task_env: Dict[str, str]) -> Dict[str, str]:
        """Configure MPI to use a hostfile for resource determination.

        In the event the ThirdPartyTask is calling `mpirun` or similar, the automated
        resource determination can fail inside the SLURM job depending on the
        environment and/or MPI version. By using a hostfile, the resources can
        always be determined. The execution layer sets this up before hand
        and defines the `LUTE_MPI_HOSTFILE_PATH` environment variable.

        Args:
            task_env (Dict[str, str]): The new Task environment. It should already
                be processed (e.g. LUTE_TENV_ removal). It will be checked for the
                `LUTE_MPI_HOSTFILE_PATH` environment variable.

        Returns:
            mpi_updates (Dict[str, str]): A set of new environment variables which
                should be added to the Task environment dictionary for MPI
                configuration.
        """
        hostfile_path: Optional[str] = task_env.get("LUTE_MPI_HOSTFILE_PATH")
        if hostfile_path is None:
            return {}

        # You may find some info here: https://docs.open-mpi.org/en/v5.0.7/mca.html
        # BUT - things change between versions. Tread carefully...
        mca_config: Dict[str, str] = {
            "PRTE_MCA_ras_slurm_priority": "0",
            "PRTE_MCA_plm": "slurm",
            "PRTE_MCA_plm_slurm_args": "--overlap --export=ALL",
            "PRTE_MCA_prte_default_hostfile": hostfile_path,
            "OMPI_MCA_orte_default_hostfile": hostfile_path,
            "MPIEXEC_HOSTFILE": hostfile_path,  # Not sure if this exists/is needed?
            # Can turn this on for debugging - display MPI's discovered resources
            # "PRTE_MCA_rmaps_base_display_allocation": "1",
            # Can turn this on for debugging - report how ranks are bound
            # "OMPI_MCA_hwloc_base_report_bindings": "1",
            "PRTE_MCA_plm_rsh_pass_path": "1",
        }

        return mca_config

    def _setup_env(self) -> Dict[str, str]:
        new_env: Dict[str, str] = {}
        mpi_hostfile: str = ""
        found_tenv: bool = False
        for key, value in os.environ.items():
            if "LUTE_TENV_" in key:
                # Set if using a custom environment
                found_tenv = True
                new_key: str = key[10:]
                new_env[new_key] = value
            # SLURM vars and the hostfile are needed for MPI
            elif "SLURM_" in key:
                new_env[key] = value
            elif key == "LUTE_MPI_HOSTFILE_PATH":
                mpi_hostfile = value
        if not found_tenv:
            # No shell script sourced - will use the base environment
            # The SLURM_ keys will be in this copy.
            new_env = os.environ.copy()

        if mpi_hostfile:
            new_env["LUTE_MPI_HOSTFILE_PATH"] = mpi_hostfile

        return new_env
