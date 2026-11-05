"""Classes and utilities for communication between Executors and subprocesses.

Communicators manage message passing and parsing between subprocesses. They
maintain a limited public interface of "read" and "write" operations. Behind
this interface the methods of communication vary from serialization across
pipes to Unix sockets, etc. All communicators pass a single object called a
"Message" which contains an arbitrary "contents" field as well as an optional
"signal" field.


Classes:
    Party: Enum describing whether Communicator is on Task-side or Executor-side.

    Message: A dataclass used for passing information from Task to Executor.

    Communicator: Abstract base class for Communicator types.

    PipeCommunicator: Manages communication between Task and Executor via pipes
        (stderr and stdout).

    SocketCommunicator: Manages communication using sockets, either raw or using
        zmq. Supports both TCP and Unix sockets.
"""

from __future__ import annotations

__all__ = [
    "Party",
    "Message",
    "Communicator",
    "PipeCommunicator",
    "LUTE_SIGNALS",
    "SocketCommunicator",
]
__author__ = "Gabriel Dorlhiac"

import io
import logging
import os
import pickle
import select
import socket
import subprocess
import sys
import threading
import time
import queue
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    ClassVar,
    Dict,
    IO,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)
from typing_extensions import Self

LUTE_SIGNALS: Set[str] = {
    "NO_PICKLE_MODE",
    "TASK_STARTED",
    "TASK_FAILED",
    "TASK_STOPPED",
    "TASK_DONE",
    "TASK_CANCELLED",
    "TASK_RESULT",
    "TASK_LOG",
    "TASK_REQUEST",
    "TASK_RESPONSE",
    "TASK_METADATA",
}

if __debug__:
    warnings.simplefilter("default")
    os.environ["PYTHONWARNINGS"] = "default"
    logging.basicConfig(level=logging.DEBUG)
    logging.captureWarnings(True)
else:
    logging.basicConfig(level=logging.INFO)
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

logger: logging.Logger = logging.getLogger(__name__)
zmq_env: Optional[str] = os.getenv("LUTE_USE_ZMQ")
USE_ZMQ: bool
if zmq_env is None:
    logger.debug("Preference of ZMQ usage not specified. Defaulting to using ZMQ.")
    try:
        import zmq

        USE_ZMQ = True
    except ModuleNotFoundError:
        logger.warning("ZMQ not found. Will use `socket` implementation.")
        USE_ZMQ = False
elif zmq_env == "0":
    logger.debug("Requested use of `socket` over ZMQ for IPC.")
    USE_ZMQ = False
else:
    logger.debug("Requested use of ZMQ for IPC.")
    USE_ZMQ = True


class Party(Enum):
    """Identifier for which party (side/end) is using a communicator.

    For some types of communication streams there may be different interfaces
    depending on which side of the communicator you are on. This enum is used
    by the communicator to determine which interface to use.
    """

    TASK = 0
    """
    The Task (client) side.
    """
    EXECUTOR = 1
    """
    The Executor (server) side.
    """


ContentsType = TypeVar("ContentsType")
SignalType = TypeVar("SignalType")


class MessageType(Protocol[ContentsType, SignalType]):
    contents: ContentsType
    signal: SignalType


@dataclass
class Message:
    contents: Optional[Any] = None
    signal: Optional[str] = None


@dataclass
class TaskRequest:
    request: Any
    for_task: Optional[str] = None
    for_manager: bool = False


@dataclass
class TaskResponse:
    response: Optional[Any] = None
    for_task: Optional[str] = None


@dataclass
class TaskRequestMessage:
    contents: TaskRequest
    signal: str = "TASK_REQUEST"


@dataclass
class TaskResponseMessage:
    contents: TaskResponse
    signal: str = "TASK_RESPONSE"


@dataclass
class TaskMetadataMessage:
    contents: Dict[str, Any]
    signal: str = "TASK_METADATA"


class Communicator(ABC):
    def __init__(self, party: Party = Party.TASK, use_pickle: bool = True) -> None:
        """Abstract Base Class for IPC Communicator objects.

        Args:
            party (Party): Which object (side/process) the Communicator is
                managing IPC for. I.e., is this the "Task" or "Executor" side.
            use_pickle (bool): Whether to serialize data using pickle prior to
                sending it.
        """
        self._party = party
        self._use_pickle = use_pickle
        self.desc = "Communicator abstract base class."

    @abstractmethod
    def read(
        self, proc: Optional[subprocess.Popen], wait: Optional[float] = None
    ) -> Message:
        """Method for reading data through the communication mechanism."""
        ...

    @abstractmethod
    def write(self, msg: MessageType, proc: Optional[subprocess.Popen] = None) -> None:
        """Method for sending data through the communication mechanism."""
        ...

    def __str__(self) -> str:
        name: str = str(type(self)).split("'")[1].split(".")[-1]
        return f"{name}: {self.desc}"

    def __repr__(self) -> str:
        return self.__str__()

    def __enter__(self) -> Self:
        return self

    def __exit__(self) -> None: ...

    @property
    def has_messages(self) -> bool:
        """Whether the Communicator has remaining messages.

        The precise method for determining whether there are remaining messages
        will depend on the specific Communicator sub-class.
        """
        return False

    def stage_communicator(self):
        """Alternative method for staging outside of context manager."""
        self.__enter__()

    def clear_communicator(self):
        """Alternative exit method outside of context manager."""
        self.__exit__()

    def delayed_setup(self):
        """Any setup that should be done later than init."""
        ...


class PipeCommunicator(Communicator):
    """Provides communication through pipes over stderr/stdout.

    The implementation of this communicator has reading and writing ocurring
    on stderr and stdout. In general the `Task` will be writing while the
    `Executor` will be reading. `stderr` is used for sending signals.
    """

    def __init__(self, party: Party = Party.TASK, use_pickle: bool = True) -> None:
        """IPC through pipes.

        Arbitrary objects may be transmitted using pickle to serialize the data.
        If pickle is not used

        Args:
            party (Party): Which object (side/process) the Communicator is
                managing IPC for. I.e., is this the "Task" or "Executor" side.
            use_pickle (bool): Whether to serialize data using Pickle prior to
                sending it. If False, data is assumed to be text whi
        """
        super().__init__(party=party, use_pickle=use_pickle)
        self.desc = "Communicates through stderr and stdout using pickle."

    @property
    def use_pickle(self) -> bool:
        return self._use_pickle

    @use_pickle.setter
    def use_pickle(self, use_pickle: bool) -> None:
        self._use_pickle = use_pickle

    def read(
        self, proc: Optional[subprocess.Popen] = None, wait: Optional[float] = None
    ) -> Message:
        """Read from stdout and stderr (Executor) or stdin (Task).

        Args:
            proc (subprocess.Popen): The process to read from (Executor side).

            wait (Optional[float]): Optionally wait for data for the specified time.
                By default all operations are non-blocking, by setting wait to some
                non-zero value, the Communicator will block for that amount of time.

        Returns:
            msg (Message): The message read, containing contents and signal.
        """
        signal: Optional[str]
        contents: Optional[Any]
        raw_signal: Optional[bytes]
        raw_contents: Optional[bytes]

        if wait is not None:
            wait_on: List[IO]
            if self._party == Party.EXECUTOR:
                assert proc is not None
                wait_on = [fd for fd in [proc.stderr, proc.stdin] if fd is not None]
            else:
                wait_on = [sys.stdin]
            # We're ignoring the output here to not rewrite the rest of the code
            select.select(wait_on, [], [], wait)

        if self._party == Party.EXECUTOR:
            assert proc is not None
            raw_signal = proc.stderr.read() if proc.stderr is not None else None
            raw_contents = proc.stdout.read() if proc.stdout is not None else None
        else:
            # Task side - reading from stdin which Executor writes to.
            # Do not currently have a split like Executor using stderr/stdout
            raw_signal = None
            try:
                raw_contents = sys.stdin.buffer.read()
            except (AttributeError, io.UnsupportedOperation):
                # Fallback if stdin is not a buffer or closed
                raw_contents = None

        if raw_signal is not None:
            try:
                signal = raw_signal.decode()
            except UnicodeDecodeError:
                logger.debug(
                    "PipeCommunicator - Stderr signal contains non-UTF-8 bytes. "
                    "Decoding with errors='replace'."
                )
                signal = raw_signal.decode(errors="replace")
        else:
            signal = None

        if raw_contents:
            if self._use_pickle:
                try:
                    contents = pickle.loads(raw_contents)
                    if self._party == Party.TASK and isinstance(contents, Message):
                        # Executor writes a whole Message object to stdin
                        signal = contents.signal
                        contents = contents.contents
                except (
                    pickle.UnpicklingError,
                    ValueError,
                    EOFError,
                    ModuleNotFoundError,
                ):
                    logger.debug("PipeCommunicator - Set _use_pickle=False")
                    self._use_pickle = False
                    contents = self._safe_unpickle_decode(raw_contents)
            else:
                try:
                    contents = raw_contents.decode()
                except UnicodeDecodeError:
                    logger.debug("PipeCommunicator - Set _use_pickle=True")
                    self._use_pickle = True
                    contents = self._safe_unpickle_decode(raw_contents)
        else:
            contents = None

        if self._party == Party.EXECUTOR and signal and signal not in LUTE_SIGNALS:
            # Some tasks write on stderr
            # If the signal channel has "non-signal" info, add it to
            # contents
            if not contents:
                contents = f"({signal})"
            else:
                contents = f"{contents} ({signal})"
            signal = None

        return Message(contents=contents, signal=signal)

    def _safe_unpickle_decode(self, maybe_mixed: bytes) -> Optional[str]:
        """This method is used to unpickle and/or decode a bytes object.

        It attempts to handle cases where contents can be mixed, i.e., part of
        the message must be decoded and the other part unpickled. It handles
        only two-way splits. If there are more complex arrangements such as:
        <pickled>:<unpickled>:<pickled> etc, it will give up.

        The simpler two way splits are unlikely to occur in normal usage. They
        may arise when debugging if, e.g., `print` statements are mixed with the
        usage of the `_report_to_executor` method.

        Note that this method works because ONLY text data is assumed to be
        sent via the pipes. The method needs to be revised to handle non-text
        data if the `Task` is modified to also send that via PipeCommunicator.
        The use of pickle is supported to provide for this option if it is
        necessary. It may be deprecated in the future.

        Be careful when making changes. This method has seemingly redundant
        checks because unpickling will not throw an error if a full object can
        be retrieved. That is, the library will ignore extraneous bytes. This
        method attempts to retrieve that information if the pickled data comes
        first in the stream.

        Args:
            maybe_mixed (bytes): A bytes object which could require unpickling,
                decoding, or both.

        Returns:
            contents (Optional[str]): The unpickled/decoded contents if possible.
                Otherwise, None.
        """
        contents: Optional[str]
        try:
            contents = pickle.loads(maybe_mixed)
            repickled: bytes = pickle.dumps(contents)
            if len(repickled) < len(maybe_mixed):
                # Successful unpickling, but pickle stops even if there are more bytes
                try:
                    additional_data: str = maybe_mixed[len(repickled) :].decode()
                    contents = f"{contents}{additional_data}"
                except UnicodeDecodeError:
                    # Can't decode the bytes left by pickle, so they are lost
                    missing_bytes: int = len(maybe_mixed) - len(repickled)
                    logger.debug(
                        f"PipeCommunicator has truncated message. Unable to retrieve {missing_bytes} bytes."
                    )
        except (pickle.UnpicklingError, ValueError, EOFError, ModuleNotFoundError):
            # Pickle may also throw a ValueError, e.g. this bytes: b"Found! \n"
            # Pickle may also throw an EOFError, eg. this bytes: b"F0\n"
            # Pickle may also throw a ModuleNotFoundError, e.g. this bytes: b"co\nco\n"
            try:
                contents = maybe_mixed.decode()
            except UnicodeDecodeError as err2:
                try:
                    contents = maybe_mixed[: err2.start].decode()
                    contents = f"{contents}{pickle.loads(maybe_mixed[err2.start:])}"
                except Exception as err3:
                    logger.debug(
                        f"PipeCommunicator unable to decode/parse data! {err3}"
                    )
                    contents = None
        return contents

    def write(self, msg: MessageType, proc: Optional[subprocess.Popen] = None) -> None:
        """Write to stdout and stderr (Task) or proc.stdin (Executor).

         The signal component is sent to `stderr` while the contents of the
         Message are sent to `stdout`.

        Args:
            msg (Message): The Message to send.
            proc (subprocess.Popen, optional): The process to write to (Executor side).
        """
        if self._party == Party.EXECUTOR:
            assert proc is not None
            if proc.stdin is None:
                logger.error("Attempted to write to process with no stdin pipe!")
                return

            if self._use_pickle:
                # We don't have separate signal/content channels for reverse yet
                # Just send the whole message object
                data: bytes = pickle.dumps(msg)
                proc.stdin.write(data)
                proc.stdin.flush()
            else:
                if not isinstance(msg.contents, str) and msg.contents is not None:
                    raise ValueError(f"Cannot send {type(msg.contents)} without pickle")
                data_str: str = str(msg.contents or "")
                proc.stdin.write(data_str)
                proc.stdin.flush()
            return

        # Task side
        if self._use_pickle:
            signal: bytes
            if msg.signal:
                signal = msg.signal.encode()
            else:
                signal = b""

            contents: bytes = pickle.dumps(msg.contents)

            sys.stderr.buffer.write(signal)
            sys.stdout.buffer.write(contents)

            sys.stderr.buffer.flush()
            sys.stdout.buffer.flush()
        else:
            raw_signal: str
            if msg.signal:
                raw_signal = msg.signal
            else:
                raw_signal = ""

            raw_contents: str
            if isinstance(msg.contents, str):
                raw_contents = msg.contents
            elif msg.contents is None:
                raw_contents = ""
            else:
                raise ValueError(
                    f"Cannot send msg contents of type: {type(msg.contents)} when not using pickle!"
                )
            sys.stderr.write(raw_signal)
            sys.stdout.write(raw_contents)
            sys.stderr.flush()
            sys.stdout.flush()


class SocketCommunicator(Communicator):
    """Provides communication over Unix or TCP sockets.

    Communication is provided either using sockets with the Python socket library
    or using ZMQ. The choice of implementation is controlled by the global bool
    `USE_ZMQ`.

    Whether to use TCP or Unix sockets is controlled by the environment:
                           `LUTE_USE_TCP=1`
    If defined, TCP sockets will be used, otherwise Unix sockets will be used.

    Regardless of socket type, the environment variable
                      `LUTE_EXECUTOR_HOST=<hostname>`
    will be defined by the Executor-side Communicator.


    For TCP sockets:
    The Executor-side Communicator should be run first and will bind to all
    interfaces on the port determined by the environment variable:
                            `LUTE_PORT=###`
    If no port is defined, a port scan will be performed and the Executor-side
    Communicator will bind the first one available from a random selection. It
    will then define the environment variable so the Task-side can pick it up.

    For Unix sockets:
    The path to the Unix socket is defined by the environment variable:
                      `LUTE_SOCKET=/path/to/socket`
    This class assumes proper permissions and that this above environment
    variable has been defined. The `Task` is configured as what would commonly
    be referred to as the `client`, while the `Executor` is configured as the
    server.

    If the Task process is run on a different machine than the Executor, the
    Task-side Communicator will open a ssh-tunnel to forward traffic from a local
    Unix socket to the Executor Unix socket. Opening of the tunnel relies on the
    environment variable:
                      `LUTE_EXECUTOR_HOST=<hostname>`
    to determine the Executor's host. This variable should be defined by the
    Executor and passed to the Task process automatically, but it can also be
    defined manually if launching the Task process separately. The Task will use
    the local socket `<LUTE_SOCKET>.task{##}`. Multiple local sockets may be
    created. Currently, it is assumed that the user is identical on both the Task
    machine and Executor machine.
    """

    ACCEPT_TIMEOUT: ClassVar[float] = 0.01
    """
    Maximum time to wait to accept connections. Used by Executor-side.
    """
    MSG_HEAD: ClassVar[bytes] = b"MSG"
    """
    Start signal of a message. The end of a message is indicated by MSG_HEAD[::-1].
    """
    MSG_SEP: ClassVar[bytes] = b";;;"
    """
    Separator for parts of a message. Messages have a start, length, message and end.
    """

    def __init__(self, party: Party = Party.TASK, use_pickle: bool = True) -> None:
        """IPC over a TCP or Unix socket.

        Unlike with the PipeCommunicator, pickle is always used to send data
        through the socket.

        Args:
            party (Party): Which object (side/process) the Communicator is
                managing IPC for. I.e., is this the "Task" or "Executor" side.

            use_pickle (bool): Whether to use pickle. Always True currently,
                passing False does not change behaviour.
        """
        super().__init__(party=party, use_pickle=use_pickle)

    def delayed_setup(self) -> None:
        """Delays the creation of socket objects.

        The Executor initializes the Communicator when it is created. Since
        all Executors are created and available at once we want to delay
        acquisition of socket resources until a single Executor is ready
        to use them.
        """
        self._data_socket: Union[socket.socket, zmq.sugar.socket.Socket]
        self.desc: str
        use_tcp: Optional[str] = os.getenv("LUTE_USE_TCP")
        sock_type: str
        if use_tcp is not None:
            sock_type = "TCP"
        else:
            sock_type = "Unix"
        if USE_ZMQ:
            self.desc = f"Communicates using ZMQ through {sock_type} sockets."
            self._context: zmq.Context = zmq.Context()
            self._data_socket = self._create_socket_zmq()
        else:
            self.desc = f"Communicates through {sock_type} sockets."
            self._data_socket = self._create_socket_raw()
            self._data_socket.settimeout(SocketCommunicator.ACCEPT_TIMEOUT)

        if self._party == Party.EXECUTOR:
            # Executor created first so we can define the hostname env variable
            os.environ["LUTE_EXECUTOR_HOST"] = socket.gethostname()
            # Setup reader thread
            self._reader_thread: threading.Thread = threading.Thread(
                target=self._read_socket
            )
            self._msg_queue: queue.Queue = queue.Queue()
            self._partial_msg: Optional[bytes] = None
            self._stop_thread: bool = False
            self._reader_thread.start()
        else:
            # Only used by Party.TASK
            self._use_ssh_tunnel: bool = False
            self._ssh_proc: Optional[subprocess.Popen] = None
            self._local_socket_path: Optional[str] = None

    # Read
    ############################################################################

    def read(
        self, proc: Optional[subprocess.Popen] = None, wait: Optional[float] = None
    ) -> Message:
        """Return a message from the queue if available.

        Socket(s) are continuously monitored, and read from when new data is
        available.

        Args:
            proc (subprocess.Popen): The process to read from. Provided for
                compatibility with other Communicator subtypes. Is ignored.

        Returns:
             msg (Message): The message read, containing contents and signal.
        """
        msg: Message
        try:
            msg = self._msg_queue.get(timeout=SocketCommunicator.ACCEPT_TIMEOUT)
        except queue.Empty:
            msg = Message()

        return msg

    def _read_socket(self) -> None:
        """Read data from a socket.

        Socket(s) are continuously monitored, and read from when new data is
        available.

        Calls an underlying method for either raw sockets or ZMQ.
        """

        while True:
            if self._stop_thread:
                logger.debug("Stopping socket reader thread.")
                break
            if USE_ZMQ:
                self._read_socket_zmq()
            else:
                self._read_socket_raw()

    def _read_socket_raw(self) -> None:
        """Read data from a socket.

        Raw socket implementation for the reader thread.
        """
        assert isinstance(self._data_socket, socket.socket)
        connection: socket.socket
        _: Union[str, Tuple[str, int]]
        try:
            connection, _ = self._data_socket.accept()
            full_data: bytes = b""
            while True:
                data: bytes = connection.recv(8192)
                if data:
                    full_data += data
                else:
                    break
            connection.close()
            self._unpack_messages(full_data)
        except socket.timeout:
            pass

    def _read_socket_zmq(self) -> None:
        """Read data from a socket.

        ZMQ implementation for the reader thread.
        """
        try:
            full_data: bytes = self._data_socket.recv(0)
            self._unpack_messages(full_data)
        except zmq.ZMQError:
            pass

    def _unpack_messages(self, data: bytes) -> None:
        """Unpacks a byte stream into individual messages.

        Messages are encoded in the following format:
                 <HEAD><SEP><len(msg)><SEP><msg><SEP><HEAD[::-1]>
        The items between <> are replaced as follows:
            - <HEAD>: A start marker
            - <SEP>: A separator for components of the message
            - <len(msg)>: The length of the message payload in bytes.
            - <msg>: The message payload in bytes
            - <HEAD[::-1]>: The start marker in reverse to indicate the end.

        Partial messages (a series of bytes which cannot be converted to a full
        message) are stored for later. An attempt is made to reconstruct the
        message with the next call to this method.

        Args:
            data (bytes): A raw byte stream containing anywhere from a partial
                message to multiple full messages.
        """
        msg: Message
        working_data: bytes
        if self._partial_msg:
            # Concatenate the previous partial message to the beginning
            working_data = self._partial_msg + data
            self._partial_msg = None
        else:
            working_data = data
        while working_data:
            try:
                # Message encoding: <HEAD><SEP><len><SEP><msg><SEP><HEAD[::-1]>
                end = working_data.find(
                    SocketCommunicator.MSG_SEP + SocketCommunicator.MSG_HEAD[::-1]
                )
                msg_parts: List[bytes] = working_data[:end].split(
                    SocketCommunicator.MSG_SEP
                )
                if len(msg_parts) != 3:
                    self._partial_msg = working_data
                    break

                cmd: bytes
                nbytes: bytes
                raw_msg: bytes
                cmd, nbytes, raw_msg = msg_parts
                if len(raw_msg) != int(nbytes):
                    self._partial_msg = working_data
                    break
                msg = pickle.loads(raw_msg)
                self._msg_queue.put(msg)
            except pickle.UnpicklingError:
                self._partial_msg = working_data
                break
            if end < len(working_data):
                # Add len(SEP+HEAD) since end marks the start of <SEP><HEAD[::-1]
                offset: int = len(
                    SocketCommunicator.MSG_SEP + SocketCommunicator.MSG_HEAD
                )
                working_data = working_data[end + offset :]
            else:
                working_data = b""

    # Write
    ############################################################################

    def _write_socket(self, msg: MessageType) -> None:
        """Sends data over a socket from the 'client' (Task) side.

        Messages are encoded in the following format:
                 <HEAD><SEP><len(msg)><SEP><msg><SEP><HEAD[::-1]>
        The items between <> are replaced as follows:
            - <HEAD>: A start marker
            - <SEP>: A separator for components of the message
            - <len(msg)>: The length of the message payload in bytes.
            - <msg>: The message payload in bytes
            - <HEAD[::-1]>: The start marker in reverse to indicate the end.

        This structure is used for decoding the message on the other end.
        """
        data: bytes = pickle.dumps(msg)
        cmd: bytes = SocketCommunicator.MSG_HEAD
        size: bytes = b"%d" % len(data)
        end: bytes = SocketCommunicator.MSG_HEAD[::-1]
        sep: bytes = SocketCommunicator.MSG_SEP
        packed_msg: bytes = cmd + sep + size + sep + data + sep + end
        if USE_ZMQ:
            self._data_socket.send(packed_msg)
        else:
            assert isinstance(self._data_socket, socket.socket)
            self._data_socket.sendall(packed_msg)

    def write(self, msg: MessageType, proc: Optional[subprocess.Popen] = None) -> None:
        """Send a single Message.

        The entire Message (signal and contents) is serialized and sent through
        a connection over Unix socket.

        Args:
            msg (Message): The Message to send.
        """
        self._write_socket(msg)

    # Generic create
    ############################################################################

    def _create_socket_raw(self) -> socket.socket:
        """Create either a Unix or TCP socket.

        If the environment variable:
                              `LUTE_USE_TCP=1`
        is defined, a TCP socket is returned, otherwise a Unix socket.

        Refer to the individual initialization methods for additional environment
        variables controlling the behaviour of these two communication types.

        Returns:
            data_socket (socket.socket): TCP or Unix socket.
        """
        import struct

        use_tcp: Optional[str] = os.getenv("LUTE_USE_TCP")
        sock: socket.socket
        if use_tcp is not None:
            if self._party == Party.EXECUTOR:
                logger.info("Will use raw TCP sockets.")
            sock = self._init_tcp_socket_raw()
        else:
            if self._party == Party.EXECUTOR:
                logger.info("Will use raw Unix sockets.")
            sock = self._init_unix_socket_raw()
        sock.setsockopt(
            socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 10000)
        )
        return sock

    def _create_socket_zmq(self) -> zmq.sugar.socket.Socket:
        """Create either a Unix or TCP socket.

        If the environment variable:
                              `LUTE_USE_TCP=1`
        is defined, a TCP socket is returned, otherwise a Unix socket.

        Refer to the individual initialization methods for additional environment
        variables controlling the behaviour of these two communication types.

        Returns:
            data_socket (socket.socket): Unix socket object.
        """
        socket_type: int  # zmq.SocketType - either zmq.PULL or zmq.PUSH
        if self._party == Party.EXECUTOR:
            socket_type = zmq.PULL
        else:
            socket_type = zmq.PUSH

        data_socket: zmq.sugar.socket.Socket = self._context.socket(socket_type)
        data_socket.set_hwm(160000)
        # Need to multiply by 1000 since ZMQ uses ms
        data_socket.setsockopt(
            zmq.RCVTIMEO, int(SocketCommunicator.ACCEPT_TIMEOUT * 1000)
        )
        # Try TCP first
        use_tcp: Optional[str] = os.getenv("LUTE_USE_TCP")
        if use_tcp is not None:
            if self._party == Party.EXECUTOR:
                logger.info("Will use TCP (ZMQ).")
            self._init_tcp_socket_zmq(data_socket)
        else:
            if self._party == Party.EXECUTOR:
                logger.info("Will use Unix sockets (ZMQ).")
            self._init_unix_socket_zmq(data_socket)

        return data_socket

    # TCP Init
    ############################################################################

    def _find_random_port(
        self, min_port: int = 41923, max_port: int = 64324, max_tries: int = 100
    ) -> Optional[int]:
        """Find a random open port to bind to if using TCP."""
        from random import choices

        sock: socket.socket
        ports: List[int] = choices(range(min_port, max_port), k=max_tries)
        for port in ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(("", port))
                sock.close()
                del sock
                return port
            except Exception:
                continue
        return None

    def _init_tcp_socket_raw(self) -> socket.socket:
        """Initialize a TCP socket.

        Executor-side code should always be run first. It checks to see if
        the environment variable
                                `LUTE_PORT=###`
        is defined, if so binds it, otherwise find a free port from a selection
        of random ports. If a port search is performed, the `LUTE_PORT` variable
        will be defined so it can be picked up by the the Task-side Communicator.

        In the event that no port can be bound on the Executor-side, or the port
        and hostname information is unavailable to the Task-side, the program
        will exit.

        Returns:
            data_socket (socket.socket): TCP socket object.
        """
        data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port: Optional[Union[str, int]] = os.getenv("LUTE_PORT")
        if self._party == Party.EXECUTOR:
            if port is None:
                # If port is None find one
                # Executor code executes first
                port = self._find_random_port()
                if port is None:
                    # Failed to find a port to bind
                    logger.info(
                        "Executor failed to bind a port. "
                        "Try providing a LUTE_PORT directly! Exiting!"
                    )
                    sys.exit(-1)
                # Provide port env var for Task-side
                os.environ["LUTE_PORT"] = str(port)
            data_socket.bind(("", int(port)))
            data_socket.listen()
        else:
            hostname: str = socket.gethostname()
            executor_hostname: Optional[str] = os.getenv("LUTE_EXECUTOR_HOST")
            if executor_hostname is None or port is None:
                logger.info(
                    "Task-side does not have host/port information!"
                    " Check environment variables! Exiting!"
                )
                sys.exit(-1)
            if hostname == executor_hostname:
                data_socket.connect(("localhost", int(port)))
            else:
                data_socket.connect((executor_hostname, int(port)))
        return data_socket

    def _init_tcp_socket_zmq(self, data_socket: zmq.sugar.socket.Socket) -> None:
        """Initialize a TCP socket using ZMQ.

        Equivalent as the method above but requires passing in a ZMQ socket
        object instead of returning one.

        Args:
            data_socket (zmq.socket.Socket): Socket object.
        """
        port: Optional[Union[str, int]] = os.getenv("LUTE_PORT")
        if self._party == Party.EXECUTOR:
            if port is None:
                new_port: int = data_socket.bind_to_random_port("tcp://*")
                if new_port is None:
                    # Failed to find a port to bind
                    logger.info(
                        "Executor failed to bind a port. "
                        "Try providing a LUTE_PORT directly! Exiting!"
                    )
                    sys.exit(-1)
                port = new_port
                os.environ["LUTE_PORT"] = str(port)
            else:
                data_socket.bind(f"tcp://*:{port}")
            logger.debug(f"Executor bound port {port}")
        else:
            executor_hostname: Optional[str] = os.getenv("LUTE_EXECUTOR_HOST")
            if executor_hostname is None or port is None:
                logger.info(
                    "Task-side does not have host/port information!"
                    " Check environment variables! Exiting!"
                )
                sys.exit(-1)
            data_socket.connect(f"tcp://{executor_hostname}:{port}")

    # Unix Init
    ############################################################################

    def _get_socket_path(self) -> str:
        """Return the socket path, defining one if it is not available.

        Returns:
            socket_path (str): Path to the Unix socket.
        """
        socket_path: str
        try:
            socket_path = os.environ["LUTE_SOCKET"]
        except KeyError:
            import uuid
            import tempfile

            # Define a path, and add to environment
            # Executor-side always created first, Task will use the same one
            socket_path = f"{tempfile.gettempdir()}/lute_{uuid.uuid4().hex}.sock"
            os.environ["LUTE_SOCKET"] = socket_path
            logger.debug(f"SocketCommunicator defines socket_path: {socket_path}")
        if USE_ZMQ:
            return f"ipc://{socket_path}"
        else:
            return socket_path

    def _init_unix_socket_raw(self) -> socket.socket:
        """Returns a Unix socket object.

        Executor-side code should always be run first. It checks to see if
        the environment variable
                                `LUTE_SOCKET=XYZ`
        is defined, if so binds it, otherwise it will create a new path and
        define the environment variable for the Task-side to find.

        On the Task (client-side), this method will also open a SSH tunnel to
        forward a local Unix socket to an Executor Unix socket if the Task and
        Executor processes are on different machines.

        Returns:
            data_socket (socket.socket): Unix socket object.
        """
        socket_path: str = self._get_socket_path()
        data_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if self._party == Party.EXECUTOR:
            if os.path.exists(socket_path):
                os.unlink(socket_path)
            data_socket.bind(socket_path)
            data_socket.listen()
        elif self._party == Party.TASK:
            hostname: str = socket.gethostname()
            executor_hostname: Optional[str] = os.getenv("LUTE_EXECUTOR_HOST")
            if executor_hostname is None:
                logger.info("Hostname for Executor process not found! Exiting!")
                data_socket.close()
                sys.exit(-1)
            if hostname == executor_hostname:
                data_socket.connect(socket_path)
            else:
                self._local_socket_path = self._setup_unix_ssh_tunnel(
                    socket_path, hostname, executor_hostname
                )
                while 1:
                    # Keep trying reconnect until ssh tunnel works.
                    try:
                        data_socket.connect(self._local_socket_path)
                        break
                    except FileNotFoundError:
                        continue

        return data_socket

    def _init_unix_socket_zmq(self, data_socket: zmq.sugar.socket.Socket) -> None:
        """Initialize a Unix socket object, using ZMQ.

        Equivalent as the method above but requires passing in a ZMQ socket
        object instead of returning one.

        Args:
            data_socket (socket.socket): ZMQ object.
        """
        socket_path = self._get_socket_path()
        if self._party == Party.EXECUTOR:
            if os.path.exists(socket_path):
                os.unlink(socket_path)
            data_socket.bind(socket_path)
        elif self._party == Party.TASK:
            hostname: str = socket.gethostname()
            executor_hostname: Optional[str] = os.getenv("LUTE_EXECUTOR_HOST")
            if executor_hostname is None:
                logger.info("Hostname for Executor process not found! Exiting!")
                self._data_socket.close()
                sys.exit(-1)
            if hostname == executor_hostname:
                data_socket.connect(socket_path)
            else:
                # Need to remove ipc:// from socket_path for forwarding
                self._local_socket_path = self._setup_unix_ssh_tunnel(
                    socket_path[6:], hostname, executor_hostname
                )
                # Need to add it back
                path: str = f"ipc://{self._local_socket_path}"
                data_socket.connect(path)

    def _setup_unix_ssh_tunnel(
        self, socket_path: str, hostname: str, executor_hostname: str
    ) -> str:
        """Prepares an SSH tunnel for forwarding between Unix sockets on two hosts.

        An SSH tunnel is opened with `ssh -L <local>:<remote> sleep 2`.
        This method of communication is slightly slower and incurs additional
        overhead - it should only be used as a backup. If communication across
        multiple hosts is required consider using TCP.  The Task will use
        the local socket `<LUTE_SOCKET>.task{##}`. Multiple local sockets may be
        created. It is assumed that the user is identical on both the
        Task machine and Executor machine.

        Returns:
            local_socket_path (str): The local Unix socket to connect to.
        """
        if "uuid" not in globals():
            import uuid
        local_socket_path = f"{socket_path}.task{uuid.uuid4().hex[:4]}"
        self._use_ssh_tunnel = True
        ssh_cmd: List[str] = [
            "ssh",
            "-o",
            "LogLevel=quiet",
            "-L",
            f"{local_socket_path}:{socket_path}",
            executor_hostname,
            "sleep",
            "2",
        ]
        logger.debug(f"Opening tunnel from {hostname} to {executor_hostname}")
        self._ssh_proc = subprocess.Popen(ssh_cmd)
        time.sleep(0.4)  # Need to wait... -> Use single Task comm at beginning?
        return local_socket_path

    # Clean up and properties
    ############################################################################

    def _clean_up(self) -> None:
        """Clean up connections."""
        if self._party == Party.EXECUTOR:
            self._stop_thread = True
            self._reader_thread.join()
            logger.debug("Closed reading thread.")

        self._data_socket.close()
        if USE_ZMQ:
            self._context.term()
        else:
            ...

        if os.getenv("LUTE_USE_TCP"):
            return
        else:
            if self._party == Party.EXECUTOR:
                os.unlink(cast(str, os.getenv("LUTE_SOCKET")))  # Should be defined
                return
            elif self._use_ssh_tunnel:
                if self._ssh_proc is not None:
                    self._ssh_proc.terminate()

    @property
    def has_messages(self) -> bool:
        if self._party == Party.TASK:
            # Shouldn't be called on Task-side
            return False

        if self._msg_queue.qsize() > 0:
            return True
        return False

    def __exit__(self):
        self._clean_up()
