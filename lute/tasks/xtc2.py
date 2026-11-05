"""
Task for converting xtc1 files to xtc2 format using zmq-based communication
between psana1 and psana2 environments.

This module includes the classes needed to write XTC2 files and process them.

Classes:
    - WriteXtc2(Task): Write a new converted XTC2 file with data received from
        a ReadXtc1 Task running in parallel.

Based on Mona's converter:
    https://github.com/monarin/xtc1to2
"""

import logging
import pickle
import re
import socket
import zlib
from pathlib import PosixPath
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Type, TypedDict, Union

import numpy as np
import numpy.typing as npt
import zmq
from psana.dgramedit import DgramEdit, AlgDef, DetectorDef  # type: ignore
from psana.psexp import TransitionId  # type: ignore
from mpi4py import MPI

from lute.execution.logging import get_logger
from lute.io.models.xtc import WriteXtc2Parameters
from lute.tasks.dataclasses import TaskStatus
from lute.tasks.task import Task

logger: logging.Logger = get_logger(__name__)


class ZmqReceiver:

    def __init__(self, addr: str = "tcp://*") -> None:
        """
        A helper for receiving messages using pyzmq.
        Bind to socket (e.g. tcp://127.0.0.1:5557) with PULL
        """
        self._zmq_context: zmq.Context = zmq.Context()
        self.zmq_socket: zmq.Socket = self._zmq_context.socket(zmq.PULL)

        self._zmq_port: Optional[int] = self.zmq_socket.bind_to_random_port(addr)
        if self._zmq_port is None:
            logger.error("Could not find a port to bind!")

    @property
    def zmq_port(self) -> Optional[int]:
        return self._zmq_port

    def recv_zipped_pickle(self, flags: int = 0, protocol: int = -1) -> Any:
        """Receive zipped pickle"""
        z: bytes = self.zmq_socket.recv(flags)
        p: bytes = zlib.decompress(z)
        return pickle.loads(p)

    def recv_array(
        self, md: Any, flags: int = 0, copy: bool = True, track: bool = False
    ) -> None:
        """Receive a numpy array"""
        msg: Any = self.zmq_socket.recv(flags=flags, copy=copy, track=track)
        buf: Any = memoryview(msg)
        temp: Any = np.frombuffer(buf, dtype=md["dtype"])
        return temp.reshape(md["shape"])

    def close(self):
        """Close zmq socket"""
        self.zmq_socket.close()


class TimingDef(TypedDict):
    pulseId: np.uint64
    timeStamp: np.uint64
    fixedRates: npt.NDArray[np.uint8]
    acRates: npt.NDArray[np.uint8]
    timeSlot: np.uint8
    timeSlotPhase: np.uint16
    ebeamPresent: np.uint8
    ebeamDestn: np.uint8
    ebeamCharge: np.uint16
    ebeamEnergy: npt.NDArray[np.uint16]
    xWavelength: npt.NDArray[np.uint16]
    dmod5: np.uint16
    mpsLimits: npt.NDArray[np.uint8]
    mpsPowerClass: npt.NDArray[np.uint8]
    sequenceValues: npt.NDArray[np.uint16]
    inhibitCounts: npt.NDArray[np.uint32]


def write_timing(timestamp: int) -> TimingDef:
    timing_data: TimingDef = {
        "pulseId": np.uint64(0),
        "timeStamp": np.uint64(timestamp),
        "fixedRates": np.uint8([0] * 10),  # type: ignore
        "acRates": np.uint8([0] * 6),  # type: ignore
        "timeSlot": np.uint8(0),
        "timeSlotPhase": np.uint16(0),
        "ebeamPresent": np.uint8(1),
        "ebeamDestn": np.uint8(0),
        "ebeamCharge": np.uint16(0),
        "ebeamEnergy": np.uint16([0] * 4),  # type: ignore
        "xWavelength": np.uint16([0] * 2),  # type: ignore
        "dmod5": np.uint16(0),
        "mpsLimits": np.uint8([0] * 16),  # type: ignore
        "mpsPowerClass": np.uint8([0] * 16),  # type: ignore
        "sequenceValues": np.uint16([0] * 18),  # type: ignore
        "inhibitCounts": np.uint32([0] * 8),  # type: ignore
    }
    return timing_data


def save_dgramedit(dg_edit: DgramEdit, outbuf: bytearray, outfile: BinaryIO) -> None:
    """Save dgram edit to output buffer and write to file"""
    dg_edit.save(outbuf)
    outfile.write(outbuf[: dg_edit.size])


class WriteXtc2(Task):
    """
    A task that writes XTC2 files, converting them from an Xtc1Reader Task that sends
    the data over ZMQ.

    This uses a ZMQ to communicate between the two Tasks run in parallel. Any startup
    information needed to get this communication up and running is transmitted via
    maestro RPC and multi-Task communication APIs.

    Args:
        params (ConvertXtc1to2Parameters): Configuration for the conversion task.
    """

    def __init__(
        self, *, params: WriteXtc2Parameters, use_mpi: bool = True, row_ids=None
    ) -> None:
        self._task_parameters: WriteXtc2Parameters
        super().__init__(params=params, use_mpi=use_mpi, row_ids=row_ids)

        self._mpi_rank: int = MPI.COMM_WORLD.Get_rank()
        self._mpi_size: int = MPI.COMM_WORLD.Get_size()

    def _run(self) -> None:
        par: WriteXtc2Parameters = self._task_parameters
        exp: str = par.lute_config.experiment
        run: Union[int, str] = par.lute_config.run
        logger.debug(f"Starting [XTC2 Writer Rank {self._mpi_rank}] with psana 2")
        data_spec: Dict[str, Any] = {}
        detnames: List[str] = []
        for detname, specs in par.xtc1_access_pattern.items():
            det_specs: List[Any] = []
            for spec in specs:
                spec_d: Dict[str, Any]
                if isinstance(spec, dict):
                    # Case when first-party different env bootstrap
                    spec_d = spec
                elif hasattr(spec, "dict"):
                    # Case when first-party same env, no bootstrap
                    spec_d = spec.dict()
                else:
                    logger.error(
                        "Unable to interpret spec data! Will try to continue without it! Received: ",
                        spec,
                    )
                    continue
                det_specs.append(spec_d)
            data_spec[detname] = det_specs
            detnames.append(detname)

        all_detnames: List[str] = detnames + ["runinfo", "epicsinfo"]

        namesId: Dict[str, int] = {}
        for idx, detname in enumerate(all_detnames):
            namesId[detname] = idx

        namesId["chunkinfo"] = len(namesId)
        namesId["smdinfo"] = len(namesId)

        if self._mpi_rank == 0:
            zmq_recv: ZmqReceiver = ZmqReceiver()

            new_port: Optional[int] = zmq_recv.zmq_port
            if new_port is None:
                # Failed to find a port to bind
                logger.error(
                    f"[XTC2 Writer Rank {self._mpi_rank}] Marking Task as FAILED and exiting!"
                )
                self._result.task_status = TaskStatus.FAILED
                return

            # Send it via Maestro so the Xtc1Reader can use it
            self.publish_metadata(
                {
                    "xtc1_zmq_port": new_port,
                    "xtc1_zmq_host": socket.gethostname(),
                    "xtc1_exp": exp,
                    "xtc1_run": run,
                }
            )

        ##################################################################

        # Allocating memory for DgramEdit output buffer
        MEMSIZE: int = 640000000
        outbuf: bytearray = bytearray(MEMSIZE)

        # Open output file for writing, but check directory structure
        hutch: str = exp[:3]
        required_pattern: str = f"{hutch}/{exp}/xtc"
        if required_pattern not in par.output_file:
            raise RuntimeError(
                f"Output directory must contain {required_pattern}. Check `output_file` "
                "in the configuration YAML!\n"
                f"Received: {par.output_file}"
            )

        # NOTE: To parallelize, each rank will write its own chunk
        #       HOWEVER - While each "big data" .xtc2 can be written in separate -cXYZ
        #       chunk files, the .smd.xtc2 "small data" file should be written as one
        #       single file with `-c000` chunk number. (This is a hard requirement)
        #       Therefore we do:
        #         1. Rank 0 Opens its -c000.xtc2 and -c000.smd.xtc2 files and writes all
        #            transitions through ENABLE.
        #         2. Each rank then writes its -cXYZ.xtc2 in parallel filled with the
        #            L1Accepts and chronologically ordered timestamps. They record the
        #            size and local offset of each datagram.
        #         3. Ranks < (size - 1) write a series of Disable, EndStep, BeginStep and
        #            then Enable again. This "intermediate Enable" has a chunkinfo object
        #            that points to the next chunk file.
        #
        #            Rank (size - 1) writes the final transitions (Disable -> EndRun) to
        #            its -cXYZ.xtc2 file.
        #
        #         4. All ranks send their recorded datagram offsets and sizes to rank 0.
        #            Rank 0 then writes the smalldata datagrams for these offsets. The
        #            transitions are indicated by the negative of the TransitionId enum
        #            cast as an int. These must also be included in the .smd.xtc2. The
        #            rank 0 process will track them, and also add the chunkinfo as needed.

        output_file: str = par.output_file.replace("-c000.", f"-c{self._mpi_rank:03d}.")
        xtc_path: PosixPath = PosixPath(output_file)
        xtc_dir: PosixPath = xtc_path.parent.absolute()
        xtc_dir.mkdir(parents=True, exist_ok=True)

        xtc2file: BinaryIO = open(output_file, "wb")

        smalldata_dir: PosixPath = xtc_dir / "smalldata"
        smalldata_dir.mkdir(parents=True, exist_ok=True)
        xtc_name_smd: str = PosixPath(par.output_file).stem
        final_smd_name: str = str(smalldata_dir / xtc_name_smd) + ".smd.xtc2"

        if self._mpi_rank == 0:
            smdfile: BinaryIO = open(final_smd_name, "wb")

        # Create config, algorithm, and detector
        config: DgramEdit = DgramEdit(transition_id=TransitionId.Configure)

        generic_det_alg: AlgDef = AlgDef("xtc1dump", 0, 1, 0)

        det_defs: Dict[str, DetectorDef] = {}
        serial_num: str = "detnum1"
        counter: int = 2
        for detname in detnames:
            det_def: DetectorDef = DetectorDef(detname, "generic_container", serial_num)
            serial_num += str(counter)
            counter += 1
            det_defs[detname] = det_def

        runinfo_alg: AlgDef = AlgDef("runinfo", 0, 0, 1)
        runinfo_det: DetectorDef = DetectorDef("runinfo", "runinfo", "")

        # Need to include the chunkinfo detector
        chunkinfo_alg: AlgDef = AlgDef("chunkinfo", 0, 0, 1)
        chunkinfo_det: DetectorDef = DetectorDef("chunkinfo", "chunkinfo", "")

        # Setup the algorithm/detector definitions for recording offsets in .smd.xtc2
        smdinfo_alg: AlgDef = AlgDef("offsetAlg", 0, 0, 0)
        smdinfo_det: DetectorDef = DetectorDef("smdinfo", "offsetAlg", "")

        # Define data formats
        ##############################
        runinfodef: Dict[str, Tuple[Type, int]] = {
            "expt": (str, 1),
            "runnum": (np.uint32, 0),
        }

        # Need to include the chunkinfo fields
        chunkinfodef: Dict[str, Tuple[Type, int]] = {
            "filename": (str, 1),
            "chunkid": (np.uint32, 0),
        }

        # Field definitions for the smalldata offsets in .smd.xtc2 file
        smdinfodef: Dict[str, Tuple[Type, int]] = {
            "intOffset": (np.uint64, 0),
            "intDgramSize": (np.uint64, 0),
        }

        # Hold calibration information
        # These will be stored as "epics" detectors
        calib_serial_num: str = "detnum"
        calib_detectors: Dict[str, DetectorDef] = {}
        epics_alg: AlgDef = AlgDef("raw", 2, 0, 0)
        # epics_det: DetectorDef = DetectorDef("epics", "epics", "detnum1234")
        # epics_def: Dict[str, Tuple[Type, int]] = {}

        # Base epicsinfo (for psana compat)
        epicsinfo_det: DetectorDef = DetectorDef("epicsinfo", "epicsinfo", "detnum1234")
        epicsinfo_alg: AlgDef = AlgDef("epicsinfo", 1, 0, 0)
        epicsinfo_def: Dict[str, Tuple[Type, int]] = {"keys": (str, 1)}
        epicsinfo: Optional[config.Detector] = None

        timing_alg: AlgDef = AlgDef("raw", 2, 1, 0)
        timing_det: DetectorDef = DetectorDef("timing", "ts", "detnum1234")
        timing_def: Dict[str, Tuple[Type, int]] = {
            "pulseId": (np.uint64, 0),
            "timeStamp": (np.uint64, 0),
            "fixedRates": (np.uint8, 1),
            "acRates": (np.uint8, 1),
            "timeSlot": (np.uint8, 0),
            "timeSlotPhase": (np.uint16, 0),
            "ebeamPresent": (np.uint8, 0),
            "ebeamDestn": (np.uint8, 0),
            "ebeamCharge": (np.uint16, 0),
            "ebeamEnergy": (np.uint16, 1),
            "xWavelength": (np.uint16, 1),
            "dmod5": (np.uint16, 0),
            "mpsLimits": (np.uint8, 1),
            "mpsPowerClass": (np.uint8, 1),
            "sequenceValues": (np.uint16, 1),
            "inhibitCounts": (np.uint32, 1),
        }
        namesId["timing"] = len(namesId)
        timing: Optional[config.Detector] = None

        # This will be sent before anything else - contains rank and type
        # of all the information to be stored for the detector
        # detname: {field: (type, rank)}
        datadef: Optional[Dict[str, Dict[str, Tuple[Type, int]]]] = None

        # Start saving data
        logger.info(f"[XTC2 Writer Rank {self._mpi_rank}]: Starting receiving")
        detector: config.Detector
        detectors: Dict[str, config.Detector] = {}
        namesId["epics"] = len(namesId)
        finished_ranks: int = 0

        # Each rank will record a tuple of (timestamp, offset, size) for each L1Accept
        # datagram it writes. The offset is a local offset. At the end these get sent
        # to rank 0 so it can finish writing out the .smd.xtc2 file
        smd_events: List[Tuple[int, int, int]] = []

        # Each chunk also writes out a Disable, EndStep, BeginStep, Enable at the end
        # these each have timestamps, of course, so we need a modifier to the L1Accept
        # timestamps to leave space for them
        chunk_l1_ts_modifier: int = 4 * self._mpi_rank  # 1 for each transition
        while True:
            if self._mpi_rank == 0:
                obj = zmq_recv.recv_zipped_pickle()
                dest_rank: int
                if "rank" in obj:
                    # Must pop, so it isn't treated inappropriately in loops below
                    dest_rank = obj.pop("rank")
                else:
                    dest_rank = 0
                if dest_rank != 0:
                    MPI.COMM_WORLD.send(obj, dest=dest_rank, tag=99)
                    if "end" in obj:
                        finished_ranks += 1
                    if finished_ranks == self._mpi_size:
                        break
                    else:
                        continue
                else:
                    if "end" in obj:
                        finished_ranks += 1
            else:
                obj = MPI.COMM_WORLD.recv(source=0, tag=99)
            # Begin timestamp is needed (we calculate this from the first L1Accept)
            # to set the correct timestamp for all transitions prior to the first L1.
            if "DATA_TYPE_INFO" in obj:
                datadef = obj["DATA_TYPE_INFO"]
                # detname: {
                #    "calib": (np.float32, 3),
                #    "photon_energy": (np.float64, 0),
                # }
                # Create detetors
                assert datadef is not None
                for detname in datadef:
                    if detname in detnames:
                        detector = config.Detector(
                            det_defs[detname],
                            generic_det_alg,
                            datadef[detname],
                            nodeId=int(par.node_id),
                            namesId=namesId[detname],
                        )
                        detectors[detname] = detector
                    elif "_calib" in detname:
                        # Constants are too big... We will split them up and attach them to
                        # indiviudal SlowUpdate datagrams as "epics" detectors
                        calib_type_info: Dict[str, Tuple[Type, int]] = datadef[detname]
                        for const_name, const_type in calib_type_info.items():
                            prefixed_name: str = (
                                f"{detname.replace('_calib','')}_{const_name}"
                            )
                            calib_serial_num += str(len(calib_detectors))
                            calib_epics_det: DetectorDef = DetectorDef(
                                "epics", "epics", calib_serial_num
                            )
                            # epics_alg: AlgDef = AlgDef("raw", 2, 0, 0)
                            namesId[prefixed_name] = len(namesId)
                            detector = config.Detector(
                                calib_epics_det,
                                epics_alg,
                                {prefixed_name: const_type},
                                nodeId=int(par.node_id),
                                namesId=namesId[prefixed_name],
                            )
                            epicsinfo_def.update({prefixed_name: (str, 1)})
                            calib_detectors[prefixed_name] = detector
                runinfo = config.Detector(
                    runinfo_det,
                    runinfo_alg,
                    runinfodef,
                    nodeId=int(par.node_id),
                    namesId=namesId["runinfo"],
                )
                chunkinfo = config.Detector(
                    chunkinfo_det,
                    chunkinfo_alg,
                    chunkinfodef,
                    nodeId=int(par.node_id),
                    namesId=namesId["chunkinfo"],
                )
                smdinfo = config.Detector(
                    smdinfo_det,
                    smdinfo_alg,
                    smdinfodef,
                    nodeId=int(par.node_id),
                    namesId=namesId["smdinfo"],
                )
                epicsinfo = config.Detector(
                    epicsinfo_det,
                    epicsinfo_alg,
                    epicsinfo_def,
                    nodeId=int(par.node_id),
                    namesId=namesId["epicsinfo"],
                )
                timing = config.Detector(
                    timing_det,
                    timing_alg,
                    timing_def,
                    nodeId=int(par.node_id),
                    namesId=namesId["timing"],
                )
            elif "start" in obj:
                # NOTE: Only run the actual writes on rank 0!
                #       We'll get started writing out the .smd.xtc2 through Enable too
                config_timestamp = obj["config_timestamp"]
                config.updatetimestamp(config_timestamp)

                if epicsinfo is not None:
                    for calib_name in calib_detectors.keys():
                        setattr(epicsinfo.epicsinfo, calib_name, calib_name)
                    epicsinfo.epicsinfo.keys = "epicsname"
                    config.adddata(epicsinfo.epicsinfo)

                # Only rank 0 writes
                if self._mpi_rank == 0:
                    save_dgramedit(config, outbuf, xtc2file)
                    save_dgramedit(config, outbuf, smdfile)

                beginrun = DgramEdit(
                    transition_id=TransitionId.BeginRun,
                    config_dgramedit=config,
                    ts=config_timestamp + 1,
                )
                runinfo.runinfo.expt = exp
                runinfo.runinfo.runnum = int(run)
                beginrun.adddata(runinfo.runinfo)

                # Only rank 0 writes
                if self._mpi_rank == 0:
                    save_dgramedit(beginrun, outbuf, xtc2file)
                    save_dgramedit(beginrun, outbuf, smdfile)

                beginstep = DgramEdit(
                    transition_id=TransitionId.BeginStep,
                    config_dgramedit=config,
                    ts=config_timestamp + 2,
                )

                # Only rank 0 writes
                if self._mpi_rank == 0:
                    save_dgramedit(beginstep, outbuf, xtc2file)
                    save_dgramedit(beginstep, outbuf, smdfile)

                enable = DgramEdit(
                    transition_id=TransitionId.Enable,
                    config_dgramedit=config,
                    ts=config_timestamp + 3,
                )

                # Only rank 0 writes
                if self._mpi_rank == 0:
                    # NOTE: UNLIKE subsequent Enable transitions, the chunkid doesn't
                    #       get added to this first one
                    save_dgramedit(enable, outbuf, xtc2file)
                    save_dgramedit(enable, outbuf, smdfile)

                current_timestamp = config_timestamp + 3
                if "calib_const" in obj:
                    for detname, det_consts in obj["calib_const"].items():
                        if detname == "timestamp":
                            continue
                        for const_name, constants in det_consts.items():
                            prefixed_name = f"{detname}_{const_name}"
                            slow_update: DgramEdit = DgramEdit(
                                transition_id=TransitionId.SlowUpdate,
                                config_dgramedit=config,
                                ts=current_timestamp + 1,
                            )

                            detector = calib_detectors[prefixed_name]
                            setattr(detector.raw, prefixed_name, constants)
                            slow_update.adddata(detector.raw)

                            # Only rank 0 writes this data!!
                            if self._mpi_rank == 0:
                                save_dgramedit(slow_update, outbuf, xtc2file)
                                save_dgramedit(slow_update, outbuf, smdfile)
                            current_timestamp += 1

            elif "end" in obj:
                # NOTE: All chunks write Disable, EndStep. The final chunk will then
                #       have EndRun, the others will then have BeginStep, Enable.
                last_rank: int = self._mpi_size - 1
                current_timestamp += 1
                disable = DgramEdit(
                    transition_id=TransitionId.Disable,
                    config_dgramedit=config,
                    ts=current_timestamp,
                )

                save_dgramedit(disable, outbuf, xtc2file)
                smd_events.append((current_timestamp, -9, 0))

                current_timestamp += 1
                endstep = DgramEdit(
                    transition_id=TransitionId.EndStep,
                    config_dgramedit=config,
                    ts=current_timestamp,
                )

                save_dgramedit(endstep, outbuf, xtc2file)
                smd_events.append((current_timestamp, -7, 0))

                current_timestamp += 1
                if self._mpi_rank == last_rank:
                    endrun = DgramEdit(
                        transition_id=TransitionId.EndRun,
                        config_dgramedit=config,
                        ts=current_timestamp,
                    )

                    save_dgramedit(endrun, outbuf, xtc2file)
                    smd_events.append((current_timestamp, -5, 0))
                else:
                    beginstep = DgramEdit(
                        transition_id=TransitionId.BeginStep,
                        config_dgramedit=config,
                        ts=current_timestamp,
                    )
                    save_dgramedit(beginstep, outbuf, xtc2file)
                    smd_events.append((current_timestamp, -6, 0))

                    # NOTE: For this "intermediate Enable" there must be a
                    #       `chunkinfo` which has an id and filename pointing to
                    #       the next chunk to read from.
                    current_timestamp += 1
                    enable = DgramEdit(
                        transition_id=TransitionId.Enable,
                        config_dgramedit=config,
                        ts=current_timestamp,
                    )
                    smd_events.append((current_timestamp, -8, 0))

                    chunkinfo.chunkinfo.chunkid = np.uint32(self._mpi_rank + 1)
                    next_filename: str = xtc_name_smd.replace(
                        re.findall(r"-c\d\d\d", xtc_name_smd)[0],
                        f"-c{(self._mpi_rank + 1):03d}",
                    )
                    chunkinfo.chunkinfo.filename = next_filename + ".xtc2"

                    enable.adddata(chunkinfo.chunkinfo)
                    save_dgramedit(enable, outbuf, xtc2file)

                if self._mpi_rank != 0:
                    logger.debug(f"Rank {self._mpi_rank} exiting.")
                    break
                else:
                    # Rank 0 needs to keep looping unless all ranks done
                    if finished_ranks == self._mpi_size:
                        logger.debug(
                            "Rank 0 exiting as all other ranks have completed."
                        )
                        break
            else:
                # Create L1Accept
                real_timestamp: int = obj["timestamp"]
                adjusted_timestamp: int = real_timestamp + chunk_l1_ts_modifier
                d0 = DgramEdit(
                    transition_id=TransitionId.L1Accept,
                    config_dgramedit=config,
                    ts=adjusted_timestamp,
                )

                for detname in obj:
                    if detname == "timestamp":
                        continue
                    detector = detectors[detname]
                    for attr in obj[detname]:
                        setattr(detector.xtc1dump, attr, obj[detname][attr])
                    d0.adddata(detector.xtc1dump)
                if timing is not None:
                    timing_data = write_timing(adjusted_timestamp)
                    for attr in timing_data:
                        setattr(timing.raw, attr, timing_data[attr])  # type: ignore
                    d0.adddata(timing.raw)

                # Before saving the file, get the offset to setup the smd file
                event_offset: int = xtc2file.tell()
                save_dgramedit(d0, outbuf, xtc2file)
                event_size: int = d0.size  # Also need the size of the datagram

                # Cache the tuple to later send to rank 0
                smd_events.append((adjusted_timestamp, event_offset, event_size))
                current_timestamp = adjusted_timestamp

        logger.info(f"[XTC2 Writer Rank {self._mpi_rank}]: XTC2 Writing Complete")
        xtc2file.close()

        if self._mpi_rank == 0:
            zmq_recv.close()

        MPI.COMM_WORLD.Barrier()

        # Gather the tuple of (timestamp, local_offset, dgram_size) to rank 0
        all_events: Optional[List[List[Tuple[int, int, int]]]] = MPI.COMM_WORLD.gather(
            smd_events, root=0
        )

        if self._mpi_rank == 0:
            assert all_events is not None

            logger.info(
                "[XTC2 Writer Rank 0]: Writing L1Accept data to smalldata file."
            )

            # The initial transitions and SlowUpdates were already written as we went
            # Just write the event offsets and intermediate transitions
            for r in range(self._mpi_size):
                next_rank: int = r + 1
                next_chunk_filename: str = xtc_name_smd.replace(
                    re.findall(r"-c\d\d\d", xtc_name_smd)[0],
                    f"-c{next_rank:03d}",
                )
                next_chunk_filename += ".xtc2"
                for ts, local_offset, size in all_events[r]:
                    if local_offset == -9:
                        # Indicator of a Disable
                        disable = DgramEdit(
                            transition_id=TransitionId.Disable,
                            config_dgramedit=config,
                            ts=ts,
                        )
                        save_dgramedit(disable, outbuf, smdfile)
                    elif local_offset == -7:
                        # Indicator of EndStep
                        endstep = DgramEdit(
                            transition_id=TransitionId.EndStep,
                            config_dgramedit=config,
                            ts=ts,
                        )
                        save_dgramedit(endstep, outbuf, smdfile)
                    elif local_offset == -5:
                        # Indicator of the EndRun (Should only come rank (size - 1))
                        endrun = DgramEdit(
                            transition_id=TransitionId.EndRun,
                            config_dgramedit=config,
                            ts=ts,
                        )

                        save_dgramedit(endrun, outbuf, smdfile)
                    elif local_offset == -6:
                        # Indicator of the BeginStep
                        beginstep = DgramEdit(
                            transition_id=TransitionId.BeginStep,
                            config_dgramedit=config,
                            ts=ts,
                        )
                        save_dgramedit(beginstep, outbuf, smdfile)
                    elif local_offset == -8:
                        # Indicator of the Enable
                        # NOTE: Must also add the NEXT chunk's info here
                        enable = DgramEdit(
                            transition_id=TransitionId.Enable,
                            config_dgramedit=config,
                            ts=ts,
                        )
                        chunkinfo.chunkinfo.chunkid = next_rank
                        chunkinfo.chunkinfo.filename = next_chunk_filename
                        enable.adddata(chunkinfo.chunkinfo)
                        save_dgramedit(enable, outbuf, smdfile)
                    else:
                        # L1Accept -- Record offset and Dgram size
                        # NOTE: Offset is local to the chunk's file
                        smd_d0 = DgramEdit(
                            transition_id=TransitionId.L1Accept,
                            config_dgramedit=config,
                            ts=ts,
                        )
                        smdinfo.offsetAlg.intOffset = np.uint64(local_offset)
                        smdinfo.offsetAlg.intDgramSize = np.uint64(size)
                        smd_d0.adddata(smdinfo.offsetAlg)
                        save_dgramedit(smd_d0, outbuf, smdfile)

            logger.info(
                f"[XTC2 Writer Rank 0]: Consolidated smalldata file written to: {final_smd_name}."
            )

        MPI.COMM_WORLD.Barrier()
