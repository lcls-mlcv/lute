"""
Classes for peak finding tasks in SFX.

Classes:
    CxiWriter: utility class for writing peak finding results to CXI files.

    FindPeaksSFX: peak finding using psana's PyAlgos algorithm or peakfinder8.
        Optional data compression and decompression with libpressio for data
        reduction tests.
"""

__all__ = ["CxiWriter", "FindPeaksSFX"]
__author__ = "Valerio Mariani, Gabriel Dorlhiac"

import logging
import random
import sys
from collections import namedtuple
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    TextIO,
    Tuple,
    TypedDict,
    TYPE_CHECKING,
    Union,
    cast,
)

try:
    from typing import TypeAlias  # type: ignore
except ImportError:
    from typing_extensions import TypeAlias

import h5py  # type: ignore
import holoviews as hv  # type: ignore
import numpy as np
import numpy.typing as npt
import panel as pn
from mpi4py.MPI import COMM_WORLD, SUM

from lute.execution.logging import get_logger
from lute.execution.ipc import Message
from lute.tasks.algorithms import _peakfinders_ext  # type: ignore
from lute.tasks.task import Task
from lute.tasks.dataclasses import TaskStatus, ElogSummaryPlots

if TYPE_CHECKING:
    from lute.io.models.sfx_find_peaks import FindPeaksSFXParameters
else:
    from lute.io.parameters import TaskParameters

    FindPeaksSFXParameters = TaskParameters


hv.extension("bokeh")
pn.extension()


logger: logging.Logger = get_logger("FindPeaksSFX", is_task=True)


class RadialStatistics(TypedDict):
    rstats_pixel_index: npt.NDArray[np.int32]
    rstats_radius: npt.NDArray[np.int32]
    rstats_num_pix: int
    radial_map: npt.NDArray[np.float64]


class Peakfinder8PeakList(TypedDict):
    num_peaks: int
    fs: List[float]
    ss: List[float]
    intensity: List[float]
    num_pixels: List[float]
    max_pixel_intensity: List[float]
    snr: List[float]


class Peakfinder8_v2PeakList(TypedDict):
    num_peaks: int
    fs: List[float]
    ss: List[float]
    intensity: List[float]
    num_pixels: List[float]
    max_pixel_intensity: List[float]
    snr: List[float]
    panel_number: List[int]


EventMaskData = namedtuple(
    "EventMaskData",
    ["data", "mask", "seconds", "nanoseconds", "fiducials", "clen", "photon_energy"],
)

EventData = namedtuple(
    "EventData",
    ["data", "seconds", "nanoseconds", "fiducials", "clen", "photon_energy"],
)

DetectorGeomInfo = namedtuple("DetectorGeomInfo", ["i_x", "i_y", "ipx", "ipy"])

PyAlgos: TypeAlias = Any
Detector: TypeAlias = Any


class CxiWriter:

    def __init__(
        self,
        outdir: str,
        rank: int,
        exp: str,
        run: int,
        n_events: int,
        det_shape: Tuple[int, ...],
        raw_det_shape: Tuple[int, ...],
        min_peaks: int,
        max_peaks: int,
        i_x: Any,  # Not typed becomes it comes from psana
        i_y: Any,  # Not typed becomes it comes from psana
        ipx: Any,  # Not typed becomes it comes from psana
        ipy: Any,  # Not typed becomes it comes from psana
        tag: str,
        algo: Literal["Peakfinder8", "Peakfinder8_v2", "PyAlgos"] = "Peakfinder8",
    ):
        """
        Set up the CXI files to which peak finding results will be saved.

        Parameters:

            outdir (str): Output directory for cxi file.

            rank (int): MPI rank of the caller.

            exp (str): Experiment string.

            run (int): Experimental run.

            n_events (int): Number of events to process.

            det_shape (Tuple[int, int]): Shape of the numpy array storing the detector
                data. This must be aCheetah-stile 2D array.

            raw_det_shape (Tuple[int, ...]): Shape of the numpy array storing the
                detector in raw unassembled form. Length = 2, 3, or 4.

            min_peaks (int): Minimum number of peaks per image.

            max_peaks (int): Maximum number of peaks per image.

            i_x (Any): Array of pixel indexes along x

            i_y (Any): Array of pixel indexes along y

            ipx (Any): Pixel indexes with respect to detector origin (x component)

            ipy (Any): Pixel indexes with respect to detector origin (y component)

            tag (str): Tag to append to cxi file names.

            algo (str): The algorithm being used - either Peakfinder8 or PyAlgos.
        """
        self._det_shape: Tuple[int, ...] = det_shape
        self._raw_det_shape: Tuple[int, ...] = raw_det_shape
        self._i_x: Any = i_x
        self._i_y: Any = i_y
        self._ipx: Any = ipx
        self._ipy: Any = ipy
        self._index: int = 0

        # Create and open the HDF5 file
        fname: str = f"{exp}_r{run:0>4}_{rank}{tag}.cxi"
        Path(outdir).mkdir(exist_ok=True)
        self._outh5: Any = h5py.File(Path(outdir) / fname, "w")

        # Entry_1 entry for processing with CrystFEL
        entry_1: Any = self._outh5.create_group("entry_1")

        keys: List[str]
        if "Peakfinder8" in algo:
            keys = [
                "nPeaks",
                "peakXPosRaw",
                "peakYPosRaw",
                "peakNPixels",
                "peakTotalIntensity",
                "peakMaxIntensity",
                "peakSNR",
            ]
            if algo == "Peakfinder8_v2":
                keys.append("peakPanelNumRaw")
        else:
            keys = [
                "nPeaks",
                "peakXPosRaw",
                "peakYPosRaw",
                "rcent",
                "ccent",
                "rmin",
                "rmax",
                "cmin",
                "cmax",
                "peakTotalIntensity",
                "peakMaxIntensity",
                "peakRadius",
            ]
        ds_expId: Any = entry_1.create_dataset(
            "experimental_identifier", (n_events,), maxshape=(None,), dtype=int
        )
        ds_expId.attrs["axes"] = "experiment_identifier"
        data_1: Any = entry_1.create_dataset(
            "/entry_1/data_1/data",
            (n_events, det_shape[0], det_shape[1]),
            chunks=(1, det_shape[0], det_shape[1]),
            maxshape=(None, det_shape[0], det_shape[1]),
            dtype=np.float32,
        )
        data_1.attrs["axes"] = "experiment_identifier"

        key: str
        for key in ["powderHits", "powderMisses", "mask"]:
            entry_1.create_dataset(
                f"/entry_1/data_1/{key}",
                det_shape,
                chunks=det_shape,
                maxshape=det_shape,
                dtype=float,
            )

        # Peak-related entries
        ds_x: Any
        for key in keys:
            if key == "nPeaks":
                ds_x = self._outh5.create_dataset(
                    f"/entry_1/result_1/{key}",
                    (n_events,),
                    maxshape=(None,),
                    dtype=int,
                )
                ds_x.attrs["minPeaks"] = min_peaks
                ds_x.attrs["maxPeaks"] = max_peaks
            else:
                ds_x = self._outh5.create_dataset(
                    f"/entry_1/result_1/{key}",
                    (n_events, max_peaks),
                    maxshape=(None, max_peaks),
                    chunks=(1, max_peaks),
                    dtype=float,
                )
            ds_x.attrs["axes"] = "experiment_identifier:peaks"

        # Timestamp entries
        lcls_1: Any = self._outh5.create_group("LCLS")
        keys = [
            "eventNumber",
            "machineTime",
            "machineTimeNanoSeconds",
            "fiducial",
            "photon_energy_eV",
        ]
        for key in keys:
            if key == "photon_energy_eV":
                ds_x = lcls_1.create_dataset(
                    f"{key}", (n_events,), maxshape=(None,), dtype=float
                )
            else:
                ds_x = lcls_1.create_dataset(
                    f"{key}", (n_events,), maxshape=(None,), dtype=int
                )
            ds_x.attrs["axes"] = "experiment_identifier"

        ds_x = self._outh5.create_dataset(
            "/LCLS/detector_1/EncoderValue", (n_events,), maxshape=(None,), dtype=float
        )
        ds_x.attrs["axes"] = "experiment_identifier"

    def write_event(
        self,
        img: npt.NDArray[np.float32],
        peaks: Any,  # Not typed becomes it comes from psana
        timestamp_seconds: int,
        timestamp_nanoseconds: int,
        timestamp_fiducials: int,
        photon_energy: float,
        clen: float,
        algo: Literal["Peakfinder8", "Peakfinder8_v2", "PyAlgos"] = "Peakfinder8",
    ):
        """
        Write peak finding results for an event into the HDF5 file.

        Parameters:

            img (npt.NDArray[np.float32]): Detector data for the event

            peaks: (Any): Peak information for the event, as recovered from the PyAlgos
                algorithm

            timestamp_seconds (int): Second part of the event's timestamp information

            timestamp_nanoseconds (int): Nanosecond part of the event's timestamp
                information

            timestamp_fiducials (int): Fiducials part of the event's timestamp
                information

            photon_energy (float): Photon energy for the event

            clen (float): Camera length/detector distance.
        """
        if algo == "PyAlgos":
            self._write_event_pyalgos(
                img=img,
                peaks=peaks,
                timestamp_seconds=timestamp_seconds,
                timestamp_nanoseconds=timestamp_nanoseconds,
                timestamp_fiducials=timestamp_fiducials,
                photon_energy=photon_energy,
                clen=clen,
            )
        elif algo not in ("Peakfinder8", "Peakfinder8_v2"):
            raise ValueError(f"Unsupported peakfinding algorithm {algo}!")
        else:
            v2: bool = False
            if algo == "Peakfinder8":
                v2 = False
            elif algo == "Peakfinder8_v2":
                v2 = True

            self._write_event_peakfinder8(
                img=img,
                peaks=peaks,
                timestamp_seconds=timestamp_seconds,
                timestamp_nanoseconds=timestamp_nanoseconds,
                timestamp_fiducials=timestamp_fiducials,
                photon_energy=photon_energy,
                clen=clen,
                v2=v2,
            )

    def _write_event_peakfinder8(
        self,
        img: npt.NDArray[np.float32],
        peaks: Union[Peakfinder8PeakList, Peakfinder8_v2PeakList],
        timestamp_seconds: int,
        timestamp_nanoseconds: int,
        timestamp_fiducials: int,
        photon_energy: float,
        clen: float,
        v2: bool = False,
    ) -> None:
        if self._outh5["/entry_1/data_1/data"].shape[0] <= self._index:
            self._outh5["entry_1/data_1/data"].resize(self._index + 1, axis=0)
            ds_key: str
            for ds_key in self._outh5["/entry_1/result_1"].keys():
                self._outh5[f"/entry_1/result_1/{ds_key}"].resize(
                    self._index + 1, axis=0
                )
            for ds_key in (
                "machineTime",
                "machineTimeNanoSeconds",
                "fiducial",
                "photon_energy_eV",
                "detector_1/EncoderValue",
            ):
                self._outh5[f"/LCLS/{ds_key}"].resize(self._index + 1, axis=0)

        # Entry_1 entry for processing with CrystFEL
        self._outh5["/entry_1/data_1/data"][self._index, :, :] = img.reshape(
            -1, img.shape[-1]
        )
        num_peaks: int = peaks["num_peaks"]
        self._outh5["/entry_1/result_1/nPeaks"][self._index] = num_peaks
        self._outh5["/entry_1/result_1/peakXPosRaw"][self._index, :num_peaks] = (
            np.array(peaks["fs"], dtype=np.float32)
        )
        self._outh5["/entry_1/result_1/peakYPosRaw"][self._index, :num_peaks] = (
            np.array(peaks["ss"], dtype=np.float32)
        )
        if v2:
            peaks = cast(Peakfinder8_v2PeakList, peaks)
            self._outh5["/entry_1/result_1/peakPanelNumRaw"][
                self._index, :num_peaks
            ] = np.array(peaks["panel_number"], dtype=np.float32)
        self._outh5["/entry_1/result_1/peakNPixels"][self._index, :num_peaks] = (
            np.array(peaks["num_pixels"], dtype=np.float32)
        )
        self._outh5["/entry_1/result_1/peakTotalIntensity"][self._index, :num_peaks] = (
            np.array(peaks["intensity"], dtype=np.float32)
        )
        self._outh5["/entry_1/result_1/peakMaxIntensity"][self._index, :num_peaks] = (
            np.array(peaks["max_pixel_intensity"], dtype=np.float32)
        )
        self._outh5["/entry_1/result_1/peakSNR"][self._index, :num_peaks] = np.array(
            peaks["snr"], dtype=np.float32
        )

        # LCLS entry dataset
        self._outh5["/LCLS/machineTime"][self._index] = timestamp_seconds
        self._outh5["/LCLS/machineTimeNanoSeconds"][self._index] = timestamp_nanoseconds
        self._outh5["/LCLS/fiducial"][self._index] = timestamp_fiducials
        self._outh5["/LCLS/photon_energy_eV"][self._index] = photon_energy

        # Add clen distance
        self._outh5["/LCLS/detector_1/EncoderValue"][self._index] = clen
        self._index += 1

    def _write_event_pyalgos(
        self,
        img: npt.NDArray[np.float32],
        peaks: Any,  # Not typed becomes it comes from psana
        timestamp_seconds: int,
        timestamp_nanoseconds: int,
        timestamp_fiducials: int,
        photon_energy: float,
        clen: float,
    ) -> None:
        ch_rows: npt.NDArray[np.float64] = (
            peaks[:, 0] * self._raw_det_shape[-2] + peaks[:, 1]
        )
        ch_cols: npt.NDArray[np.float64] = peaks[:, 2]

        if self._outh5["/entry_1/data_1/data"].shape[0] <= self._index:
            self._outh5["entry_1/data_1/data"].resize(self._index + 1, axis=0)
            ds_key: str
            for ds_key in self._outh5["/entry_1/result_1"].keys():
                self._outh5[f"/entry_1/result_1/{ds_key}"].resize(
                    self._index + 1, axis=0
                )
            for ds_key in (
                "machineTime",
                "machineTimeNanoSeconds",
                "fiducial",
                "photon_energy_eV",
                "detector_1/EncoderValue",
            ):
                self._outh5[f"/LCLS/{ds_key}"].resize(self._index + 1, axis=0)

        # Entry_1 entry for processing with CrystFEL
        self._outh5["/entry_1/data_1/data"][self._index, :, :] = img.reshape(
            -1, img.shape[-1]
        )
        self._outh5["/entry_1/result_1/nPeaks"][self._index] = peaks.shape[0]
        self._outh5["/entry_1/result_1/peakXPosRaw"][self._index, : peaks.shape[0]] = (
            ch_cols.astype("int")
        )
        self._outh5["/entry_1/result_1/peakYPosRaw"][self._index, : peaks.shape[0]] = (
            ch_rows.astype("int")
        )
        self._outh5["/entry_1/result_1/rcent"][self._index, : peaks.shape[0]] = peaks[
            :, 6
        ]
        self._outh5["/entry_1/result_1/ccent"][self._index, : peaks.shape[0]] = peaks[
            :, 7
        ]
        self._outh5["/entry_1/result_1/rmin"][self._index, : peaks.shape[0]] = peaks[
            :, 10
        ]
        self._outh5["/entry_1/result_1/rmax"][self._index, : peaks.shape[0]] = peaks[
            :, 11
        ]
        self._outh5["/entry_1/result_1/cmin"][self._index, : peaks.shape[0]] = peaks[
            :, 12
        ]
        self._outh5["/entry_1/result_1/cmax"][self._index, : peaks.shape[0]] = peaks[
            :, 13
        ]
        self._outh5["/entry_1/result_1/peakTotalIntensity"][
            self._index, : peaks.shape[0]
        ] = peaks[:, 5]
        self._outh5["/entry_1/result_1/peakMaxIntensity"][
            self._index, : peaks.shape[0]
        ] = peaks[:, 4]

        # Calculate and write pixel radius
        peaks_cenx: npt.NDArray[np.float64] = (
            self._i_x[
                np.array(peaks[:, 0], dtype=np.int64),
                np.array(peaks[:, 1], dtype=np.int64),
                np.array(peaks[:, 2], dtype=np.int64),
            ]
            + 0.5
            - self._ipx
        )
        peaks_ceny: npt.NDArray[np.float64] = (
            self._i_y[
                np.array(peaks[:, 0], dtype=np.int64),
                np.array(peaks[:, 1], dtype=np.int64),
                np.array(peaks[:, 2], dtype=np.int64),
            ]
            + 0.5
            - self._ipy
        )
        peak_radius: npt.NDArray[np.float64] = np.sqrt(
            (peaks_cenx**2) + (peaks_ceny**2)
        )
        self._outh5["/entry_1/result_1/peakRadius"][
            self._index, : peaks.shape[0]
        ] = peak_radius

        # LCLS entry dataset
        self._outh5["/LCLS/machineTime"][self._index] = timestamp_seconds
        self._outh5["/LCLS/machineTimeNanoSeconds"][self._index] = timestamp_nanoseconds
        self._outh5["/LCLS/fiducial"][self._index] = timestamp_fiducials
        self._outh5["/LCLS/photon_energy_eV"][self._index] = photon_energy

        # Add clen distance
        self._outh5["/LCLS/detector_1/EncoderValue"][self._index] = clen
        self._index += 1

    def write_non_event_data(
        self,
        powder_hits: npt.NDArray[np.float64],
        powder_misses: npt.NDArray[np.float64],
        mask: npt.NDArray[np.uint8],
    ):
        """
        Write to the file data that is not related to a specific event (masks, powders)

        Parameters:

            powder_hits (npt.NDArray[np.float64]): Virtual powder pattern from hits

            powder_misses (npt.NDArray[np.float64]): Virtual powder pattern from hits

            mask: (npt.NDArray[np.uint16]): Pixel ask to write into the file

        """
        # Add powders and mask to files, reshaping them to match the crystfel
        # convention
        self._outh5["/entry_1/data_1/powderHits"][:] = powder_hits
        self._outh5["/entry_1/data_1/powderMisses"][:] = powder_misses
        self._outh5["/entry_1/data_1/mask"][:] = (1 - mask).reshape(
            -1, mask.shape[-1]
        )  # Crystfel expects inverted values

    def optimize_and_close_file(
        self,
        num_hits: int,
        max_peaks: int,
        algo: Literal["PyAlgos", "Peakfinder8", "Peakfinder8_v2"] = "Peakfinder8",
    ):
        """
        Resize data blocks and write additional information to the file

        Parameters:

            num_hits (int): Number of hits for which information has been saved to the
                file

            max_peaks (int): Maximum number of peaks (per event) for which information
                can be written into the file

            algo (Literal["PyAlgos", "Peakfinder8", "Peakfinder8_v2"]): The algorithm
                being used for peakfinding. This affects the exact keys that are
                saved.
        """

        # Resize the entry_1 entry
        data_shape: Tuple[int, ...] = self._outh5["/entry_1/data_1/data"].shape
        self._outh5["/entry_1/data_1/data"].resize(
            (num_hits, data_shape[1], data_shape[2])
        )
        self._outh5["/entry_1/result_1/nPeaks"].resize((num_hits,))

        keys: List[str]
        if "Peakfinder8" in algo:
            keys = [
                "peakXPosRaw",
                "peakYPosRaw",
                "peakNPixels",
                "peakTotalIntensity",
                "peakMaxIntensity",
                "peakSNR",
            ]
            if algo == "Peakfinder8_v2":
                keys.append("peakPanelNumRaw")
        else:
            keys = [
                "peakXPosRaw",
                "peakYPosRaw",
                "rcent",
                "ccent",
                "rmin",
                "rmax",
                "cmin",
                "cmax",
                "peakTotalIntensity",
                "peakMaxIntensity",
                "peakRadius",
            ]

        key: str
        for key in keys:
            self._outh5[f"/entry_1/result_1/{key}"].resize((num_hits, max_peaks))

        # Resize LCLS entry
        for key in [
            "eventNumber",
            "machineTime",
            "machineTimeNanoSeconds",
            "fiducial",
            "detector_1/EncoderValue",
            "photon_energy_eV",
        ]:
            self._outh5[f"/LCLS/{key}"].resize((num_hits,))
        self._outh5.close()


def write_master_file(
    mpi_size: int,
    outdir: str,
    exp: str,
    run: int,
    tag: str,
    n_hits_per_rank: List[int],
    n_hits_total: int,
) -> Path:
    """
    Generate a virtual dataset to map all individual files for this run.

    Parameters:

        mpi_size (int): Number of ranks in the MPI pool.

        outdir (str): Output directory for cxi file.

        exp (str): Experiment string.

        run (int): Experimental run.

        tag (str): Tag to append to cxi file names.

        n_hits_per_rank (List[int]): Array containing the number of hits found on each
            node processing data.

        n_hits_total (int): Total number of hits found across all nodes.

    Returns:

        The path to the the written master file
    """
    # Retrieve paths to the files containing data
    fnames: List[Path] = []
    # Need to keep track of these for indexing later during master file creation
    ranks_with_hits: List[int] = []
    for fi in range(mpi_size):
        if n_hits_per_rank[fi] > 0:
            fnames.append(Path(outdir) / f"{exp}_r{run:0>4}_{fi}{tag}.cxi")
            ranks_with_hits.append(fi)
    if len(fnames) == 0:
        sys.exit("No hits found")

    # Retrieve list of entries to populate in the virtual hdf5 file
    dname_list, key_list, shape_list, dtype_list = [], [], [], []
    datasets = ["/entry_1/result_1", "/LCLS/detector_1", "/LCLS", "/entry_1/data_1"]
    f = h5py.File(fnames[0], "r")
    for dname in datasets:
        dset = f[dname]
        for key in dset.keys():
            if f"{dname}/{key}" not in datasets:
                dname_list.append(dname)
                key_list.append(key)
                shape_list.append(dset[key].shape)
                dtype_list.append(dset[key].dtype)
    f.close()

    # Compute cumulative powder hits and misses for all files
    # Copy mask as well
    mask: Optional[npt.NDArray[np.uint8]] = None
    powder_hits: Optional[npt.NDArray[np.float64]] = None
    powder_misses: Optional[npt.NDArray[np.float64]] = None
    for fn in fnames:
        f = h5py.File(fn, "r")
        if mask is None:
            mask = f["entry_1/data_1/mask"][:].copy()
        if powder_hits is None:
            powder_hits = f["entry_1/data_1/powderHits"][:].copy()
            powder_misses = f["entry_1/data_1/powderMisses"][:].copy()
        else:
            assert powder_misses is not None
            powder_hits = np.maximum(
                powder_hits, f["entry_1/data_1/powderHits"][:].copy()
            )
            powder_misses = np.maximum(
                powder_misses, f["entry_1/data_1/powderMisses"][:].copy()
            )
        f.close()

    vfname: Path = Path(outdir) / f"{exp}_r{run:0>4}{tag}.cxi"
    with h5py.File(vfname, "w") as vdf:
        # Write the virtual hdf5 file
        for dnum in range(len(dname_list)):
            dname = f"{dname_list[dnum]}/{key_list[dnum]}"
            if key_list[dnum] not in ["mask", "powderHits", "powderMisses"]:
                layout = h5py.VirtualLayout(
                    shape=(n_hits_total,) + shape_list[dnum][1:], dtype=dtype_list[dnum]
                )
                cursor = 0
                for i, fn in enumerate(fnames):
                    # Use the correct rank hit count or you get a malformed VDS
                    rank_idx: int = ranks_with_hits[i]
                    vsrc = h5py.VirtualSource(
                        fn,
                        dname,
                        shape=(n_hits_per_rank[rank_idx],) + shape_list[dnum][1:],
                    )

                    if len(shape_list[dnum]) == 1:
                        layout[cursor : cursor + n_hits_per_rank[rank_idx]] = vsrc
                    else:
                        layout[cursor : cursor + n_hits_per_rank[rank_idx], :] = vsrc
                    cursor += n_hits_per_rank[rank_idx]
                vdf.create_virtual_dataset(dname, layout, fillvalue=-1)

        vdf["entry_1/data_1/powderHits"] = powder_hits
        vdf["entry_1/data_1/powderMisses"] = powder_misses
        vdf["entry_1/data_1/mask"] = mask

    return vfname


def generate_libpressio_configuration(
    compressor: Literal["sz3", "qoz"],
    roi_window_size: int,
    bin_size: int,
    abs_error: float,
    libpressio_mask,
) -> Dict[str, Any]:
    """
    Create the configuration JSON for the libpressio library

    Parameters:

        compressor (Literal["sz3", "qoz"]): Compression algorithm to use
            ("qoz" or "sz3").

        abs_error (float): Bound value for the absolute error.

        bin_size (int): Bining Size.

        roi_window_size (int): Default size of the ROI window.

        libpressio_mask (npt.NDArray): mask to be applied to the data.

    Returns:

        lp_json (Dict[str, Any]): Dictionary storing the JSON configuration structure
        for the libpressio library
    """

    if compressor == "qoz":
        pressio_opts: Dict[str, Any] = {
            "pressio:abs": abs_error,
            "qoz": {"qoz:stride": 8},
        }
    elif compressor == "sz3":
        pressio_opts = {"pressio:abs": abs_error}

    ndims: int = len(libpressio_mask.shape)

    # Add padding for non-row/col dims
    other_dims: List[int] = [1] * (ndims - 2)
    roi_size: List[int] = other_dims * (ndims - 2) + [roi_window_size, roi_window_size]
    bin_dims: List[int] = other_dims * (ndims - 2) + [bin_size, bin_size]

    # Libpressio bug requires 4D data
    if len(roi_size) < 4:
        other_dims = [1] * (4 - len(roi_size))
        roi_size = other_dims + roi_size
        bin_dims = other_dims + bin_dims

    lp_json: Dict[str, Any] = {
        "compressor_id": "pressio",
        "early_config": {
            "pressio": {
                "pressio:compressor": "roibin",
                "roibin": {
                    "roibin:metric": "composite",
                    "roibin:background": "mask_binning",
                    "roibin:roi": "fpzip",
                    "background": {
                        "binning:compressor": "pressio",
                        "mask_binning:compressor": "pressio",
                        "pressio": {"pressio:compressor": compressor},
                    },
                    "composite": {
                        "composite:plugins": [
                            "size",
                            "time",
                            "input_stats",
                            "error_stat",
                        ]
                    },
                },
            }
        },
        "compressor_config": {
            "pressio": {
                "roibin": {
                    "roibin:roi_size": roi_size,
                    "roibin:centers": None,
                    "roibin:roi_strategy": "coordinates",
                    "roibin:nthreads": 1,
                    "roi": {"fpzip:prec": 0},
                    "background": {
                        "mask_binning:mask": None,
                        "mask_binning:shape": bin_dims,
                        "mask_binning:nthreads": 4,
                        "pressio": pressio_opts,
                    },
                }
            }
        },
        "name": "pressio",
    }

    # Need 4D array for libpressio bug
    reshaped_mask: npt.NDArray[np.uint8]
    if libpressio_mask.ndim < 4:
        new_shape: Tuple[int, ...] = (
            (1,) * (4 - libpressio_mask.ndim)
        ) + libpressio_mask.shape
        reshaped_mask = libpressio_mask.reshape(new_shape)
    else:
        reshaped_mask = libpressio_mask

    lp_json["compressor_config"]["pressio"]["roibin"]["background"][
        "mask_binning:mask"
    ] = (1 - reshaped_mask)

    return lp_json


def _libpressio_config_pyalgos(lp_json: Dict[str, Any], peaks: Any) -> Dict[str, Any]:
    # assert not isinstance(peaks, dict)
    peaks_new: npt.NDArray[np.uint64] = np.zeros((len(peaks), 4), dtype=np.uint64)
    peaks_new[:, 1:] = np.ascontiguousarray(np.uint64(peaks[:, [0, 1, 2]])).astype(
        np.uint64
    )
    lp_json["compressor_config"]["pressio"]["roibin"]["roibin:centers"] = peaks_new

    return lp_json


def _libpressio_config_pf8(
    lp_json: Dict[str, Any], peaks: Peakfinder8PeakList
) -> Dict[str, Any]:
    peaks_rotated: npt.NDArray[np.uint64] = np.zeros(
        (peaks["num_peaks"], 4), dtype=np.uint64
    )
    peaks_rotated[:, 2] = np.array(peaks["ss"]).astype(np.uint64)
    peaks_rotated[:, 3] = np.array(peaks["fs"]).astype(np.uint64)
    lp_json["compressor_config"]["pressio"]["roibin"]["roibin:centers"] = peaks_rotated
    return lp_json


def add_peaks_to_libpressio_configuration(
    lp_json: Dict[str, Any],
    peaks: Union[Peakfinder8PeakList, Peakfinder8_v2PeakList, Any],
    algo: Literal["PyAlgos", "Peakfinder8", "Peakfinder8_v2"],
) -> Dict[str, Any]:
    """
    Add peak infromation to libpressio configuration

    Parameters:

        lp_json: Dictionary storing the configuration JSON structure for the libpressio
            library.

        peaks (Any): Peak information as returned by psana.

    Returns:

        lp_json: Updated configuration JSON structure for the libpressio library.
    """
    if "Peakfinder8" in algo:
        return _libpressio_config_pf8(lp_json=lp_json, peaks=peaks)
    else:
        return _libpressio_config_pyalgos(lp_json=lp_json, peaks=peaks)


class FindPeaksSFX(Task):
    """
    Task that performs peak finding using the PyAlgos peak finding algorithms and
    writes the peak information to CXI files.
    """

    def __init__(
        self, *, params: FindPeaksSFXParameters, use_mpi: bool = True, row_ids=None
    ) -> None:
        self._task_parameters: FindPeaksSFXParameters = params
        super().__init__(params=params, use_mpi=use_mpi, row_ids=row_ids)

        self._algo: Literal["PyAlgos", "Peakfinder8", "Peakfinder8_v2"] = (
            self._task_parameters.algorithm
        )

    def get_radius_map(self) -> npt.NDArray[np.float64]:
        from lute.tasks.util.geometry import (
            CrystfelDetectorGeometry,
            PixelMaps,
            crystfel_to_pixel_map,
            parse_crystfel_geometry_file,
        )

        if self._task_parameters.geometry_file is not None:
            geom_desc: CrystfelDetectorGeometry = parse_crystfel_geometry_file(
                file_path=self._task_parameters.geometry_file
            )
            pixel_maps: PixelMaps = crystfel_to_pixel_map(geometry_desc=geom_desc)
            return pixel_maps["radius_map"]
        else:
            raise RuntimeError("Need a geometry file to compute radius map!")

    def _compute_num_radial_bins(
        self, indices: Tuple[slice, ...], radius_map: npt.NDArray[np.float64]
    ) -> int:
        return int(np.ceil(radius_map[indices].max()) + 1)

    def compute_radial_statistics(
        self,
        radius_map: npt.NDArray[np.float64],
        shape: Tuple[int, ...],
        num_bins: int = 100,
    ) -> RadialStatistics:
        radius_map_int: npt.NDArray[np.int8] = np.rint(radius_map).astype(int).ravel()
        peak_index: List[int] = []
        radius: List[int] = []
        for idx in np.split(
            np.argsort(radius_map_int, kind="mergesort"),
            np.cumsum(np.bincount(radius_map_int)[:-1]),
        ):
            if len(idx) < num_bins:
                peak_index.extend(idx)
                radius.extend(radius_map_int[(idx,)])
            else:
                idx_sample: List[int] = random.sample(list(idx), num_bins)
                peak_index.extend(idx_sample)
                radius.extend(radius_map_int[(idx_sample,)])

        rstats_pixel_index: npt.NDArray[np.int32] = np.array(peak_index).astype(
            np.int32
        )
        rstats_radius: npt.NDArray[np.int32] = np.array(radius).astype(np.int32)

        logger.debug(
            f"Computed radial statistics for {num_bins} bins. "
            f"rstats_num_pix: {rstats_pixel_index.size}"
        )

        return {
            "rstats_pixel_index": rstats_pixel_index,
            "rstats_radius": rstats_radius,
            "rstats_num_pix": rstats_pixel_index.size,
            "radial_map": radius_map.reshape(shape),
        }

    def _retrieve_psana_detector_data(self, lcls2: bool = True) -> DetectorGeomInfo:
        exp: str = self._task_parameters.lute_config.experiment
        run_num: int = int(self._task_parameters.lute_config.run)

        if lcls2:
            import psana  # type: ignore

            ds2: psana.psexp.mpi_ds.ParallelDataSource = psana.DataSource(
                exp=exp,
                run=run_num,
            )
            run2: psana.psexp.mpi_ds.RunParallel = next(ds2.runs())
            det2 = run2.Detector(self._task_parameters.det_name)

            i_x, i_y = det2.raw._pixel_coord_indexes()

            ipx, ipy = det2.raw._pixel_xy_at_z()
            return DetectorGeomInfo(i_x=i_x, i_y=i_y, ipx=ipx, ipy=ipy)
        else:
            from psana import Detector, MPIDataSource  # type: ignore

            ds: Any = MPIDataSource(f"exp={exp}:run={run_num}:smd")  # noqa

            det: Any = Detector(self._task_parameters.det_name)
            det.do_reshape_2d_to_3d(flag=True)
            i_x = det.indexes_x(run_num).astype(np.int64)
            i_y = det.indexes_y(run_num).astype(np.int64)
            ipx, ipy = det.point_indexes(run_num, pxy_um=(0, 0))

            return DetectorGeomInfo(i_x=i_x, i_y=i_y, ipx=ipx, ipy=ipy)

    def _event_generator(
        self, lcls2: bool = True
    ) -> Generator[Union[EventMaskData, EventData], None, None]:
        exp: str = self._task_parameters.lute_config.experiment
        run_num: int = int(self._task_parameters.lute_config.run)

        first_event: bool = True
        if lcls2:
            import psana  # type: ignore

            kwargs: Dict[str, Any] = {
                "exp": exp,
                "run": run_num,
            }
            if self._task_parameters.n_events != 0:
                kwargs["max_events"] = self._task_parameters.n_events

            ds2: psana.psexp.mpi_ds.ParallelDataSource = psana.DataSource(**kwargs)
            run2: psana.psexp.mpi_ds.RunParallel = next(ds2.runs())
            timing2 = run2.Detector("timing")

            det2 = run2.Detector(self._task_parameters.det_name)

            # TODO: Need to handle other BLD?
            ebeamh = run2.Detector("ebeamh")

            if first_event:
                mask2: Optional[npt.NDArray[np.uint8]] = None
            for evt in run2.events():
                data: Optional[npt.NDArray[np.float32]] = det2.raw.calib(evt)

                raw_timestamp: Optional[int] = timing2.raw.timestamp(evt)

                # Fiducial only makes sense if actually on LCLS1 timing
                raw_fiducial2: Optional[int] = None
                seconds2: Optional[int] = None
                nanoseconds2: Optional[int] = None
                if raw_timestamp:
                    raw_fiducial2 = raw_timestamp & 0x1FFFF
                    nanoseconds2 = raw_timestamp & 0xFFFFFFFF
                    seconds2 = raw_timestamp >> 32

                photon_energy2: Optional[float] = None
                try:
                    photon_energy2 = ebeamh.raw.ebeamPhotonEnergy(evt)
                    if photon_energy2 and np.isinf(photon_energy2):
                        raise ValueError

                except (AttributeError, ValueError):
                    ebeam_raw: Optional[float] = run2.Detector("SIOC:SYS0:ML00:AO192")(
                        evt
                    )
                    if ebeam_raw:
                        photon_energy2 = (1.23984197386209e-06 / ebeam_raw) * 1e9

                if isinstance(self._task_parameters.pv_camera_length, float):
                    clen2: float = self._task_parameters.pv_camera_length
                else:
                    clen2 = run2.Detector(self._task_parameters.pv_camera_length)(evt)

                if data is not None:
                    det2_shape: Tuple[int, ...] = data.shape
                    if len(det2_shape) == 3:
                        det2_shape = (det2_shape[0] * det2_shape[1], det2_shape[2])
                    else:
                        det2_shape = data.shape
                    if first_event:
                        if hasattr(det2.raw, "_mask"):
                            mask2 = det2.raw._mask()
                        else:
                            mask2 = np.ones(det2_shape, dtype=np.uint8)

                        first_event = False
                        yield EventMaskData(
                            data,
                            mask2,
                            seconds2,
                            nanoseconds2,
                            raw_fiducial2,
                            clen2,
                            photon_energy2,
                        )
                    else:
                        yield EventData(
                            data,
                            seconds2,
                            nanoseconds2,
                            raw_fiducial2,
                            clen2,
                            photon_energy2,
                        )
        else:
            from psana import Detector, EventId, MPIDataSource  # type: ignore

            ds: Any = MPIDataSource(f"exp={exp}:run={run_num}:smd")
            det: Any = Detector(self._task_parameters.det_name)
            det.do_reshape_2d_to_3d(flag=True)
            evr: Any = Detector(self._task_parameters.event_receiver)
            ebeam: Any = Detector("EBeam")

            if self._task_parameters.n_events != 0:
                ds.break_after(self._task_parameters.n_events)

            for evt in ds.events():
                evt_id: Any = evt.get(EventId)
                seconds: int = evt_id.time()[0]
                nanoseconds: int = evt_id.time()[1]
                fiducials: int = evt_id.fiducials()

                if self._task_parameters.event_logic:
                    event_codes: Any = evr.eventCodes(evt)
                    if (
                        event_codes is None
                        or self._task_parameters.event_code not in event_codes
                    ):
                        continue

                if isinstance(self._task_parameters.pv_camera_length, float):
                    clen: float = self._task_parameters.pv_camera_length
                else:
                    clen = (
                        ds.env()
                        .epicsStore()
                        .value(self._task_parameters.pv_camera_length)
                    )

                photon_energy: float
                try:
                    photon_energy = ebeam.get(evt).ebeamPhotonEnergy()
                    if np.isinf(photon_energy):
                        raise ValueError
                except (AttributeError, ValueError):
                    photon_energy = (
                        1.23984197386209e-06
                        / ds.env().epicsStore().value("SIOC:SYS0:ML00:AO192")
                    ) * 1e9

                # Keep the detector's own (panels, ss, fs) shape, as the psana2 branch
                # does: _run reshapes image and mask to the CrystFEL slab itself.
                data = det.calib(evt)
                if first_event:
                    if data is None:
                        continue
                    mask: npt.NDArray[np.uint8] = np.ones(data.shape, dtype=np.uint8)
                    psana_mask: Optional[npt.NDArray[Any]] = det.mask(
                        run_num,
                        calib=False,
                        status=True,
                        edges=False,
                        central=False,
                        unbond=False,
                        unbondnbrs=False,
                    )
                    if psana_mask is not None:
                        mask = psana_mask.astype(np.uint8).reshape(data.shape)
                    first_event = False
                    yield EventMaskData(
                        data,
                        mask,
                        seconds,
                        nanoseconds,
                        fiducials,
                        clen,
                        photon_energy,
                    )
                else:
                    yield EventData(
                        data,
                        seconds,
                        nanoseconds,
                        fiducials,
                        clen,
                        photon_energy,
                    )

    def _run(self) -> None:
        rank: int = COMM_WORLD.Get_rank()
        size: int = COMM_WORLD.Get_size()

        lcls2: bool = self._task_parameters.psana_version == 2
        i_x, i_y, ipx, ipy = self._retrieve_psana_detector_data(lcls2=lcls2)

        alg: Optional[
            Union[
                PyAlgos, _peakfinders_ext.peakfinder_8, _peakfinders_ext.peakfinder_8_v2
            ]
        ] = None
        num_hits: int = 0
        num_events: int = 0
        num_empty_images: int = 0
        tag: str = self._task_parameters.tag

        if (tag != "") and (tag[0] != "_"):
            tag = "_" + tag

        first_event: bool = True
        mask: Optional[npt.NDArray[np.uint8]] = None
        # Need to make this None.... psana2 ranks don't all get into this loop...
        file_writer: Optional[CxiWriter] = None
        powder_hits: Optional[npt.NDArray[np.float64]] = None
        powder_misses: Optional[npt.NDArray[np.float64]] = None
        for event_data in self._event_generator(lcls2=lcls2):
            if first_event:
                assert isinstance(event_data, EventMaskData)
                (
                    img,
                    mask,
                    timestamp_seconds,
                    timestamp_nanoseconds,
                    timestamp_fiducials,
                    clen,
                    photon_energy,
                ) = event_data

                first_event = False
            else:
                assert isinstance(event_data, EventData)
                (
                    img,
                    timestamp_seconds,
                    timestamp_nanoseconds,
                    timestamp_fiducials,
                    clen,
                    photon_energy,
                ) = event_data
            if img is None:
                num_empty_images += 1
                continue

            det_shape: Tuple[int, ...] = img.shape
            if len(det_shape) == 3:
                det_shape = (det_shape[0] * det_shape[1], det_shape[2])
            else:
                det_shape = img.shape

            shape: Tuple[int, ...] = img.shape
            img_reshaped: npt.NDArray[np.float32]
            if len(shape) == 2 or self._algo == "Peakfinder8_v2":
                img_reshaped = img
            else:
                img_reshaped = img.reshape(shape[0] * shape[1], shape[2])

            radial_stats: Optional[RadialStatistics] = None
            base_peakfinder_options: Dict[str, Any]
            if alg is None:
                # Mask should only come on first event
                assert mask is not None

                mask_reshaped: npt.NDArray[np.uint8]
                if len(shape) == 2 or self._algo == "Peakfinder8_v2":
                    mask_reshaped = mask
                else:
                    mask_reshaped = mask.reshape(shape[0] * shape[1], shape[2])

                if "Peakfinder8" in self._algo:
                    radius_map: npt.NDArray[np.float64] = self.get_radius_map()
                    adc_thresh = (
                        self._task_parameters.amax_thr
                    )  # Minimum ADC threshold for peak detection
                    hitfinder_min_snr = self._task_parameters.son_min
                    hitfinder_min_pix_count = self._task_parameters.npix_min
                    hitfinder_max_pix_count = self._task_parameters.npix_max
                    local_bg_radius = self._task_parameters.r0

                    if self._algo == "Peakfinder8":
                        alg = _peakfinders_ext.peakfinder_8

                        num_radial_bins: int = self._compute_num_radial_bins(
                            indices=tuple(slice(None, dim) for dim in radius_map.shape),
                            radius_map=radius_map,
                        )
                        radial_stats = self.compute_radial_statistics(
                            radius_map=radius_map,
                            shape=det_shape,
                            num_bins=num_radial_bins,
                        )

                        asic_nx: int  # Size in the fs axis (1 panel/asic)
                        asic_ny: int  # Size in the ss axis (1 panel/asic)
                        nasics_x: int  # Number of asics/panels in the fs axis
                        nasics_y: int  # Number of asics/pnaels in the ss axis
                        if len(shape) > 2:
                            asic_nx = img.shape[2]  # fs size
                            asic_ny = img.shape[1]  # ss size
                            nasics_x = 1  # Number of panels in fs dim
                            nasics_y = img.shape[0]  # Number of panels in ss dim
                        else:
                            asic_nx = img.shape[1]
                            asic_ny = img.shape[0]
                            nasics_x = 1
                            nasics_y = 1
                        base_peakfinder_options = {
                            "max_num_peaks": self._task_parameters.max_peaks,
                            "data": img_reshaped,  # Must be updated each time afterwards
                            "mask": mask_reshaped.astype(np.int8),
                            "pix_r": radial_stats["radial_map"],
                            "rstats_num_pix": radial_stats["rstats_num_pix"],
                            "rstats_pidx": radial_stats["rstats_pixel_index"],
                            "rstats_radius": radial_stats["rstats_radius"],
                            "fast": 1,
                            "asic_nx": asic_nx,
                            "asic_ny": asic_ny,
                            "nasics_x": nasics_x,
                            "nasics_y": nasics_y,
                            "adc_thresh": adc_thresh,
                            "hitfinder_min_snr": hitfinder_min_snr,
                            "hitfinder_min_pix_count": hitfinder_min_pix_count,
                            "hitfinder_max_pix_count": hitfinder_max_pix_count,
                            "hitfinder_local_bg_radius": int(
                                local_bg_radius
                            ),  # Must be int
                        }
                    else:
                        alg = _peakfinders_ext.peakfinder_8_v2

                        base_peakfinder_options = {
                            "max_num_peaks": self._task_parameters.max_peaks,
                            "data": img,  # Must be updated each time afterwards
                            "mask": mask.astype(np.int8),
                            "pix_r": radius_map,
                            "adc_thresh": adc_thresh,
                            "hitfinder_min_snr": hitfinder_min_snr,
                            "hitfinder_min_pix_count": hitfinder_min_pix_count,
                            "hitfinder_max_pix_count": hitfinder_max_pix_count,
                            "hitfinder_local_bg_radius": int(
                                local_bg_radius
                            ),  # Must be int
                        }

                        # Reset the det_shape since we don't use the "slab" in v2
                        det_shape = img.shape
                else:
                    alg = PyAlgos(mask=mask, pbits=0)  # pbits controls verbosity
                    base_peakfinder_options = {
                        "rank": self._task_parameters.peak_rank,
                        "r0": self._task_parameters.r0,
                        "dr": self._task_parameters.dr,
                        "nsigm": self._task_parameters.nsigm,
                    }
                    assert alg is not None
                    alg.set_peak_selection_pars(
                        npix_min=self._task_parameters.npix_min,
                        npix_max=self._task_parameters.npix_max,
                        amax_thr=self._task_parameters.amax_thr,
                        atot_thr=self._task_parameters.atot_thr,
                        son_min=self._task_parameters.son_min,
                    )
                hdffh: Any
                if self._task_parameters.mask_file is not None:
                    with h5py.File(self._task_parameters.mask_file, "r") as hdffh:
                        loaded_mask: npt.NDArray[np.int64] = hdffh[
                            "entry_1/data_1/mask"
                        ][:]
                        mask *= loaded_mask.astype(np.uint8)

                file_writer = CxiWriter(
                    outdir=self._task_parameters.outdir,
                    rank=rank,
                    exp=self._task_parameters.lute_config.experiment,
                    run=int(self._task_parameters.lute_config.run),
                    n_events=self._task_parameters.n_events,
                    det_shape=det_shape,
                    raw_det_shape=img.shape,
                    i_x=i_x,
                    i_y=i_y,
                    ipx=ipx,
                    ipy=ipy,
                    min_peaks=self._task_parameters.min_peaks,
                    max_peaks=self._task_parameters.max_peaks,
                    tag=tag,
                    algo=self._algo,
                )

                libpressio_config: Optional[Dict[str, Any]] = None
                if self._task_parameters.compression is not None:
                    libpressio_config = generate_libpressio_configuration(
                        compressor=self._task_parameters.compression.compressor,
                        roi_window_size=self._task_parameters.compression.roi_window_size,
                        bin_size=self._task_parameters.compression.bin_size,
                        abs_error=self._task_parameters.compression.abs_error,
                        libpressio_mask=(
                            mask_reshaped if self._algo == "Peakfinder8" else mask
                        ),
                    )

                powder_hits = np.zeros(det_shape)
                powder_misses = np.zeros(det_shape)

            peaks: Union[Peakfinder8PeakList, Any]
            num_peaks: int = 0
            assert alg
            if self._algo == "Peakfinder8":
                shape = img.shape
                if len(img.shape) == 2:
                    img_reshaped = img
                else:
                    img_reshaped = img.reshape(shape[0] * shape[1], shape[2])

                # Update the options each time
                base_peakfinder_options["data"] = img_reshaped
                raw_pf8_lists: Tuple[
                    List[float],
                    List[float],
                    List[float],
                    List[int],
                    List[float],
                    List[float],
                    List[float],
                    List[float],
                ] = alg(**base_peakfinder_options)

                pf8_peaks: Peakfinder8PeakList = {
                    "num_peaks": len(raw_pf8_lists[0]),
                    "fs": raw_pf8_lists[0],
                    "ss": raw_pf8_lists[1],
                    "intensity": raw_pf8_lists[2],
                    "num_pixels": raw_pf8_lists[4],
                    "max_pixel_intensity": raw_pf8_lists[5],
                    "snr": raw_pf8_lists[6],
                }

                num_peaks = pf8_peaks["num_peaks"]
                peaks = pf8_peaks
            elif self._algo == "Peakfinder8_v2":
                # Update the options each time
                base_peakfinder_options["data"] = img
                raw_pf8_v2_lists: Tuple[
                    List[float],
                    List[float],
                    List[float],
                    List[int],
                    List[float],
                    List[float],
                    List[float],
                    List[float],
                    List[int],
                ] = alg(**base_peakfinder_options)
                pf8_v2_peaks: Peakfinder8_v2PeakList = {
                    "num_peaks": len(raw_pf8_v2_lists[0]),
                    "fs": raw_pf8_v2_lists[0],
                    "ss": raw_pf8_v2_lists[1],
                    "intensity": raw_pf8_v2_lists[2],
                    "num_pixels": raw_pf8_v2_lists[4],
                    "max_pixel_intensity": raw_pf8_v2_lists[5],
                    "snr": raw_pf8_v2_lists[6],
                    "panel_number": raw_pf8_v2_lists[8],
                }

                num_peaks = pf8_v2_peaks["num_peaks"]
                peaks = pf8_v2_peaks
            else:
                raw_pyalgos_peaks = alg(img, **base_peakfinder_options)
                num_peaks = raw_pyalgos_peaks.shape[0]

                peaks = raw_pyalgos_peaks

            num_events += 1
            if (num_peaks >= self._task_parameters.min_peaks) and (
                num_peaks <= self._task_parameters.max_peaks
            ):

                if self._task_parameters.compression is not None:
                    from libpressio import PressioCompressor  # type: ignore

                    assert libpressio_config
                    libpressio_config_with_peaks = (
                        add_peaks_to_libpressio_configuration(
                            lp_json=libpressio_config,
                            peaks=peaks,
                            algo=self._algo,
                        )
                    )
                    compressor = PressioCompressor.from_config(
                        libpressio_config_with_peaks
                    )
                    # Currently have a bug in libpressio so need 4D array
                    original_shape: Tuple[int, ...]
                    new_shape: Tuple[int, ...]
                    data_for_libpressio: npt.NDArray[np.float32]
                    if self._algo in ("Peakfinder8_v2", "PyAlgos"):
                        original_shape = img.shape
                        if img.ndim < 4:
                            new_shape = ((1,) * (4 - img.ndim)) + img.shape
                            data_for_libpressio = img.reshape(new_shape)
                        else:
                            data_for_libpressio = img
                    else:
                        original_shape = img_reshaped.shape
                        new_shape = (
                            1,
                            1,
                        ) + img_reshaped.shape
                        data_for_libpressio = img_reshaped.reshape(new_shape)

                    compressed_img: bytes = compressor.encode(data_for_libpressio)
                    decompressed_img: npt.NDArray[np.float32] = np.zeros_like(
                        data_for_libpressio
                    )
                    _ = compressor.decode(compressed_img, decompressed_img)
                    if self._algo in ("Peakfinder8_v2", "PyAlgos"):
                        img = decompressed_img.reshape(original_shape)
                    else:
                        img_reshaped = decompressed_img.reshape(original_shape)

                assert file_writer is not None
                file_writer.write_event(
                    img=img_reshaped if self._algo == "Peakfinder8" else img,
                    peaks=peaks,
                    timestamp_seconds=timestamp_seconds,
                    timestamp_nanoseconds=timestamp_nanoseconds,
                    timestamp_fiducials=timestamp_fiducials,
                    photon_energy=photon_energy,
                    clen=clen,
                    algo=self._algo,
                )
                num_hits += 1

            if num_peaks >= self._task_parameters.min_peaks:
                assert powder_hits is not None
                powder_hits = np.maximum(
                    powder_hits,
                    img.reshape(det_shape),
                )
            else:
                assert powder_misses is not None
                powder_misses = np.maximum(
                    powder_misses,
                    img.reshape(det_shape),
                )

        if num_empty_images != 0:
            msg: Message = Message(
                contents=f"Rank {rank} encountered {num_empty_images} empty images."
            )
            self._report_to_executor(msg)

        if file_writer is not None:
            assert mask is not None
            assert powder_hits is not None
            assert powder_misses is not None
            file_writer.write_non_event_data(
                powder_hits=powder_hits,
                powder_misses=powder_misses,
                mask=mask,
            )

            file_writer.optimize_and_close_file(
                num_hits=num_hits,
                max_peaks=self._task_parameters.max_peaks,
                algo=self._algo,
            )

        COMM_WORLD.Barrier()

        num_hits_per_rank: List[int] = cast(
            List[int], COMM_WORLD.gather(num_hits, root=0)
        )
        num_hits_total: int = cast(int, COMM_WORLD.reduce(num_hits, SUM))
        num_events_total: int = cast(int, COMM_WORLD.reduce(num_events, SUM))

        if rank == 0:
            master_fname: Path = write_master_file(
                mpi_size=size,
                outdir=self._task_parameters.outdir,
                exp=self._task_parameters.lute_config.experiment,
                run=int(self._task_parameters.lute_config.run),
                tag=tag,
                n_hits_per_rank=num_hits_per_rank,
                n_hits_total=num_hits_total,
            )

            # Write final summary file
            f: Union[TextIO, h5py.File]
            with open(
                Path(self._task_parameters.outdir) / f"peakfinding{tag}.summary", "w"
            ) as f:
                print(f"Number of events processed: {num_events_total}", file=f)
                print(f"Number of hits found: {num_hits_total}", file=f)
                print(
                    "Fractional hit rate: " f"{(num_hits_total/num_events_total):.2f}",
                    file=f,
                )
                print(f"No. hits per rank: {num_hits_per_rank}", file=f)

            with h5py.File(master_fname, "r") as f:
                final_powder_hits: npt.NDArray[np.float64] = f[
                    "entry_1/data_1/powderHits"
                ][:]
                final_powder_misses: npt.NDArray[np.float64] = f[
                    "entry_1/data_1/powderMisses"
                ][:]
                f.close()

            text_summary: Dict[str, str] = {
                "Number of events processed": str(num_events_total),
                "Number of hits found": str(num_hits_total),
                "Fractional hit rate": f"{num_hits_total/num_events_total:.2f}",
            }

            task_summary: Union[Dict[str, str], Tuple[Dict[str, str], ElogSummaryPlots]]
            if self._task_parameters.make_powder_plots:
                powder_plots: pn.Tabs = self._create_powder_plots(
                    powder_hits=final_powder_hits.astype(np.float64),
                    powder_misses=final_powder_misses.astype(np.float64),
                    i_x=i_x,
                    i_y=i_y,
                )
                task_summary = (
                    text_summary,
                    ElogSummaryPlots(
                        f"r{self._task_parameters.lute_config.run}/powders",
                        powder_plots,
                    ),
                )
            else:
                task_summary = text_summary

            self._result.summary = task_summary
            with open(Path(self._task_parameters.out_file), "w") as f:
                print(f"{master_fname}", file=f)

    def _post_run(self) -> None:
        super()._post_run()
        self._result.task_status = TaskStatus.COMPLETED

    def _assemble_image(
        self,
        img: npt.NDArray[np.float64],
        i_x: npt.NDArray[np.uint64],
        i_y: npt.NDArray[np.uint64],
    ) -> npt.NDArray[np.float64]:
        """Assemble an image based on psana geometry.

        Args:
            img (np.ndarray[np.float64]): The image to assemble. Should
                generally be of shape (n_panels, ss, fs)

            i_x (Any): Array of pixel indexes along x

            i_y (Any): Array of pixel indexes along y

        Returns:
            assembled_img(np.ndarray[np.float64]): Assembled 2D image.
        """
        img_reshaped: npt.NDArray[np.float64] = img.reshape(i_x.shape)
        idx_max_y: int = int(np.max(i_x) + 1)
        idx_max_x: int = int(np.max(i_y) + 1)
        assembled_img: npt.NDArray[np.float64] = np.zeros(
            (idx_max_y, idx_max_x), dtype=np.float64
        )
        assembled_img[i_x, i_y] = img_reshaped

        return assembled_img

    def _create_powder_plots(
        self,
        powder_hits: npt.NDArray[np.float64],
        powder_misses: npt.NDArray[np.float64],
        i_x: npt.NDArray[np.uint64],
        i_y: npt.NDArray[np.uint64],
    ) -> pn.Tabs:
        """Create a tabbed display of hits and misses 'powder' plots.

        Args:
            powder_hits (np.ndarray[np.float64]): Total max/sum projection of
                hits across the run.

            powder_misses (np.ndarray[np.float64]): Total max/sum projection of
                misses across the run.

            i_x (Any): Array of pixel indexes along x

            i_y (Any): Array of pixel indexes along y

        Returns:
            tabs (pn.Tabs): Tabbed display of the image plots.
        """
        assembled_powder_hits: npt.NDArray[np.float64] = self._assemble_image(
            img=powder_hits, i_x=i_x, i_y=i_y
        )
        assembled_powder_misses: npt.NDArray[np.float64] = self._assemble_image(
            img=powder_misses, i_x=i_x, i_y=i_y
        )

        grid_hits: pn.GridSpec = pn.GridSpec(
            sizing_mode="stretch_both",
            max_width=700,
            name=f"{self._task_parameters.det_name} - Hits",
        )
        dim: hv.Dimension = hv.Dimension(
            ("image", "Hits"),
            range=(
                np.nanpercentile(assembled_powder_hits, 1),
                np.nanpercentile(assembled_powder_hits, 99),
            ),
        )
        grid_hits[0, 0] = pn.Row(
            hv.Image(assembled_powder_hits, vdims=[dim], name=dim.label).options(
                colorbar=True, cmap="rainbow"
            )
        )

        grid_misses: pn.GridSpec = pn.GridSpec(
            sizing_mode="stretch_both",
            max_width=700,
            name=f"{self._task_parameters.det_name} - Misses",
        )
        dim = hv.Dimension(
            ("image", "Misses"),
            range=(
                np.nanpercentile(assembled_powder_misses, 1),
                np.nanpercentile(assembled_powder_misses, 99),
            ),
        )
        grid_misses[0, 0] = pn.Row(
            hv.Image(assembled_powder_misses, vdims=[dim], name=dim.label).options(
                colorbar=True, cmap="rainbow"
            )
        )

        tabs: pn.Tabs = pn.Tabs(grid_hits)
        tabs.append(grid_misses)
        return tabs
