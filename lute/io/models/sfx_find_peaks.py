import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import pydantic

from lute.io.models.base import ThirdPartyParameters, TaskParameters, TemplateConfig

pydantic_ver: int = int(pydantic.version.VERSION[0])
if pydantic_ver > 1:
    from pydantic.v1 import BaseModel, Field, PositiveInt, validator, root_validator  # type: ignore
else:
    from pydantic import BaseModel, Field, PositiveInt, validator, root_validator


class SZCompressorParameters(BaseModel):
    compressor: Literal["qoz", "sz3"] = Field(
        "qoz", description='Compression algorithm ("qoz" or "sz3")'
    )
    abs_error: float = Field(10.0, description="Absolute error bound")
    bin_size: int = Field(2, description="Bin size")
    roi_window_size: int = Field(
        9,
        description="Default window size",
    )


class FindPeaksSFXParameters(TaskParameters):
    """Parameters for crystallographic (Bragg) peak finding using PyAlgos or Peakfinder8.

    The Peakfinder8 algorithm comes in two flavors "Peakfinder8" and "Peakfinder8_v2",
    the latter does not use a "slab" geometry. This, however, is not currently supported
    by downstream CrystFEL so should not yet be used for full pipelines.


    This peak finding Task optionally has the ability to compress/decompress
    data with SZ for the purpose of compression validation.
    """

    class Config(TaskParameters.Config):
        set_result: bool = True
        """Whether the Executor should mark a specified parameter as a result."""

    algorithm: Literal["PyAlgos", "Peakfinder8", "Peakfinder8_v2"] = Field(
        default="Peakfinder8",
        description="The peakfinding algorithm to use. Either Peakfinder8 (v1 or v2) or PyAlgos.",
    )

    outdir: str = Field(
        description="Output directory for cxi files",
    )
    n_events: int = Field(
        0,
        description="Number of events to process (0 to process all events)",
    )
    det_name: str = Field(
        description="Psana name of the detector storing the image data",
    )
    event_receiver: Literal["evr0", "evr1"] = Field(
        description="Event Receiver to be used: evr0 or evr1",
    )
    tag: str = Field(
        "",
        description="Tag to add to the output file names",
    )
    pv_camera_length: Union[float, str] = Field(
        description="PV associated with camera length "
        "(if a number, camera length directly)",
    )
    event_logic: bool = Field(
        False,
        description="True if only events with a specific event code should be "
        "processed. False if the event code should be ignored",
    )
    event_code: int = Field(
        0,
        description="Required events code for events to be processed if event logic "
        "is True",
    )
    psana_version: Literal[1, 2] = Field(
        2,
        description="psana generation of the data: 2 for LCLS-II xtc2, 1 for LCLS-I xtc. "
        "psana 1 must run in a psana1 environment: use the PeakFinderSFXPsana1 managed "
        "Task, which sources conda1.",
    )
    psana_mask: bool = Field(
        False,
        description="If True, apply mask from psana Detector object",
    )
    mask_file: Union[str, None] = Field(
        None,
        description="File with a custom mask to apply. If None, no custom mask is "
        "applied",
    )
    min_peaks: int = Field(2, description="Minimum number of peaks per image")
    max_peaks: int = Field(
        2048,
        description="Maximum number of peaks per image",
    )
    npix_min: int = Field(
        2,
        description="Minimum number of pixels per peak",
    )
    npix_max: int = Field(
        30,
        description="Maximum number of pixels per peak",
    )
    amax_thr: float = Field(
        80.0,
        description="Minimum intensity threshold for starting a peak",
    )
    atot_thr: float = Field(
        120.0,
        description="Minimum summed intensity threshold for pixel collection",
    )
    son_min: float = Field(
        7.0,
        description="Minimum signal-to-noise ratio to be considered a peak",
    )
    peak_rank: int = Field(
        3,
        description="Radius in which central peak pixel is a local maximum",
    )
    r0: float = Field(
        3.0,
        description="Radius of ring for background evaluation in pixels",
    )
    dr: float = Field(
        2.0,
        description="Width of ring for background evaluation in pixels",
    )
    nsigm: float = Field(
        7.0,
        description="Intensity threshold to include pixel in connected group",
    )
    compression: Optional[SZCompressorParameters] = Field(
        None,
        description="Options for the SZ Compression Algorithm",
    )
    out_file: str = Field(
        "",
        description="Path to output file.",
        is_result=True,
    )
    geometry_file: Optional[str] = Field(
        None,
        description="A path to a CrystFEL geometry file (for pf8).",
    )
    make_powder_plots: bool = Field(
        True,
        description="Whether to generate assembled powder plots in the eLog.",
    )

    @validator("out_file", always=True)
    def validate_out_file(cls, out_file: str, values: Dict[str, Any]) -> str:
        if out_file == "":
            fname: Path = (
                Path(values["outdir"])
                / f"{values['lute_config'].experiment}_{values['lute_config'].run}_"
                f"{values['tag']}.list"
            )
            return str(fname)
        return out_file


class FindPeaksPsocakeParameters(ThirdPartyParameters):
    """Parameters for crystallographic (Bragg) peak finding using Psocake.

    This peak finding Task optionally has the ability to compress/decompress
    data with SZ for the purpose of compression validation.
    NOTE: This Task is deprecated and provided for compatibility only.
    """

    class Config(TaskParameters.Config):
        set_result: bool = True
        """Whether the Executor should mark a specified parameter as a result."""

        result_from_params: str = ""
        """Defines a result from the parameters. Use a validator to do so."""

    executable: str = Field("mpirun", description="MPI executable.", flag_type="")
    np: PositiveInt = Field(
        max(int(os.environ.get("SLURM_NPROCS", len(os.sched_getaffinity(0)))) - 1, 1),
        description="Number of processes",
        flag_type="-",
    )
    mca: str = Field(
        "btl ^openib", description="Mca option for the MPI executable", flag_type="--"
    )
    p_arg1: str = Field(
        "python", description="Executable to run with mpi (i.e. python).", flag_type=""
    )
    u: str = Field(
        "", description="Python option for unbuffered output.", flag_type="-"
    )
    p_arg2: str = Field(
        "findPeaksSZ.py",
        description="Executable to run with mpi (i.e. python).",
        flag_type="",
    )
    d: str = Field(description="Detector name", flag_type="-")
    e: str = Field("", description="Experiment name", flag_type="-")
    r: int = Field(-1, description="Run number", flag_type="-")
    outDir: str = Field(
        description="Output directory where .cxi will be saved", flag_type="--"
    )
    algorithm: int = Field(1, description="PyAlgos algorithm to use", flag_type="--")
    alg_npix_min: float = Field(
        1.0, description="PyAlgos algorithm's npix_min parameter", flag_type="--"
    )
    alg_npix_max: float = Field(
        45.0, description="PyAlgos algorithm's npix_max parameter", flag_type="--"
    )
    alg_amax_thr: float = Field(
        250.0, description="PyAlgos algorithm's amax_thr parameter", flag_type="--"
    )
    alg_atot_thr: float = Field(
        330.0, description="PyAlgos algorithm's atot_thr parameter", flag_type="--"
    )
    alg_son_min: float = Field(
        10.0, description="PyAlgos algorithm's son_min parameter", flag_type="--"
    )
    alg1_thr_low: float = Field(
        80.0, description="PyAlgos algorithm's thr_low parameter", flag_type="--"
    )
    alg1_thr_high: float = Field(
        270.0, description="PyAlgos algorithm's thr_high parameter", flag_type="--"
    )
    alg1_rank: int = Field(
        3, description="PyAlgos algorithm's rank parameter", flag_type="--"
    )
    alg1_radius: int = Field(
        3, description="PyAlgos algorithm's radius parameter", flag_type="--"
    )
    alg1_dr: int = Field(
        1, description="PyAlgos algorithm's dr parameter", flag_type="--"
    )
    psanaMask_on: str = Field(
        "True", description="Whether psana's mask should be used", flag_type="--"
    )
    psanaMask_calib: str = Field(
        "True", description="Psana mask's calib parameter", flag_type="--"
    )
    psanaMask_status: str = Field(
        "True", description="Psana mask's status parameter", flag_type="--"
    )
    psanaMask_edges: str = Field(
        "True", description="Psana mask's edges parameter", flag_type="--"
    )
    psanaMask_central: str = Field(
        "True", description="Psana mask's central parameter", flag_type="--"
    )
    psanaMask_unbond: str = Field(
        "True", description="Psana mask's unbond parameter", flag_type="--"
    )
    psanaMask_unbondnrs: str = Field(
        "True", description="Psana mask's unbondnbrs parameter", flag_type="--"
    )
    mask: str = Field(
        "", description="Path to an additional mask to apply", flag_type="--"
    )
    clen: str = Field(
        description="Epics variable storing the camera length", flag_type="--"
    )
    coffset: float = Field(0, description="Camera offset in m", flag_type="--")
    minPeaks: int = Field(
        15,
        description="Minimum number of peaks to mark frame for indexing",
        flag_type="--",
    )
    maxPeaks: int = Field(
        15,
        description="Maximum number of peaks to mark frame for indexing",
        flag_type="--",
    )
    minRes: int = Field(
        0,
        description="Minimum peak resolution to mark frame for indexing ",
        flag_type="--",
    )
    sample: str = Field("", description="Sample name", flag_type="--")
    instrument: Union[None, str] = Field(
        None, description="Instrument name", flag_type="--"
    )
    pixelSize: float = Field(0.0, description="Pixel size", flag_type="--")
    auto: str = Field(
        "False",
        description=(
            "Whether to automatically determine peak per event peak "
            "finding parameters"
        ),
        flag_type="--",
    )
    detectorDistance: float = Field(
        0.0, description="Detector distance from interaction point in m", flag_type="--"
    )
    access: Literal["ana", "ffb"] = Field(
        "ana", description="Data node type: {ana,ffb}", flag_type="--"
    )
    szfile: str = Field("qoz.json", description="Path to SZ's JSON configuration file")
    lute_template_cfg: TemplateConfig = Field(
        TemplateConfig(
            template_name="sz.json",
            output_path="",  # Will want to change where this goes...
        ),
        description="Template information for the sz.json file",
    )
    sz_parameters: SZCompressorParameters = Field(
        description="Configuration parameters for SZ Compression", flag_type=""
    )

    @validator("e", always=True)
    def validate_e(cls, e: str, values: Dict[str, Any]) -> str:
        if e == "":
            return values["lute_config"].experiment
        return e

    @validator("r", always=True)
    def validate_r(cls, r: int, values: Dict[str, Any]) -> int:
        if r == -1:
            return values["lute_config"].run
        return r

    @validator("lute_template_cfg", always=True)
    def set_output_path(
        cls, lute_template_cfg: TemplateConfig, values: Dict[str, Any]
    ) -> TemplateConfig:
        if lute_template_cfg.output_path == "":
            lute_template_cfg.output_path = values["szfile"]
        return lute_template_cfg

    @validator("sz_parameters", always=True)
    def set_sz_compression_parameters(
        cls, sz_parameters: SZCompressorParameters, values: Dict[str, Any]
    ) -> None:
        values["compressor"] = sz_parameters.compressor
        values["binSize"] = sz_parameters.bin_size
        values["roiWindowSize"] = sz_parameters.roi_window_size
        if sz_parameters.compressor == "qoz":
            values["pressio_opts"] = {
                "pressio:abs": sz_parameters.abs_error,
                "qoz": {"qoz:stride": 8},
            }
        else:
            values["pressio_opts"] = {"pressio:abs": sz_parameters.abs_error}
        return None

    @root_validator(pre=False)
    def define_result(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        exp: str = values["lute_config"].experiment
        run: int = int(values["lute_config"].run)
        directory: str = values["outDir"]
        fname: str = f"{exp}_{run:04d}.lst"

        cls.Config.result_from_params = f"{directory}/{fname}"
        return values
