"""Models for smalldata_tools Tasks.

Classes:
    SubmitSMDParameters(ThirdPartyParameters): Parameters to run smalldata_tools
        to produce a smalldata HDF5 file.

    AnalyzeSmallDataXSSParameters(TaskParameters): Parameter model for the
        AnalyzeSmallDataXSS Task. Used to determine spatial/temporal overlap
        based on XSS difference signal and provide basic XSS feedback.

    AnalyzeSmallDataXASParameters(TaskParameters): Parameter model for the
        AnalyzeSmallDataXAS Task. Used to determine spatial/temporal overlap
        based on XAS difference signal and provide basic XAS feedback.

    AnalyzeSmallDataXESParameters(TaskParameters): Parameter model for the
        AnalyzeSmallDataXES Task. Used to determine spatial/temporal overlap
        based on XES difference signal and provide basic XES feedback.
"""

__all__ = [
    "SubmitSMDParameters",
    "AnalyzeSmallDataXSSParameters",
    "AnalyzeSmallDataXASParameters",
    "AnalyzeSmallDataXESParameters",
    "RunBeamlineSummaryParameters",
]
__author__ = "Gabriel Dorlhiac"

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import (
    BaseModel,
    HttpUrl,
    PositiveInt,
    NonNegativeInt,
    Field,
    root_validator,
    validator,
)

from lute.io.models.base import TaskParameters, ThirdPartyParameters, TemplateConfig
from lute.io.models.validators import validate_smd_path, template_parameter_validator


class SubmitSMDParameters(ThirdPartyParameters):
    """Parameters for running smalldata to produce reduced HDF5 files."""

    class Config(ThirdPartyParameters.Config):
        """Identical to super-class Config but includes a result."""

        set_result: bool = True
        """Whether the Executor should mark a specified parameter as a result."""

        result_from_params: str = ""
        """Defines a result from the parameters. Use a validator to do so."""

    class ProducerParameters(BaseModel):
        class IntgParams(BaseModel):
            intg_main: str = Field(
                description=(
                    "The integrating detector to be passed to psana. "
                    "Should be the SLOWEST integrating detector in the data."
                )
            )
            intg_addl: List[str] = Field(
                description=(
                    "Additional detectors to be analyzed as integrating detectors. "
                    "The readout frequency must be commensurate and in-phase with intg_main."
                )
            )

        class ROIParams(BaseModel):
            ROI: Optional[List[List[List[int]]]] = Field(
                description="Definition of ROIs, can define multiple."
            )

            writeArea: bool = Field(
                False, description="Whether to write out the area image of the ROI."
            )

            thresADU: Optional[float] = Field(
                None, description="Optional threshold on ADU."
            )

            calcPars: Optional[bool] = Field(False, description="...")

        class AzIntParams(BaseModel):
            eBeam: float = Field(description="Beam energy in keV.")

            center: List[float] = Field(description="Beam center in micrometers")

            dis_to_sam: float = Field(description="Detector distance in millimeters.")

            tx: float = Field(0, description="Tilt in x, degrees")

            ty: float = Field(0, description="Tilt in y, degress")

            ADU_per_Photon: float = Field(1.0, description="ADU per photon.")

            phiBins: int = Field(1, description="Number of phi bins.")

            qbin: float = Field(5e-3, description="Width of Q bin.")

            thresRms: Optional[Union[int, float]] = Field(
                None, description="Lower threshold in RMS."
            )

            thresADU: Optional[Union[int, float]] = Field(
                None, description="Lower threshold in ADU."
            )

            thresADUhigh: Optional[Union[int, float]] = Field(
                None, description="High threshold in ADU."
            )

            geomCorr: bool = Field(
                True, description="Whether to apply geometric correction."
            )

            polCorr: bool = Field(
                True, description="Whether to apply polarization correction."
            )

            userMask: Optional[str] = Field(
                None,
                description=(
                    "Path to a numpy array (.npy) to use as a mask for azimuthal "
                    "integration."
                ),
            )

        class AzIntPyFAIParams(BaseModel):
            class AiKwargs(BaseModel):
                dist: float = Field(description="Detector distance.")
                poni1: float = Field(description="First poni.")
                poni2: float = Field(description="Second poni.")

            poni_file: Optional[str] = Field(
                None,
                description="Path to a poni file. Must provide poni_file or ai_kwargs.",
            )

            ai_kwargs: Optional[AiKwargs] = Field(
                None, description="Integration paramters if not using a poni file."
            )

            npts: int = Field(512, description="Number of q points/bins.")

            npts_az: int = Field(13, description="Number of phi bins.")

            int_units: str = Field("2th_deg", description="Integration units")

            return2d: bool = Field(
                False, description="Whether to return the 2D q/phi integration."
            )

        class PhotonParams(BaseModel):
            ADU_per_photon: float = Field(9.5, description="Number of ADU per photon.")

            thresADU: float = Field(
                0.8, description="Threshold in fraction of ADU_per_photon."
            )

        class DropletParams(BaseModel):
            name: str = Field(
                "droplet", description="HDF5 key name for storing droplet data."
            )

            # mask: Optional[np.ndarray] = Field(None, description="Optionally pass a separate mask.")

            threshold: float = Field(
                5,
                description="Threshold for pixel to be part of a droplet. Sigma or ADU depending on useRms.",
            )

            thresholdLow: float = Field(
                5, description="Lower threshold to make spectrum sharper."
            )

            thresADU: float = Field(
                60, description="Threshold on droplet ADU. Rejects droplets below this."
            )

            useRms: bool = Field(
                True,
                description="If True, threshold/thresholdLow are RMS of data, otherwise in ADU.",
            )

            nData: Optional[int] = Field(1e5, description="(float,int or None).")

            relabel: bool = Field(
                True, description="After initial finding, relabel image."
            )

        class Droplet2PhotonParams(BaseModel):
            class DropletParams(BaseModel):
                threshold: float = Field(
                    5,
                    description="Threshold for pixel to be part of a droplet. Sigma or ADU depending on useRms.",
                )

                thresholdLow: float = Field(
                    5, description="Lower threshold to make spectrum sharper."
                )

                thresADU: float = Field(
                    60,
                    description="Threshold on droplet ADU. Rejects droplets below this.",
                )

                useRms: bool = Field(
                    True,
                    description="If True, threshold/thresholdLow are RMS of data, otherwise in ADU.",
                )

            droplet: DropletParams = Field(
                DropletParams(threshold=5, thresholdLow=5, thresADU=60, useRms=True),
                description="Droplet finding parameters.",
            )

            aduspphot: int = Field(162, description="")

            cputime: bool = Field(True, description="")

            nData: float = Field(3e4, description="")

        class SvdParams(BaseModel):
            name: str = Field("svdFit", description="DetObject name.")

            n_components: int = Field(
                2, description="Number of components to use. Max is 25."
            )

            basis_file: Optional[str] = Field(None, description="")

            n_pulse: int = Field(1, description="Number of pulses to fit.")

            delay: Optional[List[float]] = Field(
                [0], description="Delay between pulses."
            )

            mode: str = Field(
                "max",
                description="Method to calculate pulse amplitudes. max, norm, or both.",
            )
            return_reconstructed: bool = Field(
                False, description="Return the reconstructed waveforms."
            )

        class AutocorrParams(BaseModel):
            class IlluminationParams(BaseModel):
                correction: str = Field(
                    description="Path to correction arrays. One per mask/ROI."
                )

                kernel: int = Field(
                    description="Kernel size used in the creation of the correction."
                )

            name: str = Field("autocorr", description="DetObject name.")

            threshADU: List[float] = Field(
                [-1e6, 1e6], description="Low and high pixel intensity thresholds."
            )

            mask: Optional[str] = Field(
                None, description="Mask to define a non-rectangular ROI."
            )

            save_lineout: bool = Field(
                False,
                description="Save autocorr image or only vertical/horizontal lineouts.",
            )

            save_range: Tuple[int, int] = Field(
                (
                    50,
                    50,
                ),
                description="Size of the autocorr image to save.",
            )

            illumination_correction: Optional[IlluminationParams] = Field(
                None, description="Corrections for each mask/ROI."
            )

        class PressioCompression(BaseModel):
            class Sz3CompressorArgs(BaseModel):
                abs_error_bound: Union[int, float] = Field(
                    default=10,
                    description="The bound on absolute error for the compression.",
                )

            compressor_id: Literal["sz3"] = Field(
                default=None, description="The type of compressor to use."
            )
            compressor_args: Sz3CompressorArgs = Field(
                default=Sz3CompressorArgs(),
                description="Compressor specific arguments.",
            )

        detnames: Optional[List[str]] = Field(
            None, description="List of detectors to process."
        )

        xdetectors: Optional[List[str]] = Field(
            None, description="List of detectors to exclude from loading by psana."
        )

        integrating_detectors: Optional[List[str]] = Field(
            None,
            description="List of integrating detectors to process. Only for LCLS2.",
        )

        epicsPV: Optional[List[Union[str, Tuple[str, str]]]] = Field(
            None,
            description=(
                "LCLS1: List of PVs to save once per event. The lists are either "
                "PV names, alias names, or tuples of (PV name, alias name).\n"
                "LCLS2: List of PVs to grab from the ARCHIVER. The lists are either "
                "PV names or tuples of (PV name, alias name).\n"
                "NOTE: It appears to be shot-to-shot, but this is for convenience only."
            ),
        )

        epicsArchFilePV: Optional[List[Union[str, Tuple[str, str]]]] = Field(
            None,
            description=(
                "List PVs to save shot-to-shot. NOTE: It appears to be shot-to-shot, "
                "but this is for convenience only. The lists are either PV names, "
                "alias names, or tuples of (PV name, alias name)."
            ),
        )

        epicsOncePV: Optional[List[Union[str, Tuple[str, str]]]] = Field(
            None,
            description=(
                "List of PVs to save once per run.The lists are either "
                "PV names, alias names, or tuples of (PV name, alias name)."
            ),
        )

        ttCalib: Optional[List[float]] = Field(
            None, description="Alternative calibration parameters for the timetool."
        )

        aioParams: Optional[List[List[Union[str, int, float]]]] = Field(
            None,
            description="Save analog inputs and give them nice names. [[inp],['name']]",
        )

        get_intg: Optional[IntgParams] = Field(
            None, description="Integrating detector configuration. LCLS2 only."
        )

        getROIs: Optional[Dict[str, List[ROIParams]]] = Field(
            None, description="Dictionary of ROI parameters by detector."
        )

        getAzIntParams: Optional[Dict[str, AzIntParams]] = Field(
            None,
            description="Dictionary of azimuthal integration parameters by detector.",
        )

        getAzIntPyFAIParams: Optional[Dict[str, AzIntPyFAIParams]] = Field(
            None,
            description="Dictionary of azimuthal integration with PyFAI parameters.",
        )

        getPhotonParams: Optional[Dict[str, PhotonParams]] = Field(
            None,
            description="Dictionary of photon counting parameters by detector.",
        )

        getDropletParams: Optional[Dict[str, DropletParams]] = Field(
            None,
            description="Dictionary of droplet finding parameters by detector.",
        )

        getDroplet2Photons: Optional[Dict[str, Droplet2PhotonParams]] = Field(
            None,
            description="Dictionary of droplet2photon parameters by detector.",
        )

        getSvdParams: Optional[Dict[str, SvdParams]] = Field(
            None,
            description="Dictionary of SVD parameters by detector.",
        )

        getAutocorrParams: Optional[Dict[str, AutocorrParams]] = Field(
            None,
            description="Dictionary of auto-correlation parameters by detector.",
        )

        getPressioCompression: Optional[Dict[str, PressioCompression]] = Field(
            None, description="Per detector compression arguments."
        )

        detSumAlgos: Optional[Dict[str, List[str]]] = Field(
            None,
            description=(
                "Detector sum algorithms. Can add a key for `all` to apply the algorithm "
                "to all detectors, in addition to providing algorithms for each detector "
                "individually."
            ),
        )

    _set_producer_template_parameters = template_parameter_validator(
        "producer_parameters"
    )

    executable: str = Field("mpirun", description="MPI executable.", flag_type="")
    np: PositiveInt = Field(
        max(int(os.environ.get("SLURM_NPROCS", len(os.sched_getaffinity(0)))) - 1, 1),
        description="Number of processes",
        flag_type="-",
    )
    # This will interfere with affinities set by executor -- need smarter handling
    map_by: Optional[str] = Field(
        None, description="MPI rank mapping.", flag_type="--", rename_param="map-by"
    )
    bind_to: Optional[str] = Field(
        None,
        description="MPI rank to resource binding.",
        flag_type="--",
        rename_param="bind-to",
    )
    p_arg1: str = Field(
        "python", description="Executable to run with mpi (i.e. python).", flag_type=""
    )
    u: str = Field(
        "", description="Python option for unbuffered output.", flag_type="-"
    )
    m: str = Field(
        "mpi4py.run",
        description="Python option to execute a module's contents as __main__ module.",
        flag_type="-",
    )
    producer: str = Field(
        "", description="Path to the SmallData producer Python script.", flag_type=""
    )
    run: str = Field(
        os.environ.get("RUN_NUM", ""), description="DAQ Run Number.", flag_type="--"
    )
    experiment: str = Field(
        os.environ.get("EXPERIMENT", ""),
        description="LCLS Experiment Number.",
        flag_type="--",
    )
    stn: NonNegativeInt = Field(0, description="Hutch endstation.", flag_type="--")
    psdm_dir: Optional[str] = Field(
        None,
        description="Optionally override SIT_PSDM_DATA.",
        flag_type="--",
    )
    config: Optional[str] = Field(
        None,
        description="Alternative config file to use for producer configuration",
        flag_type="--",
    )
    nevents: int = Field(
        int(1e9), description="Number of events to process.", flag_type="--"
    )
    directory: Optional[str] = Field(
        None,
        description="Optional output directory. If None, will be in ${EXP_FOLDER}/hdf5/smalldata.",
        flag_type="--",
    )
    ## Need mechanism to set result_from_param=True ...
    gather_interval: PositiveInt = Field(
        25, description="Number of events to collect at a time.", flag_type="--"
    )
    norecorder: bool = Field(
        False, description="Whether to ignore recorder streams.", flag_type="--"
    )
    url: HttpUrl = Field(
        "https://pswww.slac.stanford.edu/ws-auth/lgbk",
        description="Base URL for eLog posting.",
        flag_type="--",
    )
    epicsAll: bool = Field(
        False,
        description="Whether to store all EPICS PVs. Use with care.",
        flag_type="--",
    )
    full: bool = Field(
        False,
        description="Whether to store all data. Use with EXTRA care.",
        flag_type="--",
    )
    fullSum: bool = Field(
        False,
        description="Whether to store sums for all area detector images.",
        flag_type="--",
    )
    default: bool = Field(
        False,
        description="Whether to store only the default minimal set of data.",
        flag_type="--",
    )
    image: bool = Field(
        False,
        description="Whether to save everything as images. Use with care.",
        flag_type="--",
    )
    tiff: bool = Field(
        False,
        description="Whether to save all images as a single TIFF. Use with EXTRA care.",
        flag_type="--",
    )
    centerpix: bool = Field(
        False,
        description="Whether to mask center pixels for Epix10k2M detectors.",
        flag_type="--",
    )
    postRuntable: bool = Field(
        False,
        description="Whether to post run tables. Also used as a trigger for summary jobs.",
        flag_type="--",
    )
    wait: bool = Field(
        False, description="Whether to wait for a file to appear.", flag_type="--"
    )
    xtcav: bool = Field(
        False,
        description="Whether to add XTCAV processing to the HDF5 generation.",
        flag_type="--",
    )
    noarch: bool = Field(
        False, description="Whether to not use archiver data.", flag_type="--"
    )

    lute_template_cfg: TemplateConfig = TemplateConfig(
        template_name="smd_producer_template.py", output_path=""
    )

    producer_parameters: Optional[ProducerParameters] = Field(
        None,
        description="Optional parameters to fill in a producer file.",
        flag_type="",  # Does nothing since always None by time it's seen by Task
    )

    @validator("producer", always=True)
    def validate_producer_path(cls, producer: str, values: Dict[str, Any]) -> str:
        if producer == "":
            exp: str = values["lute_config"].experiment
            hutch: str = exp[:3]
            base_path: str = f"/sdf/data/lcls/ds/{hutch}/{exp}/results/smalldata_tools"
            path: str
            is_daq2: bool
            if "psdm_dir" in values and values["psdm_dir"]:
                # Assume we are only overriding SIT_PSDM_DATA if using converted xtc2
                is_daq2 = True
            else:
                try:
                    import psana  # type: ignore

                    _ = psana.xtc_version
                    # xtc_version fails in psana1
                    is_daq2 = True
                except AttributeError:
                    is_daq2 = False
            if not is_daq2:
                path = f"{base_path}/lcls1_producers/smd_producer.py"
            else:
                path = f"{base_path}/lcls2_producers/smd_producer.py"
            return path
        return producer

    @validator("lute_template_cfg", always=True)
    def use_producer(
        cls, lute_template_cfg: TemplateConfig, values: Dict[str, Any]
    ) -> TemplateConfig:
        exp: str = values["lute_config"].experiment
        hutch: str = exp[:3]
        if not lute_template_cfg.output_path:
            cfg: str
            if values["config"] is not None:
                prod_cfg: str = f"prod_config_{values['config']}"
                cfg = str(Path(values["producer"]).parent / prod_cfg)
            else:
                cfg = str(Path(values["producer"]).parent / f"prod_config_{hutch}.py")
            lute_template_cfg.output_path = cfg
            is_daq2: bool
            if "psdm_dir" in values and values["psdm_dir"]:
                # Assume we are only overriding SIT_PSDM_DATA if using converted xtc2
                is_daq2 = True
            else:
                # Try using xtc_version now available in psana2 to figure out lcls1
                # or lcls2
                try:
                    import psana  # type: ignore

                    _ = psana.xtc_version
                    # xtc_version fails in psana1
                    is_daq2 = True
                except AttributeError:
                    is_daq2 = False
            if not is_daq2:
                lute_template_cfg.template_name = "smd1_prod_config_template.py"
            else:
                lute_template_cfg.template_name = "smd2_prod_config_template.py"
        return lute_template_cfg

    @root_validator(pre=False)
    def define_result(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        exp: str = values["lute_config"].experiment
        hutch: str = exp[:3]
        run: int = int(values["lute_config"].run)
        directory: Optional[str] = values["directory"]
        if directory is None:
            directory = f"/sdf/data/lcls/ds/{hutch}/{exp}/hdf5/smalldata"
        fname: str = f"{exp}_Run{run:04d}.h5"

        cls.Config.result_from_params = f"{directory}/{fname}"
        return values


class AnalyzeSmallDataXSSParameters(TaskParameters):
    """TaskParameter model for AnalyzeSmallDataXSS Task.

    This Task does basic analysis of XSS data based on a SmallData HDF5 output
    file. It calculates difference scattering and signal binned by various
    scanned motors.
    """

    class Thresholds(BaseModel):
        min_Iscat: float = Field(
            10.0, description="Minimum scattering intensity to use for filtering."
        )
        min_ipm: float = Field(
            1000.0, description="Minimum X-ray intensity to use for filtering."
        )

    class AnalysisFlags(BaseModel):
        use_pyfai: bool = True
        use_asymls: bool = False

    _find_smd_path = validate_smd_path("smd_path")

    smd_path: str = Field(
        "", description="Path to the Small Data HDF5 file to analyze."
    )
    xss_detname: Optional[str] = Field(
        None, description="Name of the detector with scattering data."
    )
    ipm_var: str = Field(
        description="Name of the IPM to use for X-Ray intensity filtering."
    )
    scan_var: Optional[Union[List[str], str]] = Field(
        None,
        description=(
            "Name of a scan variable or a list of scan variables to analyze. "
            "E.g. lxt, lens_h, etc."
        ),
    )
    thresholds: Thresholds = Field(Thresholds(min_Iscat=10.0, min_ipm=1000.0))
    # analysis_flags: AnalysisFlags


class AnalyzeSmallDataXASParameters(TaskParameters):
    """TaskParameter model for AnalyzeSmallDataXAS Task.

    This Task does basic analysis of XAS data based on a SmallData HDF5 output
    file. It calculates difference absorption and signal binned by various
    scanned motors.
    """

    class Thresholds(BaseModel):
        min_Iscat: float = Field(
            10.0, description="Minimum scattering intensity to use for filtering."
        )
        min_ipm: float = Field(
            1000.0, description="Minimum X-ray intensity to use for filtering."
        )

    _find_smd_path = validate_smd_path("smd_path")

    smd_path: str = Field(
        "", description="Path to the Small Data HDF5 file to analyze."
    )
    xas_detname: str = Field(description="Name of the detector with absorption data.")
    xss_detname: Optional[str] = Field(
        None,
        description="Name of the detector with scattering data, for normalization.",
    )
    ipm_var: str = Field(
        description="Name of the IPM to use for X-Ray intensity filtering."
    )
    scan_var: Optional[Union[List[str], str]] = Field(
        None,
        description=(
            "Name of a scan variable or a list of scan variables to analyze. "
            "E.g. lxt, lens_h, etc."
        ),
    )
    ccm: str = Field(description="Name of the PV for CCM position readback.")
    ccm_set: Optional[str] = Field(
        None, description="Name of the PV for the setpoint of the CCM."
    )
    thresholds: Thresholds = Field(Thresholds(min_Iscat=10.0, min_ipm=1000.0))
    element: Optional[str] = Field(
        None,
        description="Element under investigation. Currently unused. For future EXAFS.",
    )


class AnalyzeSmallDataXESParameters(TaskParameters):
    """TaskParameter model for AnalyzeSmallDataXES Task.

    This Task does basic analysis of XES data based on a SmallData HDF5 output
    file. It calculates difference emission and signal binned by various
    scanned motors.
    """

    class Thresholds(BaseModel):
        min_Iscat: float = Field(
            10.0, description="Minimum scattering intensity to use for filtering."
        )
        min_ipm: float = Field(
            1000.0, description="Minimum X-ray intensity to use for filtering."
        )

    _find_smd_path = validate_smd_path("smd_path")

    smd_path: str = Field(
        "", description="Path to the Small Data HDF5 file to analyze."
    )
    xes_detname: str = Field(description="Name of the detector with absorption data.")
    xss_detname: Optional[str] = Field(
        None,
        description="Name of the detector with scattering data, for normalization.",
    )
    ipm_var: str = Field(
        description="Name of the IPM to use for X-Ray intensity filtering."
    )
    scan_var: Optional[Union[List[str], str]] = Field(
        None,
        description=(
            "Name of a scan variable or a list of scan variables to analyze. "
            "E.g. lxt, lens_h, etc."
        ),
    )
    thresholds: Thresholds = Field(Thresholds(min_Iscat=10.0, min_ipm=1000.0))
    invert_xes_axes: bool = Field(
        False,
        description=(
            "Flip the projection axes depending on detector orientation. "
            "Default is that projection along axis 1 is spectrum."
        ),
    )
    rot_angle: Optional[float] = Field(
        None,
        description="Optionally rotate the ROIs by a small amount before projection.",
    )
    batch_size: int = Field(
        0,
        description="If non-zero load ROIs in batches. Slower but may help OOM errors.",
    )

class RunBeamlineSummaryParameters(ThirdPartyParameters):
    """Parameters for running Small Data beamline summary scripts (only for MFX for now).

    This task runs the beamline summary scripts for MFX that generate summary plots for each run based on the Small Data HDF5 file.
    """

    class Config(ThirdPartyParameters.Config):
        """Identical to super-class Config but includes a result."""

        set_result: bool = False
        """Whether the Executor should mark a specified parameter as a result."""

    executable: str = Field("python", description="Python executable.", flag_type="")
    python_script: str = Field(
        description="Path to the beamline summary Python script",
        flag_type="",
    )

    _find_smd_path = validate_smd_path("smd_path")

    smd_path: str = Field(
        "",
        description="Path to the Small Data HDF5 file to analyze.",
        flag_type="--",
    )

    experiment: str = Field(
        os.environ.get("EXPERIMENT", ""),
        description="Experiment Tag.",
        flag_type="--",
    )

    run: str = Field(
        os.environ.get("RUN_NUM", ""),
        description="DAQ Run Number.",
        flag_type="--",
    )

    postElog: bool = Field(
        False,
        description="Whether to post summary plots to the eLog. Requires url and credentials in environment variables.",
        flag_type="--",
    )