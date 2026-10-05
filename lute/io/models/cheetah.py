"""Models for running the Cheetah Task

Classes:
    RunCheetahParameters(ThirdPartyParameters): Parameters to run Cheetah.
    More information on Cheetah: https://github.com/omdevteam/om
"""

__all__ = ["RunCheetahParameters"]

import os
import warnings
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, root_validator, validator

from lute.io.models.base import TemplateConfig, ThirdPartyParameters
from lute.io.models.sfx_find_peaks import SZCompressorParameters
from lute.io.models.validators import template_parameter_validator

CommonDataSourceTypes = Literal[
    "TimestampFromEvent",
    "FloatValueFromConfiguration",
    "IntValueFromConfiguration",
    "ArrayFromHdf5File",
]

PsanaDataSourceTypes = Literal[
    "RayonixPsana",
    "OpalPsana",
    "Epix100aPsana",
    "AcqirisPsana",
    "AssembledDetector",
    "Wave8TotalIntensity",
    "EpicsVariablePsana",
    "DiodeTotalIntensity",
    "BeamEnergyFromEpicsVariablePsana",
    "AreaDetectorPsana",
    "CspadPsana",
    "TimestampPsana",
    "EventIdPsana",
    "BeamEnergyPsana",
    "EvrCodesPsana",
    "EvrCodelistPsana",
    "LclsExtraPsana",
]

Psana2DataSourceTypes = Literal[
    "AssembledDetector2",
    "EpicsVariablePsana2",
    "BeamEnergyFromEpicsVariablePsana2",
    "AreaDetectorPsana2",
    "TimestampPsana2",
    "EventIdPsana2",
    "BeamEnergyPsana2",
    "EvrCodelistPsana2",
]

ParallelizationLayerTypes = Literal[
    "MpiParallelization",
    "MultiprocessingParallelization",
]

RetrievalLayerTypes = Literal[
    "PsanaDataEventHandler",
    "Psana2DataEventHandler",
    # Probably won't be used - at LCLS at any rate.
    "AsapoDataEventHandler",
    "Jungfrau1MZmqDataEventHandler",
    "PilatusFilesEventHandler",
    "Jungfrau1MFilesDataEventHandler",
    "EigerFilesDataEventHandler",
    "RayonixMccdFilesEventHandler",
    "Lambda1M5FilesDataEventHandler",
    "EigerHttpDataEventHandler",
]


class RunCheetahParameters(ThirdPartyParameters):
    """Parameters for running OM Cheetah Task."""

    class Config(ThirdPartyParameters.Config):
        """Identical to super-class Config but includes a result."""

        set_result: bool = True
        """Whether the Executor should mark a specified parameter as a result."""

        result_from_params: str = ""
        """Defines a result from the parameters. Use a validator to do so."""

    class CheetahSubconfigParameters(BaseModel):
        """Parameters for OM Cheetah itself."""

        class OmParameters(BaseModel):
            """Parameters for OM layers"""

            parallelization_layer: ParallelizationLayerTypes = Field(
                default="MpiParallelization",
                description="What type of parallelization use (MPI or multiprocess.)",
            )
            data_retrieval_layer: RetrievalLayerTypes = Field(
                default="PsanaDataEventHandler",
                description="What kind of data retrieval to use (e.g. psana).",
            )
            processing_layer: str = Field(
                default="CheetahProcessing",
                description="What type of processing to perform (e.g. Cheetah for Crystallography.)",
            )

        class DataRetrievalLayerParameters(BaseModel):
            """Parameters for the data retrieving layer."""

            class PsanaDataSourcesParameters(BaseModel):
                """Parameters for the data retrieval layer data sources subfields."""

                class PsanaDataSourceTimestamp(BaseModel):
                    """Parameters for retrieving the event ID."""

                    type: Optional[
                        Union[
                            CommonDataSourceTypes,
                            PsanaDataSourceTypes,
                            Psana2DataSourceTypes,
                        ]
                    ] = Field(
                        default="TimestampPsana",
                        description="Timestamp object for the retrieval layer.",
                    )
                    psana_name: str = Field(default="EventID", description="")

                class PsanaDataSourceEventId(BaseModel):
                    """Parameters for retrieving the event ID."""

                    type: Optional[
                        Union[
                            CommonDataSourceTypes,
                            PsanaDataSourceTypes,
                            Psana2DataSourceTypes,
                        ]
                    ] = Field(
                        default="EventIdPsana",
                        description="Event ID object for the retrieval layer.",
                    )

                class PsanaDataSourceDetectorDataParameters(BaseModel):
                    """Data retrieval layer, datasource detector data parameters."""

                    type: Optional[
                        Union[
                            CommonDataSourceTypes,
                            PsanaDataSourceTypes,
                            Psana2DataSourceTypes,
                        ]
                    ] = Field(
                        default="AreaDetectorPsana",
                        description="Type name for retrieving the detector data.",
                    )
                    psana_name: str = Field(
                        default="epix10k2M",
                        description="Name of the specific psana detector.",
                    )
                    calibration: bool = Field(
                        default=True, description="Whether to use calibration."
                    )
                    psana_algorithm: str = Field(
                        default="raw",
                        description=(
                            "The psana2 `algorithm` for accessing the data. "
                            "E.g. `raw`, `fex`"
                        ),
                    )

                class PsanaDataSourceDetectorDistanceParameters(BaseModel):
                    """Data retrieval layer, datasource detector distance parameters."""

                    type: Optional[
                        Union[
                            CommonDataSourceTypes,
                            PsanaDataSourceTypes,
                            Psana2DataSourceTypes,
                        ]
                    ] = Field(
                        default="EpicsVariablePsana",
                        description="Type name for retrieving the detector distance variable.",
                    )
                    psana_name: Optional[str] = Field(
                        default="detector_z",
                        description="The specific detector name in psana.",
                    )
                    value: Union[float, int, str] = Field(
                        default="DetectorDistanceValue",
                        description=(
                            "Value type for retrieving the data from the psana detector. "
                            "Or this is the value if you are providing it from the YAML."
                        ),
                    )

                    @validator("psana_name", always=True)
                    def validate_psana_name(
                        cls, psana_name: Optional[str], values: Dict[str, Any]
                    ) -> Optional[str]:
                        common_types: List[str] = [
                            t for t in vars(CommonDataSourceTypes)["__args__"]
                        ]

                        source_type: str = values["type"]
                        if source_type in common_types:
                            if psana_name is not None:
                                warnings.warn(
                                    f"For `type` {source_type} psana_name field is not used! "
                                    "We will ignore this. But check your configuration!"
                                )
                            return None
                        else:
                            if not isinstance(psana_name, str):
                                raise ValueError(
                                    "`psana_name` must be a string for the psana detector to use."
                                )
                        return psana_name

                    @validator("value", always=True)
                    def validate_value_field(
                        cls, value: Union[float, int, str], values: Dict[str, Any]
                    ) -> Union[float, int, str]:
                        common_types: List[str] = [
                            t for t in vars(CommonDataSourceTypes)["__args__"]
                        ]

                        source_type: str = values["type"]
                        if source_type in common_types:
                            if source_type == "IntValueFromConfiguration":
                                try:
                                    new_val_int: int = int(value)
                                    return new_val_int
                                except ValueError:
                                    raise ValueError(
                                        f"Must provide an int value if using type {source_type}! "
                                        f"Provided {value}."
                                    )

                            elif source_type == "FloatValueFromConfiguration":
                                try:
                                    new_val_float: float = float(value)
                                    return new_val_float
                                except ValueError:
                                    raise ValueError(
                                        f"Must provide a float value if using type {source_type}! "
                                        f"Provided {value}."
                                    )
                            else:
                                raise ValueError(
                                    f"`type` of {source_type} not supported for Detector Distance."
                                )

                        else:
                            return value

                class PsanaDataSourceBeamEnergy(BaseModel):
                    """Parameters for retrieving the event ID."""

                    type: Optional[
                        Union[
                            CommonDataSourceTypes,
                            PsanaDataSourceTypes,
                            Psana2DataSourceTypes,
                        ]
                    ] = Field(
                        default="BeamEnergyPsana",
                        description="Beam energy object for the retrieval layer.",
                    )
                    value: Optional[Union[str, int, float]] = Field(
                        default=None,
                        value="If providing a value from the configuration file this is where to put it.",
                    )

                    @validator("value", always=True)
                    def validate_value_field(
                        cls, value: Union[float, int, str], values: Dict[str, Any]
                    ) -> Optional[Union[float, int, str]]:
                        common_types: List[str] = [
                            t for t in vars(CommonDataSourceTypes)["__args__"]
                        ]

                        source_type: str = values["type"]
                        if source_type in common_types:
                            if source_type == "IntValueFromConfiguration":
                                try:
                                    new_val_int: int = int(value)
                                    return new_val_int
                                except ValueError:
                                    raise ValueError(
                                        f"Must provide an int value if using type {source_type}! "
                                        f"Provided {value}."
                                    )

                            elif source_type == "FloatValueFromConfiguration":
                                try:
                                    new_val_float: float = float(value)
                                    return new_val_float
                                except ValueError:
                                    raise ValueError(
                                        f"Must provide a float value if using type {source_type}! "
                                        f"Provided {value}."
                                    )
                            else:
                                raise RuntimeError(
                                    f"`type` of {source_type} not supported for Detector Distance."
                                )

                        else:
                            if value is not None:
                                warnings.warn(
                                    f"For `type` {source_type} no `value` field is expected! "
                                    "We will ignore this. But check your configuration!"
                                )
                            return None

                timestamp: PsanaDataSourceTimestamp = Field(
                    description="Fields for timestamp retrieval."
                )
                event_id: PsanaDataSourceEventId = Field(
                    description="Fields for event ID retrieval."
                )
                detector_data: PsanaDataSourceDetectorDataParameters = Field(
                    description="Fields for accessing the detector data (images) via psana.",
                )
                detector_distance: PsanaDataSourceDetectorDistanceParameters = Field(
                    description="Fields for accessing the detector distance via psana.",
                )
                beam_energy: PsanaDataSourceBeamEnergy = Field(
                    default=PsanaDataSourceBeamEnergy(),
                    description="Fields for beam energy retrieval.",
                )

            # asapo
            asapo_url: Optional[str] = Field(
                default=None, desription="ASAPO url. (Not LCLS)."
            )
            asapo_path: Optional[str] = Field(
                default=None, description="ASAPO path. (Not LCLS)."
            )
            asapo_data_source: Optional[str] = Field(
                default=None, description="ASAPO data source. (Not LCLS)."
            )
            asapo_has_filesystem: Optional[bool] = Field(
                default=None, description="ASAPO use filesystem. (Not LCLS)."
            )
            asapo_token: Optional[str] = Field(
                default=None, description="ASAPO token. (Not LCLS)."
            )
            asapo_group_id: str = Field(
                default="default_om_group", description="ASPAO group ID. (Not LCLS)."
            )
            # http
            buffer_size: Optional[int] = Field(
                default=None, description="Buffer size for HTTP access. (Not LCLS)."
            )
            # psana
            psana_calibration_directory: Optional[str] = Field(
                default=None, description="Path to the psana calibration directory."
            )

            # all sources
            data_sources: Optional[PsanaDataSourcesParameters] = Field(
                default=None, description="Configuration for data source retrieval."
            )
            node_pool_size: int = Field(default=0, description="Node pool size.")

        class CheetahParameters(BaseModel):
            """Parameters for Cheetah."""

            class CheetahHDF5Parameters(BaseModel):
                detector_data: str = Field(
                    default="/entry_1/data_1/data",
                    description="Path to write the detector data (images) in the HDF5 file.",
                )
                event_id: str = Field(
                    default="/LCLS/fiducial",
                    description="Path to write the fiducial/event ID in the HDF5 file.",
                )
                beam_energy: str = Field(
                    default="/LCLS/photon_energy_eV",
                    description="Path to write the photon energy in the HDF5 file.",
                )
                detector_distance: str = Field(
                    default="/LCLS/detector_1/EncoderValue",
                    description="Path to write the detector distance in the HDF5 file.",
                )
                timestamp: str = Field(
                    default="/LCLS/timestamp",
                    description="Path to write the timestamp in the HDF5 file.",
                )
                peak_list: str = Field(
                    default="/entry_1/result_1",
                    description="Path to write the peak list to in the HDF5 file.",
                )

            processed_directory: str = Field(
                description="The output directory.",
            )
            processed_filename_prefix: str = Field(
                description="The filename prefix to use.",
            )
            processed_filename_extension: str = Field(
                default="cxi",
                description="The filename extension for the HDF5 file.",
            )
            write_class_sums: bool = Field(
                default=True,
                description="Whether to write class sums.",
            )
            class_sums_update_interval: int = Field(
                default=5,
                description="How frequently to update the class sums.",
            )
            class_sums_sending_interval: int = Field(
                default=-1,
                description="How frequently to send the class sums.",
            )
            class_sums_filename_prefix: str = Field(
                default="sums",
                description="The filename prefix for the class sums.",
            )
            status_file_update_interval: int = Field(
                default=100,
                description="How frequently to write updates to the status file.",
            )
            hdf5_file_data_type: str = Field(
                default="float32",
                description="Data type for the output in the HDF5 file.",
            )
            hdf5_file_compression: Optional[Literal["gzip", "bitshuffle_with_zstd"]] = (
                Field(
                    default=None,
                    description="Compression method to use for the HDF5 data.",
                )
            )
            hdf5_file_gzip_compression_level: int = Field(
                default=4,
                description="If using GZip compression, the compression level to apply.",
            )
            hdf5_file_zstd_compression_level: int = Field(
                default=3,
                description="If using ZStd compression, the compression level to apply.",
            )
            hdf5_file_compression_shuffle: bool = Field(
                default=False,
                description="Whether to use bitshuffle.",
            )
            hdf5_file_max_num_peaks: int = Field(
                default=2048,
                description="The maximum number of peaks to write out in the HDF5 file.",
            )
            hdf5_fields: CheetahHDF5Parameters = Field(
                default=CheetahHDF5Parameters(),
                description="Path structure for the data written into the HDF5 file.",
            )
            external_data_request_list_size: int = Field(
                default=20,
                description="Size of the lists for external requests.",
            )
            responding_url: Optional[str] = Field(
                default=None,
                description="Response URL.",
            )

        class CrystallographyParameters(BaseModel):
            """Parameters for Crystallography."""

            geometry_file: Optional[str] = Field(
                default="/sdf/scratch/users/k/kmecseki/cheetah/test.geom",
                description="Path to the detector geometry file (CrystFEL format).",
            )
            min_num_peaks_for_hit: Optional[int] = Field(
                default=10,
                description="Minimum number of hits to consider an event a hit.",
            )
            max_num_peaks_for_hit: Optional[int] = Field(
                default=5000,
                description="Maximum number of peaks to consider an event a hit.",
            )
            running_average_window_size: Optional[int] = Field(
                default=200,
                description="The window size (in events) for the running average.",
            )
            geometry_is_optimized: Optional[bool] = Field(
                default=False,
                description="Whether the input geometry is optimized.",
            )
            speed_report_interval: Optional[int] = Field(
                default=100,
                description="Reporting interval in events.",
            )
            data_broadcast_interval: Optional[int] = Field(
                default=0,
                description="Interval for broadcast.",
            )

        class Peakfinder8PeakDetectionParameters(BaseModel):
            """Parameters for peak finder 8 peak detection."""

            max_num_peaks: int = Field(
                default=2048,
                description="The maximum number of peaks to extract from an image.",
            )
            adc_threshold: float = Field(
                default=100.0,
                description="The ADC value to use as a threshold.",
            )
            minimum_snr: float = Field(
                default=5.0,
                description="The minimum signal to noise ratio to consider a pixel part of a peak.",
            )
            min_pixel_count: int = Field(
                default=1,
                description="The minimum number of pixels a peak must have.",
            )
            max_pixel_count: int = Field(
                default=30,
                description="The maximum number of pixels a peak can have.",
            )
            local_bg_radius: int = Field(
                default=4,
                description="Radius to use for the local background calculation portion. (Pixels)",
            )
            min_res: int = Field(
                default=80,
                description="The minimum radius (in pixels) to use for the peak search space.",
            )
            max_res: int = Field(
                default=800,
                description="The maximum radius (in pixels) to use for the peak search space.",
            )
            fast_mode: bool = Field(
                default=False,
                description="Faster calculatation.",
            )
            num_pixel_per_bin_in_radial_statistics: int = Field(
                default=100,
                description="",
            )
            bad_pixel_map_filename: Optional[str] = Field(
                default=None,
                description="Filename of an HDF5 file with a bad pixel mask to apply to images.",
            )
            bad_pixel_map_hdf5_path: Optional[str] = Field(
                default="/data/data",
                description="The path in the HDF5 file to retrieve the mask (if provided).",
            )

        class DataCompressionParameters(BaseModel):
            run_compression: bool = Field(
                default=False,
                description="Whether to enable compression.",
            )
            backend: Optional[Literal["roibinsz"]] = Field(
                default=None,
                description=(
                    "The compression backend to use. Currently only RoiBinSz "
                    "with libpressio is available."
                ),
            )
            compression_parameters: Optional[SZCompressorParameters] = Field(
                default=None,
                description="The actual compression parameters. Varies by backend.",
            )

            @root_validator(pre=False)
            def check_backend_matches_parameters(
                cls, values: Dict[str, Any]
            ) -> Dict[str, Any]:
                if values["backend"] == "roibinsz":
                    if not isinstance(
                        values["compression_parameters"], SZCompressorParameters
                    ):
                        raise ValueError(
                            "For the libpressio RoiBinSz compression backend you must "
                            "use SZCompressorParameters for `compression_parameters`."
                        )
                return values

        om: OmParameters = Field(OmParameters(), description="Global options for OM.")
        data_retrieval_layer: DataRetrievalLayerParameters = Field(
            default=DataRetrievalLayerParameters(),
            description="Options for configuring the data retrieval.",
        )
        cheetah: CheetahParameters = Field(
            default=CheetahParameters(
                processed_directory=(
                    f'/sdf/data/lcls/ds/{os.getenv("EXPERIMENT","xpptut15")[:3]}/'
                    f'{os.getenv("EXPERIMENT","xpptut15")}/results'
                ),
                processed_filename_prefix=(
                    f'{os.getenv("EXPERIMENT","xpptut15")}-{os.getenv("RUN_NUM","9999")}'
                ),
            ),
            description="Cheetah specific parameters (like output directories).",
        )
        crystallography: CrystallographyParameters = Field(
            default=CrystallographyParameters(),
            description="Generic crystallography parameters, like geometry file.",
        )
        peakfinder8_peak_detection: Peakfinder8PeakDetectionParameters = Field(
            default=Peakfinder8PeakDetectionParameters(),
            description="Peakfinder8 peak finding algorithm specific parameters.",
        )
        compression: Optional[DataCompressionParameters] = Field(
            default=None, description="Data compression parameters."
        )

    _set_cheetah_template_parameters = template_parameter_validator("cheetah_subconfig")

    executable: str = Field(
        "/sdf/group/lcls/ds/tools/openmpi-5.0.6/bin/mpirun",
        description="MPI executable.",
        flag_type="",
    )
    bind_to: str = Field(
        default="core",
        description="MPI rank-to-resource binding. You may want to set to `none` if using compression.",
        flag_type="--",
        rename_param="bind-to",
    )
    n_processes: int = Field(
        4,
        description="Number of processes to launch.",
        flag_type="-",
        rename_param="n",
    )
    om_executable: str = Field(
        "/sdf/group/lcls/ds/tools/om/om/om-071725/bin/om_monitor.py",
        description="OM Cheetah binary.",
        flag_type="",
    )
    source: str = Field(
        "exp=mfx100903824:run=27",
        description="Data source string.",
        flag_type="",
    )
    om_config: str = Field(
        "/sdf/scratch/users/k/kmecseki/cheetah/test.yaml",
        description="Config file for OM Cheetah.",
        flag_type="-",
        rename_param="c",
    )
    cheetah_subconfig: Optional[CheetahSubconfigParameters] = Field(
        None,
        description="The parameter set for the OM configuration YAML.",
        flag_type="",  # Does nothing since None by time it's seen by Task
    )
    lute_template_cfg: TemplateConfig = Field(
        TemplateConfig(
            template_name="cheetah_template.yaml",
            output_path="/sdf/scratch/users/k/kmecseki/cheetah/cheetah_testconfig.yaml",
        ),
        description="Template rendering configuration",
    )

    @validator("lute_template_cfg", always=True)
    def update_output_path(
        cls, lute_template_cfg: TemplateConfig, values: Dict[str, Any]
    ) -> TemplateConfig:
        if lute_template_cfg.output_path == "":
            lute_template_cfg.output_path = values["om_config"]
        return lute_template_cfg

    @root_validator(pre=False)
    def define_result(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        out_dir: str = values["cheetah"].params.processed_directory
        prefix: str = values["cheetah"].params.processed_filename_prefix
        extension: str = values["cheetah"].params.processed_filename_extension
        n_processes: int = values["n_processes"]

        file_name_base: str = f"{out_dir}/{prefix}"
        img_list_file_name: str = f"{out_dir}/{prefix}_images.lst"
        os.makedirs(out_dir, exist_ok=True)
        with open(img_list_file_name, "w") as img_list_file:
            for i in range(1, n_processes):
                img_list_file.write(f"{file_name_base}_{i}.{extension}\n")

        cls.Config.result_from_params = img_list_file_name
        return values
