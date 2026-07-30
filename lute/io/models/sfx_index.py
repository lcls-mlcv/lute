"""Models for serial femtosecond crystallography indexing.

Classes:
    IndexCrystFELParameters(ThirdPartyParameters): Perform indexing of hits/peaks using
        CrystFEL's `indexamajig`.
"""

__all__ = [
    "IndexCrystFELParameters",
    "ConcatenateStreamFilesParameters",
    "IndexCCTBXXFELParameters",
]
__author__ = "Gabriel Dorlhiac"

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple

from pydantic import (
    BaseModel,
    AnyUrl,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    conint,
    validator,
)

from lute.io.db import read_latest_db_entry
from lute.io.models.base import ThirdPartyParameters, TaskParameters, TemplateConfig
from lute.io.models.validators import template_parameter_validator


class IndexCrystFELParameters(ThirdPartyParameters):
    """Parameters for CrystFEL's `indexamajig`.

    There are many parameters, and many combinations. For more information on
    usage, please refer to the CrystFEL documentation, here:
    https://www.desy.de/~twhite/crystfel/manual-indexamajig.html
    """

    class Config(ThirdPartyParameters.Config):
        set_result: bool = True
        """Whether the Executor should mark a specified parameter as a result."""

        long_flags_use_eq: bool = True
        """Whether long command-line arguments are passed like `--long=arg`."""

    executable: str = Field(
        "/sdf/group/lcls/ds/tools/crystfel/0.10.2/bin/indexamajig",
        description="CrystFEL's indexing binary.",
        flag_type="",
    )
    # Basic options
    in_file: Optional[str] = Field(
        "", description="Path to input file.", flag_type="-", rename_param="i"
    )
    out_file: str = Field(
        "",
        description="Path to output file.",
        flag_type="-",
        rename_param="o",
        is_result=True,
    )
    peaks: Optional[str] = Field(
        None,
        description=(
            "Peak finding algorithm, or file type. E.g. peakfinder8 to peak find, "
            "or use cxi for CXI files."
        ),
        flag_type="--",
    )
    geometry: str = Field(
        "", description="Path to geometry file.", flag_type="-", rename_param="g"
    )
    zmq_input: Optional[str] = Field(
        description="ZMQ address to receive data over. `input` and `zmq-input` are mutually exclusive",
        flag_type="--",
        rename_param="zmq-input",
    )
    zmq_subscribe: Optional[str] = Field(  # Can be used multiple times...
        description="Subscribe to ZMQ message of type `tag`",
        flag_type="--",
        rename_param="zmq-subscribe",
    )
    zmq_request: Optional[AnyUrl] = Field(
        description="Request new data over ZMQ by sending this value",
        flag_type="--",
        rename_param="zmq-request",
    )
    asapo_endpoint: Optional[str] = Field(
        description="ASAP::O endpoint. zmq-input and this are mutually exclusive.",
        flag_type="--",
        rename_param="asapo-endpoint",
    )
    asapo_token: Optional[str] = Field(
        description="ASAP::O authentication token.",
        flag_type="--",
        rename_param="asapo-token",
    )
    asapo_beamtime: Optional[str] = Field(
        description="ASAP::O beatime.",
        flag_type="--",
        rename_param="asapo-beamtime",
    )
    asapo_source: Optional[str] = Field(
        description="ASAP::O data source.",
        flag_type="--",
        rename_param="asapo-source",
    )
    asapo_group: Optional[str] = Field(
        description="ASAP::O consumer group.",
        flag_type="--",
        rename_param="asapo-group",
    )
    asapo_stream: Optional[str] = Field(
        description="ASAP::O stream.",
        flag_type="--",
        rename_param="asapo-stream",
    )
    asapo_wait_for_stream: Optional[str] = Field(
        description="If ASAP::O stream does not exist, wait for it to appear.",
        flag_type="--",
        rename_param="asapo-wait-for-stream",
    )
    data_format: Optional[str] = Field(
        description="Specify format for ZMQ or ASAP::O. `msgpack`, `hdf5` or `seedee`.",
        flag_type="--",
        rename_param="data-format",
    )
    basename: bool = Field(
        False,
        description="Remove directory parts of filenames. Acts before prefix if prefix also given.",
        flag_type="--",
    )
    prefix: Optional[str] = Field(
        description="Add a prefix to the filenames from the infile argument.",
        flag_type="--",
        # rename_param="asapo-stream",
    )
    nthreads: PositiveInt = Field(
        max(int(os.environ.get("SLURM_NPROCS", len(os.sched_getaffinity(0)))) - 1, 1),
        description="Number of threads to use. See also `max_indexer_threads`.",
        flag_type="-",
        rename_param="j",
    )
    no_check_prefix: bool = Field(
        False,
        description="Don't attempt to correct the prefix if it seems incorrect.",
        flag_type="--",
        rename_param="no-check-prefix",
    )
    highres: Optional[float] = Field(
        description="Mark all pixels greater than `x` has bad.", flag_type="--"
    )
    profile: bool = Field(
        False, description="Display timing data to monitor performance.", flag_type="--"
    )
    temp_dir: Optional[str] = Field(
        description="Specify a path for the temp files folder.",
        flag_type="--",
        rename_param="temp-dir",
    )
    wait_for_file: conint(gt=-2) = Field(  # type: ignore
        0,
        description="Wait at most `x` seconds for a file to be created. A value of -1 means wait forever.",
        flag_type="--",
        rename_param="wait-for-file",
    )
    no_image_data: bool = Field(
        False,
        description="Load only the metadata, no iamges. Can check indexability without high data requirements.",
        flag_type="--",
        rename_param="no-image-data",
    )
    # Peak-finding options
    # ....
    # Indexing options
    indexing: Optional[str] = Field(
        description="Comma-separated list of supported indexing algorithms to use. Default is to automatically detect.",
        flag_type="--",
    )
    fromfile_input_file: Optional[str] = Field(
        None,
        description="If replaying, the filename to use for indexing results.",
        flag_type="--",
        rename_param="fromfile-input-file",
    )

    cell_file: Optional[str] = Field(
        description="Path to a file containing unit cell information (PDB or CrystFEL format).",
        flag_type="-",
        rename_param="p",
    )
    tolerance: str = Field(
        "5,5,5,1.5",
        description=(
            "Tolerances (in percent) for unit cell comparison. "
            "Comma-separated list a,b,c,angle. Default=5,5,5,1.5"
        ),
        flag_type="--",
    )
    no_check_cell: bool = Field(
        False,
        description="Do not check cell parameters against unit cell. Replaces '-raw' method.",
        flag_type="--",
        rename_param="no-check-cell",
    )
    no_check_peaks: bool = Field(
        False,
        description="Do not verify peaks are accounted for by solution.",
        flag_type="--",
        rename_param="no-check-peaks",
    )
    multi: bool = Field(
        False, description="Enable multi-lattice indexing.", flag_type="--"
    )
    wavelength_estimate: Optional[float] = Field(
        description="Estimate for X-ray wavelength. Required for some methods.",
        flag_type="--",
        rename_param="wavelength-estimate",
    )
    camera_length_estimate: Optional[float] = Field(
        description="Estimate for camera distance. Required for some methods.",
        flag_type="--",
        rename_param="camera-length-estimate",
    )
    max_indexer_threads: Optional[PositiveInt] = Field(
        # 1,
        description="Some indexing algos can use multiple threads. In addition to image-based.",
        flag_type="--",
        rename_param="max-indexer-threads",
    )
    no_retry: bool = Field(
        False,
        description="Do not remove weak peaks and try again.",
        flag_type="--",
        rename_param="no-retry",
    )
    no_refine: bool = Field(
        False,
        description="Skip refinement step.",
        flag_type="--",
        rename_param="no-refine",
    )
    no_revalidate: bool = Field(
        False,
        description="Skip revalidation step.",
        flag_type="--",
        rename_param="no-revalidate",
    )
    # TakeTwo specific parameters
    taketwo_member_threshold: Optional[PositiveInt] = Field(
        # 20,
        description="Minimum number of vectors to consider.",
        flag_type="--",
        rename_param="taketwo-member-threshold",
    )
    taketwo_len_tolerance: Optional[PositiveFloat] = Field(
        # 0.001,
        description="TakeTwo length tolerance in Angstroms.",
        flag_type="--",
        rename_param="taketwo-len-tolerance",
    )
    taketwo_angle_tolerance: Optional[PositiveFloat] = Field(
        # 0.6,
        description="TakeTwo angle tolerance in degrees.",
        flag_type="--",
        rename_param="taketwo-angle-tolerance",
    )
    taketwo_trace_tolerance: Optional[PositiveFloat] = Field(
        # 3,
        description="Matrix trace tolerance in degrees.",
        flag_type="--",
        rename_param="taketwo-trace-tolerance",
    )
    # Felix-specific parameters
    # felix_domega
    # felix-fraction-max-visits
    # felix-max-internal-angle
    # felix-max-uniqueness
    # felix-min-completeness
    # felix-min-visits
    # felix-num-voxels
    # felix-sigma
    # felix-tthrange-max
    # felix-tthrange-min
    # XGANDALF-specific parameters
    xgandalf_sampling_pitch: Optional[NonNegativeInt] = Field(
        # 6,
        description="Density of reciprocal space sampling.",
        flag_type="--",
        rename_param="xgandalf-sampling-pitch",
    )
    xgandalf_grad_desc_iterations: Optional[NonNegativeInt] = Field(
        # 4,
        description="Number of gradient descent iterations.",
        flag_type="--",
        rename_param="xgandalf-grad-desc-iterations",
    )
    xgandalf_tolerance: Optional[PositiveFloat] = Field(
        # 0.02,
        description="Relative tolerance of lattice vectors",
        flag_type="--",
        rename_param="xgandalf-tolerance",
    )
    xgandalf_no_deviation_from_provided_cell: Optional[bool] = Field(
        description="Found unit cell must match provided.",
        flag_type="--",
        rename_param="xgandalf-no-deviation-from-provided-cell",
    )
    xgandalf_min_lattice_vector_length: Optional[PositiveFloat] = Field(
        # 30,
        description="Minimum possible lattice length.",
        flag_type="--",
        rename_param="xgandalf-min-lattice-vector-length",
    )
    xgandalf_max_lattice_vector_length: Optional[PositiveFloat] = Field(
        # 250,
        description="Minimum possible lattice length.",
        flag_type="--",
        rename_param="xgandalf-max-lattice-vector-length",
    )
    xgandalf_max_peaks: Optional[PositiveInt] = Field(
        # 250,
        description="Maximum number of peaks to use for indexing.",
        flag_type="--",
        rename_param="xgandalf-max-peaks",
    )
    xgandalf_fast_execution: bool = Field(
        False,
        description="Shortcut to set sampling-pitch=2, and grad-desc-iterations=3.",
        flag_type="--",
        rename_param="xgandalf-fast-execution",
    )
    # pinkIndexer parameters
    # ...
    # asdf_fast: bool = Field(False, description="Enable fast mode for asdf. 3x faster for 7% loss in accuracy.", flag_type="--", rename_param="asdf-fast")
    # Integration parameters
    integration: str = Field(
        "rings-nocen", description="Method for integrating reflections.", flag_type="--"
    )
    fix_profile_radius: Optional[float] = Field(
        description="Fix the profile radius (m^{-1})",
        flag_type="--",
        rename_param="fix-profile-radius",
    )
    fix_divergence: Optional[float] = Field(
        0,
        description="Fix the divergence (rad, full angle).",
        flag_type="--",
        rename_param="fix-divergence",
    )
    int_radius: str = Field(
        "4,5,7",
        description="Inner, middle, and outer radii for 3-ring integration.",
        flag_type="--",
        rename_param="int-radius",
    )
    int_diag: str = Field(
        "none",
        description="Show detailed information on integration when condition is met.",
        flag_type="--",
        rename_param="int-diag",
    )
    push_res: str = Field(
        "infinity",
        description="Integrate `x` higher than apparent resolution limit (nm-1).",
        flag_type="--",
        rename_param="push-res",
    )
    overpredict: bool = Field(
        False,
        description="Over-predict reflections. Maybe useful with post-refinement.",
        flag_type="--",
    )
    cell_parameters_only: bool = Field(
        False, description="Do not predict refletions at all", flag_type="--"
    )
    # Output parameters
    no_non_hits_in_stream: bool = Field(
        False,
        description="Exclude non-hits from the stream file.",
        flag_type="--",
        rename_param="no-non-hits-in-stream",
    )
    copy_hheader: Optional[str] = Field(
        description="Copy information from header in the image to output stream.",
        flag_type="--",
        rename_param="copy-hheader",
    )
    no_peaks_in_stream: bool = Field(
        False,
        description="Do not record peaks in stream file.",
        flag_type="--",
        rename_param="no-peaks-in-stream",
    )
    no_refls_in_stream: bool = Field(
        False,
        description="Do not record reflections in stream.",
        flag_type="--",
        rename_param="no-refls-in-stream",
    )
    serial_offset: Optional[PositiveInt] = Field(
        description="Start numbering at `x` instead of 1.",
        flag_type="--",
        rename_param="serial-offset",
    )
    harvest_file: Optional[str] = Field(
        description="Write parameters to file in JSON format.",
        flag_type="--",
        rename_param="harvest-file",
    )

    @validator("in_file", always=True)
    def validate_in_file(cls, in_file: str, values: Dict[str, Any]) -> str:
        if in_file == "":
            # Check Cheetah first
            filename: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}",
                "RunCheetah",
                "result.payload",
            )
            if filename is None:
                # Check FindPeaksSFX
                filename = read_latest_db_entry(
                    f"{values['lute_config'].work_dir}", "FindPeaksSFX", "out_file"
                )
                if filename is None:
                    # Check Psocake last
                    exp: str = values["lute_config"].experiment
                    run: int = int(values["lute_config"].run)
                    tag: Optional[str] = read_latest_db_entry(
                        f"{values['lute_config'].work_dir}", "FindPeaksPsocake", "tag"
                    )
                    psocake_out_dir: Optional[str] = read_latest_db_entry(
                        f"{values['lute_config'].work_dir}",
                        "FindPeaksPsocake",
                        "outDir",
                    )
                    if psocake_out_dir is not None:
                        fname: str = f"{psocake_out_dir}/{exp}_{run:04d}"
                        if tag is not None:
                            fname = f"{fname}_{tag}"
                        filename = f"{fname}.lst"
            if filename is not None:
                return filename
        return in_file

    @validator("out_file", always=True)
    def validate_out_file(cls, out_file: str, values: Dict[str, Any]) -> str:
        if out_file == "":
            # Check Cheetah first
            new_out_file: Optional[str] = None
            cheetah_out_dir: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}",
                "RunCheetah",
                "cheetah.processed_directory",
            )
            if cheetah_out_dir is not None:
                # These should be defined
                cheetah_prefix: Optional[str] = read_latest_db_entry(
                    f"{values['lute_config'].work_dir}",
                    "RunCheetah",
                    "cheetah.processed_filename_prefix",
                )
                new_out_file = f"{cheetah_out_dir}/{cheetah_prefix}.stream"
            else:
                expmt: str = values["lute_config"].experiment
                run: int = int(values["lute_config"].run)
                work_dir: str = values["lute_config"].work_dir
                tag: Optional[str] = read_latest_db_entry(
                    f"{values['lute_config'].work_dir}", "FindPeaksSFX", "tag"
                )
                if tag is None:
                    tag = read_latest_db_entry(
                        f"{values['lute_config'].work_dir}", "FindPeaksPsocake", "tag"
                    )
                fname: str = f"{expmt}_r{run:04d}"
                if tag is not None:
                    fname = f"{fname}_{tag}"
                new_out_file = f"{work_dir}/{fname}.stream"
            if new_out_file is not None:
                return new_out_file
        return out_file


class ConcatenateStreamFilesParameters(TaskParameters):
    """Parameters for stream concatenation.

    Concatenates the stream file output from CrystFEL indexing for multiple
    experimental runs.
    """

    class Config(TaskParameters.Config):
        set_result: bool = True
        """Whether the Executor should mark a specified parameter as a result."""

    in_file: str = Field(
        "",
        description="Root of directory tree storing stream files to merge.",
    )

    tag: Optional[str] = Field(
        "",
        description="Tag identifying the stream files to merge.",
    )

    out_file: str = Field(
        "", description="Path to merged output stream file.", is_result=True
    )

    @validator("in_file", always=True)
    def validate_in_file(cls, in_file: str, values: Dict[str, Any]) -> str:
        if in_file == "":
            stream_file: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}", "IndexCrystFEL", "out_file"
            )
            if stream_file:
                stream_dir: str = str(Path(stream_file).parent)
                return stream_dir
        return in_file

    @validator("tag", always=True)
    def validate_tag(cls, tag: str, values: Dict[str, Any]) -> str:
        if tag == "":
            stream_file: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}", "IndexCrystFEL", "out_file"
            )
            if stream_file:
                stream_tag: str = Path(stream_file).name.split("_")[-1].split(".")[0]
                return stream_tag
        return tag

    @validator("out_file", always=True)
    def validate_out_file(cls, tag: str, values: Dict[str, Any]) -> str:
        if tag == "":
            stream_out_file: str = str(
                Path(values["in_file"]).parent / f"{values['tag']}.stream"
            )
            return stream_out_file
        return tag


class IndexCCTBXXFELParameters(ThirdPartyParameters):
    """Parameters for indexing with cctbx.xfel."""

    class Config(ThirdPartyParameters.Config):
        set_result: bool = False
        """Whether the Executor should mark a specified parameter as a result."""

    class PhilParameters(BaseModel):
        """Template parameters for CCTBX phil file."""

        class Config(BaseModel.Config):  # type: ignore
            extra: str = "allow"

        # Generic input settings: input_
        input_reference_geometry: Optional[str] = Field(
            None,
            description=(
                "Provide an models.expt file with exactly one detector model. Data "
                "processing will use that geometry instead of the geometry found "
                "in the image headers."
            ),
        )

        # Generic geometry: geometry_
        geometry_detector_panel_origin: Optional[Tuple[float, float, float]] = Field(
            None,
            description="Override the panel origin. Requires fast_axis and slow_axis.",
        )

        # Generic output settings: output_
        output_output_dir: str = Field(
            "",
            description="Directory output files will be placed",
        )
        output_composite_output: Optional[bool] = Field(
            None,
            description=(
                "If True, save one set of experiment/reflection files per process, "
                "where each is a concatenated list of all the successful events "
                "examined by that process. If False, output a separate "
                "experiment/reflection file per image (generates a lot of files). "
                "Omitted from the rendered phil (falling back to "
                "dials.stills_process's own default of True) unless explicitly "
                "set."
            ),
        )
        output_logging_dir: str = Field(
            "", description="Directory output log files will be placed"
        )
        output_logging_option: Optional[str] = Field(
            None,
            description=(
                "Logging verbosity option. One of: normal, suppressed, disabled. "
                "'suppressed' silences per-image log output (recommended for MPI jobs). "
                "'normal' writes a log file per MPI rank. 'disabled' turns off logging. "
                "Omitted from the rendered phil (falling back to "
                "dials.stills_process's own default of 'suppressed') unless "
                "explicitly set."
            ),
        )

        # Dispatch settings: dispatch_
        dispatch_index: Optional[bool] = Field(
            None,
            description=(
                "Attempt to index images. find_spots also needs to be True for "
                "this to work. Omitted from the rendered phil (falling back to "
                "dials.stills_process's own default of True) unless explicitly "
                "set."
            ),
        )
        dispatch_refine: Optional[bool] = Field(
            None,
            description=(
                "If True, after indexing, refine the experimental models. "
                "Omitted from the rendered phil (falling back to "
                "dials.stills_process's own default of False) unless "
                "explicitly set."
            ),
        )
        dispatch_integrate: Optional[bool] = Field(
            None,
            description=(
                "Integrate indexed images. Ignored if index=False or "
                "find_spots=False. Omitted from the rendered phil (falling "
                "back to dials.stills_process's own default of True) unless "
                "explicitly set."
            ),
        )
        dispatch_hit_finder_enable: Optional[bool] = Field(
            None,
            description=(
                "Whether to filter out images that do not have at least "
                "minimum_number_of_reflections strong spots. Omitted from the "
                "rendered phil (falling back to dials.stills_process's own "
                "default) unless explicitly set."
            ),
        )
        dispatch_hit_finder_minimum_number_of_reflections: Optional[int] = Field(
            None,
            description=(
                "Minimum number of strong spots for an image to be considered "
                "a hit and processed further. Omitted from the rendered phil "
                "unless explicitly set."
            ),
        )
        dispatch_hit_finder_maximum_number_of_reflections: Optional[int] = Field(
            None,
            description=(
                "Maximum number of strong spots for an image to still be "
                "considered for processing. Omitted from the rendered phil "
                "unless explicitly set."
            ),
        )

        # Parallel processing parameters: mp_
        mp_method: str = Field(
            "mpi",  # *multiprocessing sge lsf pbs mpi
            description="The multiprocessing method to use",
        )

        # Radial average parameters: radial_average_
        radial_average_enable: Optional[bool] = Field(
            None,
            description=(
                "Calculate a radial average of each image, used e.g. to check "
                "detector distance calibration against known powder ring "
                "positions. Omitted from the rendered phil unless explicitly "
                "set."
            ),
        )
        radial_average_two_theta_low: Optional[float] = Field(
            None,
            description=(
                "Low two-theta (degrees) bound for the radial average range. "
                "Omitted from the rendered phil unless explicitly set."
            ),
        )
        radial_average_two_theta_high: Optional[float] = Field(
            None,
            description=(
                "High two-theta (degrees) bound for the radial average range. "
                "Omitted from the rendered phil unless explicitly set."
            ),
        )
        radial_average_verbose: Optional[bool] = Field(
            None,
            description=(
                "Verbose output for the radial average calculation. Omitted "
                "from the rendered phil unless explicitly set."
            ),
        )
        radial_average_output_bins: Optional[bool] = Field(
            None,
            description=(
                "Output the radial average bins. Omitted from the rendered "
                "phil unless explicitly set."
            ),
        )
        radial_average_show_plots: Optional[bool] = Field(
            None,
            description=(
                "Show plots of the radial average. Omitted from the rendered "
                "phil unless explicitly set."
            ),
        )
        radial_average_mask: Optional[str] = Field(
            None,
            description=(
                "Path to a mask file used when calculating the radial "
                "average. Omitted from the rendered phil unless explicitly "
                "set."
            ),
        )

        # Spotfinding parameters: spotfinder_
        spotfinder_lookup_mask: Optional[str] = Field(
            None, description="The path to the mask file."
        )
        spotfinder_threshold_dispersion_gain: Optional[float] = Field(
            None,
            description=(
                "Use a flat gain map for the entire detector to act as a "
                "multiplier for the gain set by the format. Cannot be used in "
                "conjunction with lookup.gain_map parameter."
            ),
        )
        spotfinder_threshold_dispersion_sigma_bkgnd: Optional[float] = Field(
            None,
            description=(
                "The number of standard deviations of the index of dispersion "
                "(variance / mean) in the local area below which the pixel "
                "will be classified as background. Omitted from the rendered "
                "phil (falling back to dials.stills_process's own default of "
                "6) unless explicitly set."
            ),
        )
        spotfinder_threshold_dispersion_sigma_strong: Optional[float] = Field(
            None,
            description=(
                "The number of standard deviations above the mean in the local "
                "area above which the pixel will be classified as strong. "
                "Omitted from the rendered phil (falling back to "
                "dials.stills_process's own default of 3) unless explicitly "
                "set."
            ),
        )
        spotfinder_threshold_dispersion_global_threshold: Optional[float] = Field(
            None,
            description=(
                "The global threshold value. Consider all pixels less than "
                "this value to be part of the background. Omitted from the "
                "rendered phil (falling back to dials.stills_process's own "
                "default of 0) unless explicitly set."
            ),
        )
        spotfinder_threshold_dispersion_kernel_size: Optional[Tuple[int, int]] = Field(
            None,
            description=(
                "The size of the local area around the spot in which to "
                "calculate the mean and variance. The kernel is given as a box "
                "of size (2 * nx + 1, 2 * ny + 1) centred at the pixel. Omitted "
                "from the rendered phil (falling back to dials.stills_process's "
                "own default of (3, 3)) unless explicitly set - note this "
                "differs from the (6, 6) this field used to hardcode "
                "unconditionally."
            ),
        )
        spotfinder_filter_min_spot_size: Optional[int] = Field(
            None,
            description=(
                "The minimum number of contiguous pixels for a spot to be "
                "accepted by the filtering algorithm. Omitted from the "
                "rendered phil (falling back to dials.stills_process's own "
                "default of 'Auto', an adaptive per-image threshold rather "
                "than a fixed count) unless explicitly set."
            ),
        )
        spotfinder_filter_d_min: Optional[float] = Field(
            None,
            description=(
                "The high resolution limit in Angstrom for a pixel to be "
                "accepted by the filtering algorithm."
            ),
        )
        spotfinder_filter_max_spot_size: Optional[int] = Field(
            None,
            description=(
                "The maximum number of contiguous pixels for a spot to be "
                "accepted by the filtering algorithm. Omitted from the "
                "rendered phil unless explicitly set."
            ),
        )
        spotfinder_threshold_algorithm: Optional[str] = Field(
            None,
            description=(
                "Spotfinding threshold algorithm: one of 'dispersion', "
                "'dispersion_extended', or 'radial_profile'. Omitted from the "
                "rendered phil (falling back to dials.stills_process's own "
                "default of dispersion_extended) unless explicitly set."
            ),
        )

        # Indexing parameters: indexing_
        indexing_stills_refine_candidates_with_known_symmetry: Optional[bool] = Field(
            None,
            description=(
                "If False, when choosing the best set of candidate basis "
                "solutions, refine the candidates in the P1 setting. If True, "
                "after indexing in P1, convert the candidates to the known "
                "symmetry and apply the corresponding change of basis to the "
                "indexed reflections. Omitted from the rendered phil (falling "
                "back to dials.stills_process's own default of False) unless "
                "explicitly set."
            ),
        )
        indexing_stills_refine_all_candidates: Optional[bool] = Field(
            None,
            description=(
                "If False, no attempt is made to refine the model from initial "
                "basis vector selection. The indexing solution with the best "
                "RMSD is chosen. Omitted from the rendered phil (falling back "
                "to dials.stills_process's own default of True) unless "
                "explicitly set."
            ),
        )
        indexing_known_symmetry_space_group: Optional[str] = Field(
            None, description="Target space group for indexing."
        )
        indexing_known_symmetry_unit_cell: Optional[str] = Field(
            None, description="Target unit cell for indexing."
        )
        indexing_refinement_protocol_d_min_start: Optional[float] = Field(
            None,
            description=(
                "Resolution (d_min) used for the initial, low-resolution "
                "indexing refinement macrocycle, before subsequent cycles "
                "step down toward the full resolution limit. Omitted from "
                "the rendered phil unless explicitly set."
            ),
        )
        indexing_multiple_lattice_search_max_lattices: Optional[int] = Field(
            None,
            description=(
                "Maximum number of candidate lattices to search for per "
                "still image, enabling multi-lattice indexing. Omitted from "
                "the rendered phil unless explicitly set."
            ),
        )

        # Integration parameters: integration_
        integration_lookup_mask: Optional[str] = Field(
            None, description="The path to the mask file used during integration."
        )
        integration_background_simple_outlier_plane_n_sigma: Optional[float] = Field(
            None,
            description=(
                "The number of standard deviations above the threshold "
                "plane to use in rejecting outliers from background "
                "calculation. Omitted from the rendered phil (falling back "
                "to dials.stills_process's own default of 4.0) unless "
                "explicitly set."
            ),
        )
        integration_summation_detector_gain: Optional[float] = Field(
            None,
            description=(
                "Multiplier for variances after integration of still images. See "
                "Leslie 1999. Omitted from the rendered phil (falling back to "
                "dials.stills_process's own default of 1) unless explicitly "
                "set."
            ),
        )

        # Profiling parameters: profile_
        profile_gaussian_rs_centroid_definition: Optional[str] = Field(
            None,
            description=(
                "The centroid to use as beam divergence (centre of mass or "
                "s1). Omitted from the rendered phil (falling back to "
                "dials.stills_process's own default of 's1') unless "
                "explicitly set."
            ),
        )

        # Refinement options: refinement_
        refinement_reflections_outlier_algorithm: Optional[str] = Field(
            None,
            description=(
                "Outlier rejection algorithm: one of 'null', 'auto', 'mcd', "
                "'tukey', 'sauter_poon'. If auto is selected, the algorithm "
                "is chosen automatically. Omitted from the rendered phil "
                "(falling back to dials.stills_process's own default of "
                "'auto') unless explicitly set."
            ),
        )

        @validator("output_output_dir", always=True)
        def set_output_dir(cls, output: str, values: Dict[str, Any]) -> str:
            if output == "":
                return os.getenv("LUTE_WORK_DIR", ".")
            return output

        @validator("output_logging_dir", always=True)
        def set_output_log_dir(cls, output: str, values: Dict[str, Any]) -> str:
            if output == "":
                return values["output_output_dir"]
            return output

    _set_phil_template_parameters = template_parameter_validator("phil_parameters")

    executable: str = Field(
        "/sdf/group/lcls/ds/tools/cctbx/psana2/conda_base_psana2/bin/mpirun",
        description="MPI executable.",
        flag_type="",
    )
    cctbx_executable: str = Field(
        "/sdf/group/lcls/ds/tools/cctbx/psana2/build/bin/cctbx.xfel.process",
        description=(
            "CCTBX indexing program. cctbx.xfel.process wraps dials.stills_process "
            "unchanged (identical phil scope for dispatch/spotfinder/indexing/"
            "integration/profile/refinement) and adds two LCLS-specific scopes: "
            "radial_average (see PhilParameters.radial_average_*) and optional "
            "MySQL experiment/trial/rungroup database logging, gated entirely on "
            "the top-level 'experiment_tag' phil parameter. LUTE does not expose "
            "experiment_tag as a field, so it is always left at its 'None' "
            "default and no database connection is ever attempted - safe to use "
            "with no DB dependency."
        ),
        flag_type="",
    )
    in_file: str = Field(
        "",
        description=(
            "The location of a data specification for LCLS. "
            "This file will be written for you based on the data_spec parameter. "
            "If not running at LCLS, this can be an input file, or a glob."
        ),
        flag_type="",
    )
    data_spec: Optional[Dict[str, Union[str, float, int]]] = Field(
        None,
        description=(
            "Provide a CCTBX specification for data access. Keys become "
            "'key=value' lines in the generated psana locator (.loc) file, "
            "so any locator parameter the psana2 format reader accepts can "
            "be passed here. Always required: 'experiment', 'run', "
            "'detector_address'. Commonly useful LCLS calibration keys "
            "(all optional, all superseded for detector geometry purposes "
            "by phil_parameters.input_reference_geometry if that is set): "
            "'<detector>.detz_offset' (float, mm) - raw detector-to-sample "
            "distance used by the format reader; irrelevant once "
            "input_reference_geometry overrides the detector model. "
            "'wavelength_offset' (float) - fixed correction applied to the "
            "nominal beamline wavelength. 'spectrum_eV_per_pixel' and "
            "'spectrum_eV_offset' (floats) - linear single-shot photon "
            "energy calibration fit by xfel.fee_calibration from FEE "
            "spectrometer notch scans, applied per-event as "
            "eV = spectrum_eV_offset + pixel_centroid * "
            "spectrum_eV_per_pixel; omitting these falls back to one fixed "
            "wavelength for every shot in the run rather than a true "
            "per-shot value (a precision refinement, not required for "
            "valid indexing)."
        ),
        flag_type="",
    )
    phil_file: str = Field(
        "",
        description="Location of the input settings ('phil') file.",
        flag_type="",
    )
    phil_parameters: Optional[PhilParameters] = Field(
        None,
        description="Optional template parameters to fill in a CCTBX phil file.",
        flag_type="",  # Does nothing since always None by time it's seen by Task
    )
    lute_template_cfg: TemplateConfig = Field(
        TemplateConfig(
            template_name="cctbx_index.phil",
            output_path="",
        ),
        description="Template information for the cctbx_index file.",
    )

    @validator("phil_file", always=True)
    def set_default_phil_path(cls, phil_file: str, values: Dict[str, Any]) -> str:
        if phil_file == "":
            return f"{values['lute_config'].work_dir}/cctbx_index.phil"
        return phil_file

    @validator("lute_template_cfg", always=True)
    def set_phil_template_path(
        cls, lute_template_cfg: TemplateConfig, values: Dict[str, Any]
    ) -> TemplateConfig:
        if lute_template_cfg.output_path == "":
            lute_template_cfg.output_path = values["phil_file"]
        return lute_template_cfg

    @validator("in_file", always=True)
    def set_in_file(cls, in_file: str, values: Dict[str, Any]) -> str:
        if in_file == "":
            exp: str = values["lute_config"].experiment
            run: str = str(values["lute_config"].run)
            work_dir: str = values["lute_config"].work_dir
            return f"{work_dir}/data_{exp}_{run}.loc"
        return in_file

    @validator("data_spec", always=True)
    def write_data_spec_file(
        cls,
        data_spec: Optional[Dict[str, Union[str, float, int]]],
        values: Dict[str, Any],
    ) -> None:
        if data_spec is not None:
            with open(values["in_file"], "w") as f:
                for key, value in data_spec.items():
                    spec_line: str = f"{key}={value}\n"
                    f.write(spec_line)
        return None
