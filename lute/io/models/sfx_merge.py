"""Models for merging reflections in serial femtosecond crystallography.

Classes:
    MergePartialatorParameters(ThirdPartyParameters): Perform merging using
        CrystFEL's `partialator`.

    CompareHKLParameters(ThirdPartyParameters): Calculate figures of merit using
        CrystFEL's `compare_hkl`.

    ManipulateHKLParameters(ThirdPartyParameters): Perform transformations on
        lists of reflections using CrystFEL's `get_hkl`.
"""

__all__ = [
    "MergePartialatorParameters",
    "ScaleCCTBXXFELParameters",
    "MergeCCTBXXFELParameters",
    "CompareHKLParameters",
    "ManipulateHKLParameters",
]
__author__ = "Gabriel Dorlhiac"

import os
from typing import Union, List, Optional, Dict, Any

from pydantic import Field, validator, BaseModel

from lute.io.db import read_latest_db_entry
from lute.io.models.base import ThirdPartyParameters, TemplateConfig
from lute.io.models.validators import template_parameter_validator


class MergePartialatorParameters(ThirdPartyParameters):
    """Parameters for CrystFEL's `partialator`.

    There are many parameters, and many combinations. For more information on
    usage, please refer to the CrystFEL documentation, here:
    https://www.desy.de/~twhite/crystfel/manual-partialator.html
    """

    class Config(ThirdPartyParameters.Config):
        long_flags_use_eq: bool = True
        """Whether long command-line arguments are passed like `--long=arg`."""

        set_result: bool = True
        """Whether the Executor should mark a specified parameter as a result."""

    executable: str = Field(
        "/sdf/group/lcls/ds/tools/crystfel/0.10.2/bin/partialator",
        description="CrystFEL's Partialator binary.",
        flag_type="",
    )
    in_file: Optional[str] = Field(
        "", description="Path to input stream.", flag_type="-", rename_param="i"
    )
    out_file: str = Field(
        "",
        description="Path to output file.",
        flag_type="-",
        rename_param="o",
        is_result=True,
    )
    symmetry: str = Field(description="Point group symmetry.", flag_type="--")
    niter: Optional[int] = Field(
        description="Number of cycles of scaling and post-refinement.",
        flag_type="-",
        rename_param="n",
    )
    no_scale: Optional[bool] = Field(
        description="Disable scaling.", flag_type="--", rename_param="no-scale"
    )
    no_Bscale: Optional[bool] = Field(
        description="Disable Debye-Waller part of scaling.",
        flag_type="--",
        rename_param="no-Bscale",
    )
    no_pr: Optional[bool] = Field(
        description="Disable orientation model.", flag_type="--", rename_param="no-pr"
    )
    no_deltacchalf: Optional[bool] = Field(
        description="Disable rejection based on deltaCC1/2.",
        flag_type="--",
        rename_param="no-deltacchalf",
    )
    model: str = Field(
        "unity",
        description="Partiality model. Options: xsphere, unity, offset, ggpm.",
        flag_type="--",
    )
    nthreads: int = Field(
        max(int(os.environ.get("SLURM_NPROCS", len(os.sched_getaffinity(0)))) - 1, 1),
        description="Number of parallel analyses.",
        flag_type="-",
        rename_param="j",
    )
    polarisation: Optional[str] = Field(
        description="Specification of incident polarisation. Refer to CrystFEL docs for more info.",
        flag_type="--",
    )
    no_polarisation: Optional[bool] = Field(
        description="Synonym for --polarisation=none",
        flag_type="--",
        rename_param="no-polarisation",
    )
    max_adu: Optional[float] = Field(
        description="Maximum intensity of reflection to include.",
        flag_type="--",
        rename_param="max-adu",
    )
    min_res: Optional[float] = Field(
        description="Only include crystals diffracting to a minimum resolution.",
        flag_type="--",
        rename_param="min-res",
    )
    min_measurements: int = Field(
        2,
        description="Include a reflection only if it appears a minimum number of times.",
        flag_type="--",
        rename_param="min-measurements",
    )
    push_res: Optional[float] = Field(
        description="Merge reflections up to higher than the apparent resolution limit.",
        flag_type="--",
        rename_param="push-res",
    )
    start_after: int = Field(
        0,
        description="Ignore the first n crystals.",
        flag_type="--",
        rename_param="start-after",
    )
    stop_after: int = Field(
        0,
        description="Stop after processing n crystals. 0 means process all.",
        flag_type="--",
        rename_param="stop-after",
    )
    no_free: Optional[bool] = Field(
        description="Disable cross-validation. Testing ONLY.",
        flag_type="--",
        rename_param="no-free",
    )
    custom_split: Optional[str] = Field(
        description="Read a set of filenames, event and dataset IDs from a filename.",
        flag_type="--",
        rename_param="custom-split",
    )
    max_rel_B: float = Field(
        100,
        description="Reject crystals if |relB| > n sq Angstroms.",
        flag_type="--",
        rename_param="max-rel-B",
    )
    output_every_cycle: bool = Field(
        False,
        description="Write per-crystal params after every refinement cycle.",
        flag_type="--",
        rename_param="output-every-cycle",
    )
    no_logs: bool = Field(
        False,
        description="Do not write logs needed for plots, maps and graphs.",
        flag_type="--",
        rename_param="no-logs",
    )
    set_symmetry: Optional[str] = Field(
        description="Set the apparent symmetry of the crystals to a point group.",
        flag_type="-",
        rename_param="w",
    )
    operator: Optional[str] = Field(
        description="Specify an ambiguity operator. E.g. k,h,-l.", flag_type="--"
    )
    force_bandwidth: Optional[float] = Field(
        description="Set X-ray bandwidth. As percent, e.g. 0.0013 (0.13%).",
        flag_type="--",
        rename_param="force-bandwidth",
    )
    force_radius: Optional[float] = Field(
        description="Set the initial profile radius (nm-1).",
        flag_type="--",
        rename_param="force-radius",
    )
    force_lambda: Optional[float] = Field(
        description="Set the wavelength. In Angstroms.",
        flag_type="--",
        rename_param="force-lambda",
    )
    harvest_file: Optional[str] = Field(
        description="Write parameters to file in JSON format.",
        flag_type="--",
        rename_param="harvest-file",
    )

    @validator("in_file", always=True)
    def validate_in_file(cls, in_file: str, values: Dict[str, Any]) -> str:
        if in_file == "":
            stream_file: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}",
                "ConcatenateStreamFiles",
                "out_file",
            )
            if stream_file:
                return stream_file
        return in_file

    @validator("out_file", always=True)
    def validate_out_file(cls, out_file: str, values: Dict[str, Any]) -> str:
        if out_file == "":
            in_file: str = values["in_file"]
            if in_file:
                tag: str = in_file.split(".")[0]
                return f"{tag}.hkl"
            else:
                return "partialator.hkl"
        return out_file


class MergeCCTBXXFELParameters(ThirdPartyParameters):
    """Parameters for merging with cctbx.xfel."""

    class Config(ThirdPartyParameters.Config):
        set_result: bool = False
        """Whether the Executor should mark a specified parameter as a result."""

    class PhilParameters(BaseModel):
        """Template parameters for CCTBX xfel.merge phil file.

        Covers the full cctbx.xfel.merge parameter space needed for both
        scaling-then-merge and combined scaling+merge workflows. All parameters
        use the underscore-flattened naming convention (dots replaced by
        underscores) matching the phil hierarchy.
        """

        class Config(BaseModel.Config):  # type: ignore
            extra: str = "allow"

        # dispatch: controls which pipeline steps run
        dispatch_step_list: Optional[str] = Field(
            None,
            description=(
                "Override the list of pipeline steps (space-separated). "
                "Default (None) runs the full pipeline: "
                "input balance model_scaling modify filter scale postrefine "
                "statistics_unitcell statistics_beam model_statistics "
                "statistics_resolution group errors_merge statistics_intensity "
                "merge statistics_intensity_cxi. "
                "For merging-only (after separate scaling) use: "
                "input model_scaling statistics_unitcell statistics_beam "
                "model_statistics statistics_resolution group errors_merge "
                "statistics_intensity merge statistics_intensity_cxi"
            ),
        )

        # input settings: input_
        input_path: Union[str, List[str]] = Field(
            "",
            description=(
                "Path(s) to directory containing integrated/scaled data. "
                "Accepts a single path string or a list of paths for "
                "multi-run merging. When empty, auto-resolved from the "
                "ScaleCCTBXXFEL output stored in the LUTE database."
            ),
        )
        input_experiments_suffix: Optional[str] = Field(
            None,
            description=(
                "Suffix used to find experiment files. Omitted from the "
                "rendered phil (falling back to cctbx.xfel.merge's own "
                "default of '_integrated.expt') unless explicitly set."
            ),
        )
        input_reflections_suffix: Optional[str] = Field(
            None,
            description=(
                "Suffix used to find reflection files. Omitted from the "
                "rendered phil (falling back to cctbx.xfel.merge's own "
                "default of '_integrated.refl') unless explicitly set."
            ),
        )
        input_parallel_file_load_method: Optional[str] = Field(
            None,
            description=(
                "How to distribute input files across MPI ranks: 'uniform' "
                "(equal files per rank) or 'node_memory' (memory-aware). "
                "Omitted from the rendered phil (falling back to "
                "cctbx.xfel.merge's own default of 'uniform') unless "
                "explicitly set."
            ),
        )

        # Filtering settings: filter_
        filter_algorithm: Optional[str] = Field(
            None,
            description=(
                "Filtering strategy to apply. Space-separated list of: "
                "n_obs, resolution, unit_cell, energy. "
                "E.g. 'n_obs resolution unit_cell' to apply multiple filters. "
                "Omitted from the rendered phil (falling back to "
                "cctbx.xfel.merge's own default of no filtering at all) "
                "unless explicitly set."
            ),
        )
        filter_unit_cell_algorithm: Optional[str] = Field(
            None,
            description=(
                "Unit cell filtering method: 'range', 'value', or 'cluster'. "
                "'value' uses explicit tolerances around a target cell. "
                "'cluster' uses a Mahalanobis distance covariance model. "
                "Omitted from the rendered phil (falling back to "
                "cctbx.xfel.merge's own default of 'value') unless "
                "explicitly set."
            ),
        )
        # filter.unit_cell.value parameters
        filter_unit_cell_value_target_unit_cell: Optional[str] = Field(
            None,
            description=(
                "Target unit cell for value-based filtering. "
                "E.g. '78 78 39 90 90 90'. Omitted from the rendered phil "
                "(falling back to cctbx.xfel.merge's own default of 'Auto', "
                "derived from the scaling model's PDB/MTZ header) unless "
                "explicitly set."
            ),
        )
        filter_unit_cell_value_relative_length_tolerance: Optional[float] = Field(
            None,
            description=(
                "Fractional tolerance on unit cell lengths for value-based "
                "filtering. E.g. 0.03 = 3% tolerance. Omitted from the "
                "rendered phil (falling back to cctbx.xfel.merge's own "
                "default of 0.1) unless explicitly set."
            ),
        )
        filter_unit_cell_value_absolute_angle_tolerance: Optional[float] = Field(
            None,
            description=(
                "Absolute tolerance on cell angles in degrees for "
                "value-based filtering. Omitted from the rendered phil "
                "(falling back to cctbx.xfel.merge's own default of 2.0) "
                "unless explicitly set."
            ),
        )
        filter_unit_cell_value_target_space_group: Optional[str] = Field(
            None,
            description=(
                "Target space group for value-based unit cell filtering. "
                "Omitted from the rendered phil (falling back to "
                "cctbx.xfel.merge's own default of 'Auto', derived from the "
                "scaling model's PDB/MTZ header) unless explicitly set."
            ),
        )
        # filter.unit_cell.cluster parameters
        filter_unit_cell_cluster_covariance_file: str = Field(
            "",
            description=(
                "Path to unit cell covariance file for cluster filtering. "
                "Must be generated by a prior CCTBX clustering run. "
                "Leave empty to skip cluster filtering."
            ),
        )
        filter_unit_cell_cluster_covariance_component: Optional[int] = Field(
            None,
            description=(
                "Which covariance component to use for cluster filtering. "
                "Omitted from the rendered phil (falling back to "
                "cctbx.xfel.merge's own default of 0) unless explicitly set."
            ),
        )
        filter_unit_cell_cluster_covariance_mahalanobis: Optional[float] = Field(
            None,
            description=(
                "Mahalanobis distance cutoff for unit cell outlier rejection. "
                "Omitted from the rendered phil (falling back to "
                "cctbx.xfel.merge's own default of 4.0) unless explicitly set."
            ),
        )
        filter_outlier_min_corr: Optional[float] = Field(
            None,
            description=(
                "Minimum per-image correlation with merged data. Images "
                "below this threshold are rejected. Omitted from the "
                "rendered phil (falling back to cctbx.xfel.merge's own "
                "default of 0.1) unless explicitly set."
            ),
        )

        # Selection settings: select_
        select_algorithm: Optional[str] = Field(
            None,
            description=(
                "Per-reflection selection method. Space-separated list of: "
                "panel, cspad_sensor, significance_filter, isolation_forest. "
                "Omitted from the rendered phil (falling back to "
                "cctbx.xfel.merge's own default of no selection filtering "
                "at all) unless explicitly set."
            ),
        )
        select_significance_filter_sigma: Optional[float] = Field(
            None,
            description=(
                "Remove high-resolution bins until all accepted bins have "
                "<I/sigma> >= this value. Omitted from the rendered phil "
                "(falling back to cctbx.xfel.merge's own default of 0.5) "
                "unless explicitly set."
            ),
        )

        # Scaling settings: scaling_
        scaling_model: str = Field(
            "",
            description=(
                "Path to a reference PDB or MTZ file for absolute scaling "
                "(mark0 algorithm). Leave empty to use internal KB/Wilson "
                "scaling without an external reference."
            ),
        )
        scaling_algorithm: Optional[str] = Field(
            None,
            description=(
                "Scaling algorithm: 'mark0' = per-image scaling against "
                "a reference (requires scaling_model or internal model). "
                "'mark1' = no scaling / Monte Carlo averaging "
                "(requires scaling_unit_cell and scaling_space_group). "
                "Omitted from the rendered phil (falling back to "
                "cctbx.xfel.merge's own default of 'mark0') unless "
                "explicitly set."
            ),
        )
        scaling_unit_cell: Optional[str] = Field(
            None,
            description=(
                "Unit cell for mark1 scaling (no reference model). "
                "E.g. '78 78 39 90 90 90'. Required when scaling_algorithm=mark1."
            ),
        )
        scaling_space_group: Optional[str] = Field(
            None,
            description=(
                "Space group for mark1 scaling (no reference model). "
                "E.g. 'P43212'. Required when scaling_algorithm=mark1."
            ),
        )
        scaling_resolution_scalar: Optional[float] = Field(
            None,
            description=(
                "Scale merging.d_min by this factor to capture reflections "
                "at the edge of the detector resolution. Omitted from the "
                "rendered phil (falling back to cctbx.xfel.merge's own "
                "default of 0.969) unless explicitly set."
            ),
        )

        # Post-refinement: postrefinement_
        postrefinement_enable: Optional[bool] = Field(
            None,
            description=(
                "Enable per-image post-refinement. Omitted from the "
                "rendered phil (falling back to cctbx.xfel.merge's own "
                "default of True) unless explicitly set."
            ),
        )
        postrefinement_algorithm: Optional[str] = Field(
            None,
            description=(
                "Post-refinement algorithm: 'rs' (reciprocal space), "
                "'rs2', 'rs_hybrid' (use analytical derivatives), or "
                "'eta_deff'. Omitted from the rendered phil (falling back "
                "to cctbx.xfel.merge's own default of 'rs') unless "
                "explicitly set."
            ),
        )
        postrefinement_target_weighting: Optional[str] = Field(
            None,
            description=(
                "Residual weighting in post-refinement: 'unit', 'variance', "
                "'gentle' (|I|/sigma^2, often best), or 'extreme'. Omitted "
                "from the rendered phil (falling back to cctbx.xfel.merge's "
                "own default of 'unit') unless explicitly set."
            ),
        )

        # Merging: merging_
        merging_d_min: float = Field(
            3.0,
            description=(
                "High-resolution cutoff in Angstroms. "
                "Always set this explicitly — cctbx.xfel.merge's own "
                "default is unset (None), which produces empty output. "
                "Unlike other fields here, there is no safe value to fall "
                "back to, so this one stays required."
            ),
        )
        merging_d_max: Optional[float] = Field(
            None,
            description=(
                "Low-resolution cutoff in Angstroms. "
                "Mainly affects CCiso statistics. None = no cutoff."
            ),
        )
        merging_merge_anomalous: Optional[bool] = Field(
            None,
            description=(
                "If True, merge Bijvoet (Friedel) pairs. Omitted from the "
                "rendered phil (falling back to cctbx.xfel.merge's own "
                "default of False) unless explicitly set."
            ),
        )
        merging_set_average_unit_cell: Optional[bool] = Field(
            None,
            description=(
                "Apply the data's average unit cell to all crystals before "
                "merging. Omitted from the rendered phil (falling back to "
                "cctbx.xfel.merge's own default of True) unless explicitly "
                "set."
            ),
        )
        merging_minimum_multiplicity: Optional[int] = Field(
            None,
            description=(
                "Minimum redundancy required to output a merged reflection. "
                "Omitted from the rendered phil (falling back to "
                "cctbx.xfel.merge's own default of 2) unless explicitly set."
            ),
        )
        merging_include_multiplicity_column: Optional[bool] = Field(
            None,
            description=(
                "Write redundancy as a separate column in the output MTZ. "
                "Omitted from the rendered phil (falling back to "
                "cctbx.xfel.merge's own default of False) unless explicitly "
                "set."
            ),
        )
        merging_error_model: Optional[str] = Field(
            None,
            description=(
                "Error model: 'ha14', 'ev11' (Evans 2011), "
                "'mm24' (Mittan-Moreau 2024), or 'errors_from_sample_residuals'. "
                "Omitted from the rendered phil (falling back to "
                "cctbx.xfel.merge's own default of 'ev11') unless "
                "explicitly set."
            ),
        )
        merging_error_ev11_minimizer: Optional[str] = Field(
            None,
            description=(
                "Minimizer for ev11 error model: 'lbfgs' or 'LevMar'. "
                "Omitted from the rendered phil (falling back to "
                "cctbx.xfel.merge's own default of 'lbfgs') unless "
                "explicitly set."
            ),
        )

        # Statistics: statistics_
        statistics_n_bins: Optional[int] = Field(
            None,
            description=(
                "Number of resolution shells for statistics output. "
                "Omitted from the rendered phil (falling back to "
                "cctbx.xfel.merge's own default of 10) unless explicitly "
                "set."
            ),
        )
        statistics_report_ML: Optional[bool] = Field(
            None,
            description=(
                "Report per-frame maximum-likelihood statistics. Omitted "
                "from the rendered phil (falling back to cctbx.xfel.merge's "
                "own default of True) unless explicitly set."
            ),
        )
        statistics_cciso_mtz_file: str = Field(
            "",
            description=(
                "Reference MTZ file for CC-iso / R-iso calculation. "
                "Recommended if a reference structure is available. Leave "
                "empty to skip."
            ),
        )
        statistics_cciso_mtz_column_F: Optional[str] = Field(
            None,
            description=(
                "Column name in the CCiso reference MTZ for structure "
                "factors. Omitted from the rendered phil (falling back to "
                "cctbx.xfel.merge's own default of 'fobs') unless "
                "explicitly set."
            ),
        )

        # Output settings: output_
        output_prefix: str = Field(
            "",
            description="Prefix for all output file names (e.g. 'myexp_r0042').",
        )
        output_output_dir: str = Field(
            "",
            description="Directory for output MTZ and log files.",
        )
        output_tmp_dir: str = Field(
            "",
            description="Temporary file directory. Defaults to output_output_dir.",
        )
        output_do_timing: Optional[bool] = Field(
            None,
            description=(
                "Log elapsed time for each pipeline step. Omitted from the "
                "rendered phil (falling back to cctbx.xfel.merge's own "
                "default of False) unless explicitly set."
            ),
        )
        output_log_level: Optional[int] = Field(
            None,
            description=(
                "Log verbosity: 0 = log everything; higher = less logging. "
                "Omitted from the rendered phil (falling back to "
                "cctbx.xfel.merge's own default of 1) unless explicitly set."
            ),
        )
        output_save_experiments_and_reflections: Optional[bool] = Field(
            None,
            description=(
                "Save the filtered/selected experiment and reflection "
                "files alongside MTZ output. Only needed if a subsequent "
                "merging-only run will read this Merge task's own output "
                "(unusual - Merge is normally the last stage). Omitted from "
                "the rendered phil (falling back to cctbx.xfel.merge's own "
                "default of False) unless explicitly set."
            ),
        )

        # Multiprocessing: mp_
        mp_method: Optional[str] = Field(
            None,
            description=(
                "Multiprocessing method. Omitted from the rendered phil "
                "(falling back to cctbx.xfel.merge's own default of 'mpi') "
                "unless explicitly set."
            ),
        )
        mp_psana2_mode: Optional[bool] = Field(
            None,
            description=(
                "Set True when using the integrate worker with psana2/XTC2 "
                "data. Omitted from the rendered phil (falling back to "
                "cctbx.xfel.merge's own default of False) unless explicitly "
                "set."
            ),
        )

        # Parallel (MPI memory): parallel_
        parallel_a2a: Optional[int] = Field(
            None,
            description=(
                "MPI all-to-all communication stride. Set to number of "
                "cores per node (e.g. 64) to reduce memory pressure on "
                "large jobs. Omitted from the rendered phil (falling back "
                "to cctbx.xfel.merge's own default of 1) unless explicitly "
                "set."
            ),
        )

        @validator("input_path", always=True, pre=True)
        def normalize_and_resolve_input_path(
            cls, v: Union[str, List[str]]
        ) -> List[str]:
            """Normalize input_path to a list and auto-resolve from DB if empty."""
            # Normalize to list first
            if isinstance(v, str):
                if v == "":
                    # Try to auto-resolve from ScaleCCTBXXFEL output in LUTE DB
                    work_dir: str = os.getenv("LUTE_WORK_DIR", "")
                    if work_dir:
                        scaled_dir: Optional[str] = read_latest_db_entry(
                            work_dir, "ScaleCCTBXXFEL", "result.payload"
                        )
                        if scaled_dir:
                            return [scaled_dir]
                    return []
                return [v]
            # Already a list — filter out empty strings
            return [p for p in v if p]

    _set_phil_template_parameters = template_parameter_validator("phil_parameters")

    executable: str = Field(
        "/sdf/group/lcls/ds/tools/cctbx/psana2/conda_base_psana2/bin/mpirun",
        description="MPI executable.",
        flag_type="",
    )
    cctbx_executable: str = Field(
        "/sdf/group/lcls/ds/tools/cctbx/psana2/build/bin/cctbx.xfel.merge",
        description="CCTBX merge program.",
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
            template_name="cctbx_merge.phil",
            output_path="",
        ),
        description="Template information for the cctbx_merge file.",
    )

    @validator("phil_file", always=True)
    def set_default_phil_path(cls, phil_file: str, values: Dict[str, Any]) -> str:
        if phil_file == "":
            return f"{values['lute_config'].work_dir}/cctbx_merge.phil"
        return phil_file

    @validator("lute_template_cfg", always=True)
    def set_phil_template_path(
        cls, lute_template_cfg: TemplateConfig, values: Dict[str, Any]
    ) -> TemplateConfig:
        if lute_template_cfg.output_path == "":
            lute_template_cfg.output_path = values["phil_file"]
        return lute_template_cfg


class ScaleCCTBXXFELParameters(ThirdPartyParameters):
    """Parameters for scaling with cctbx.xfel (scaling-only pipeline).

    This task runs ``cctbx.xfel.merge`` with a scaling-only ``dispatch.step_list``
    to produce scaled ``.expt``/``.refl`` files without performing the final
    statistical merge.  The output is stored in the LUTE database so that a
    subsequent :class:`MergeCCTBXXFELParameters` task can auto-resolve
    ``input.path`` from it.

    The default ``dispatch_step_list`` covers:
    ``input balance model_scaling modify filter scale postrefine
    statistics_unitcell statistics_beam model_statistics statistics_resolution``
    """

    class Config(ThirdPartyParameters.Config):
        set_result: bool = True
        """Store the output directory in the LUTE DB for downstream auto-resolution."""

    class ScalePhilParameters(BaseModel):
        """Template parameters for the cctbx_scale.phil file."""

        class Config(BaseModel.Config):  # type: ignore
            extra: str = "allow"

        # dispatch: controls which pipeline steps run
        dispatch_step_list: str = Field(
            (
                "input balance model_scaling modify filter scale postrefine "
                "statistics_unitcell statistics_beam model_statistics "
                "statistics_resolution"
            ),
            description=(
                "Pipeline steps for the scaling-only workflow. "
                "Override to add or remove steps."
            ),
        )

        # input settings: input_
        input_path: Union[str, List[str]] = Field(
            "",
            description=(
                "Path(s) to directory containing integrated data "
                "(*_integrated.expt / *_integrated.refl). "
                "Accepts a single path string or a list for multi-run scaling."
            ),
        )
        input_experiments_suffix: Optional[str] = Field(
            None,
            description=(
                "Suffix used to find experiment files. Omitted from the "
                "rendered phil (falling back to cctbx.xfel.merge's own "
                "default of '_integrated.expt') unless explicitly set."
            ),
        )
        input_reflections_suffix: Optional[str] = Field(
            None,
            description=(
                "Suffix used to find reflection files. Omitted from the "
                "rendered phil (falling back to cctbx.xfel.merge's own "
                "default of '_integrated.refl') unless explicitly set."
            ),
        )
        input_parallel_file_load_method: Optional[str] = Field(
            None,
            description=(
                "How to distribute input files across MPI ranks: 'uniform' "
                "(equal files per rank) or 'node_memory' (memory-aware). "
                "Omitted from the rendered phil (falling back to "
                "cctbx.xfel.merge's own default of 'uniform') unless "
                "explicitly set."
            ),
        )

        # Filtering settings: filter_
        filter_algorithm: Optional[str] = Field(
            None,
            description=(
                "Filtering strategy. Space-separated list of: "
                "n_obs, resolution, unit_cell, energy. Omitted from the "
                "rendered phil (falling back to cctbx.xfel.merge's own "
                "default of no filtering at all) unless explicitly set."
            ),
        )
        filter_unit_cell_algorithm: Optional[str] = Field(
            None,
            description=(
                "Unit cell filtering method: 'range', 'value', or 'cluster'. "
                "'value' uses explicit tolerances around a target cell. "
                "Omitted from the rendered phil (falling back to "
                "cctbx.xfel.merge's own default of 'value') unless "
                "explicitly set."
            ),
        )
        filter_unit_cell_value_target_unit_cell: Optional[str] = Field(
            None,
            description=(
                "Target unit cell for value-based filtering. E.g. "
                "'78 78 39 90 90 90'. Omitted from the rendered phil "
                "(falling back to cctbx.xfel.merge's own default of "
                "'Auto', derived from the scaling model's PDB/MTZ header) "
                "unless explicitly set."
            ),
        )
        filter_unit_cell_value_relative_length_tolerance: Optional[float] = Field(
            None,
            description=(
                "Fractional tolerance on unit cell lengths (e.g. 0.03 = "
                "3%). Omitted from the rendered phil (falling back to "
                "cctbx.xfel.merge's own default of 0.1) unless explicitly "
                "set."
            ),
        )
        filter_unit_cell_value_absolute_angle_tolerance: Optional[float] = Field(
            None,
            description=(
                "Absolute tolerance on cell angles in degrees. Omitted "
                "from the rendered phil (falling back to cctbx.xfel.merge's "
                "own default of 2.0) unless explicitly set."
            ),
        )
        filter_unit_cell_value_target_space_group: Optional[str] = Field(
            None,
            description=(
                "Target space group for value-based unit cell filtering. "
                "Omitted from the rendered phil (falling back to "
                "cctbx.xfel.merge's own default of 'Auto', derived from "
                "the scaling model's PDB/MTZ header) unless explicitly set."
            ),
        )
        filter_unit_cell_cluster_covariance_file: str = Field(
            "",
            description=(
                "Path to unit cell covariance file for cluster filtering. "
                "Leave empty to skip cluster filtering."
            ),
        )
        filter_unit_cell_cluster_covariance_component: Optional[int] = Field(
            None,
            description=(
                "Which covariance component to use. Omitted from the "
                "rendered phil (falling back to cctbx.xfel.merge's own "
                "default of 0) unless explicitly set."
            ),
        )
        filter_unit_cell_cluster_covariance_mahalanobis: Optional[float] = Field(
            None,
            description=(
                "Mahalanobis distance cutoff for unit cell outlier "
                "rejection. Omitted from the rendered phil (falling back "
                "to cctbx.xfel.merge's own default of 4.0) unless "
                "explicitly set."
            ),
        )
        filter_outlier_min_corr: Optional[float] = Field(
            None,
            description=(
                "Minimum per-image correlation with merged data. Omitted "
                "from the rendered phil (falling back to cctbx.xfel.merge's "
                "own default of 0.1) unless explicitly set."
            ),
        )

        # Selection settings: select_
        select_algorithm: Optional[str] = Field(
            None,
            description=(
                "Per-reflection selection. Space-separated list of: "
                "panel, cspad_sensor, significance_filter, isolation_forest. "
                "Omitted from the rendered phil (falling back to "
                "cctbx.xfel.merge's own default of no selection filtering "
                "at all) unless explicitly set."
            ),
        )
        select_significance_filter_sigma: Optional[float] = Field(
            None,
            description=(
                "Minimum <I/sigma> for a reflection to be included. "
                "Omitted from the rendered phil (falling back to "
                "cctbx.xfel.merge's own default of 0.5) unless explicitly "
                "set."
            ),
        )

        # Scaling settings: scaling_
        scaling_model: str = Field(
            "",
            description=(
                "Path to a reference PDB or MTZ file for mark0 scaling. "
                "Leave empty for internal KB/Wilson scaling."
            ),
        )
        scaling_algorithm: Optional[str] = Field(
            None,
            description=(
                "'mark0' = per-image scaling against reference. "
                "'mark1' = no scaling / Monte Carlo averaging "
                "(requires scaling_unit_cell and scaling_space_group). "
                "Omitted from the rendered phil (falling back to "
                "cctbx.xfel.merge's own default of 'mark0') unless "
                "explicitly set."
            ),
        )
        scaling_unit_cell: Optional[str] = Field(
            None,
            description=(
                "Unit cell for mark1 scaling. "
                "E.g. '78 78 39 90 90 90'. Required when algorithm=mark1."
            ),
        )
        scaling_space_group: Optional[str] = Field(
            None,
            description=(
                "Space group for mark1 scaling. "
                "E.g. 'P43212'. Required when algorithm=mark1."
            ),
        )
        scaling_resolution_scalar: Optional[float] = Field(
            None,
            description=(
                "Scale merging.d_min by this factor to extend resolution "
                "reach. Omitted from the rendered phil (falling back to "
                "cctbx.xfel.merge's own default of 0.969) unless explicitly "
                "set."
            ),
        )

        # Merging resolution: required by xfel.merge even in scaling-only mode
        merging_d_min: float = Field(
            3.0,
            description=(
                "High-resolution cutoff in Angstroms. Required even in "
                "scaling-only mode — cctbx.xfel.merge's own default is "
                "unset (None), which produces empty output. Unlike other "
                "fields here, there is no safe value to fall back to, so "
                "this one stays required."
            ),
        )
        merging_merge_anomalous: Optional[bool] = Field(
            None,
            description=(
                "If True, merge Bijvoet pairs. Omitted from the rendered "
                "phil (falling back to cctbx.xfel.merge's own default of "
                "False) unless explicitly set."
            ),
        )

        # Output settings: output_
        output_prefix: str = Field(
            "",
            description="Prefix for output file names (e.g. 'scaling_r0036').",
        )
        output_output_dir: str = Field(
            "",
            description=(
                "Directory for scaled output files. "
                "This path is stored in the LUTE DB so CCTBXMerger can "
                "auto-resolve input_path from it."
            ),
        )
        output_save_experiments_and_reflections: bool = Field(
            True,
            description=(
                "Save scaled experiment and reflection files. Must stay "
                "True — cctbx.xfel.merge's own default is False, which "
                "would produce no output for CCTBXMerger to read. Unlike "
                "other fields here, there is no safe value to fall back "
                "to, so this one stays required."
            ),
        )
        output_do_timing: Optional[bool] = Field(
            None,
            description=(
                "Log elapsed time per step. Omitted from the rendered "
                "phil (falling back to cctbx.xfel.merge's own default of "
                "False) unless explicitly set."
            ),
        )
        output_log_level: Optional[int] = Field(
            None,
            description=(
                "Log verbosity (0=verbose). Omitted from the rendered "
                "phil (falling back to cctbx.xfel.merge's own default of "
                "1) unless explicitly set."
            ),
        )

        # Statistics
        statistics_n_bins: Optional[int] = Field(
            None,
            description=(
                "Number of resolution shells for statistics. Omitted from "
                "the rendered phil (falling back to cctbx.xfel.merge's own "
                "default of 10) unless explicitly set."
            ),
        )

        # Multiprocessing
        mp_method: Optional[str] = Field(
            None,
            description=(
                "Multiprocessing method. Omitted from the rendered phil "
                "(falling back to cctbx.xfel.merge's own default of "
                "'mpi') unless explicitly set."
            ),
        )
        mp_psana2_mode: Optional[bool] = Field(
            None,
            description=(
                "Set True when using the integrate worker with psana2/XTC2 "
                "data. Omitted from the rendered phil (falling back to "
                "cctbx.xfel.merge's own default of False) unless explicitly "
                "set."
            ),
        )

        @validator("input_path", always=True, pre=True)
        def normalize_input_path(cls, v: Union[str, List[str]]) -> List[str]:
            """Normalize input_path to a list, filtering empty strings."""
            if isinstance(v, str):
                return [v] if v else []
            return [p for p in v if p]

        @validator("output_output_dir", always=True)
        def set_output_dir(cls, output: str, values: Dict[str, Any]) -> str:
            if output == "":
                return os.getenv("LUTE_WORK_DIR", ".")
            return output

    _set_scale_phil_template_parameters = template_parameter_validator(
        "phil_parameters"
    )

    executable: str = Field(
        "/sdf/group/lcls/ds/tools/cctbx/psana2/conda_base_psana2/bin/mpirun",
        description="MPI executable.",
        flag_type="",
    )
    cctbx_executable: str = Field(
        "/sdf/group/lcls/ds/tools/cctbx/psana2/build/bin/cctbx.xfel.merge",
        description="CCTBX merge/scale program.",
        flag_type="",
    )
    phil_file: str = Field(
        "",
        description="Location of the input settings ('phil') file.",
        flag_type="",
    )
    phil_parameters: Optional[ScalePhilParameters] = Field(
        None,
        description="Optional template parameters to fill in a CCTBX scale phil file.",
        flag_type="",
    )
    lute_template_cfg: TemplateConfig = Field(
        TemplateConfig(
            template_name="cctbx_scale.phil",
            output_path="",
        ),
        description="Template information for the cctbx_scale file.",
    )
    result_output_dir: str = Field(
        "",
        description=(
            "Output directory for scaled files. Populated automatically from "
            "phil_parameters.output_output_dir and stored in the LUTE database "
            "so that a downstream CCTBXMerger task can auto-resolve input_path."
        ),
        flag_type="",
        is_result=True,
    )

    @validator("phil_file", always=True)
    def set_default_phil_path(cls, phil_file: str, values: Dict[str, Any]) -> str:
        if phil_file == "":
            return f"{values['lute_config'].work_dir}/cctbx_scale.phil"
        return phil_file

    @validator("lute_template_cfg", always=True)
    def set_phil_template_path(
        cls, lute_template_cfg: TemplateConfig, values: Dict[str, Any]
    ) -> TemplateConfig:
        if lute_template_cfg.output_path == "":
            lute_template_cfg.output_path = values["phil_file"]
        return lute_template_cfg

    @validator("result_output_dir", always=True)
    def sync_result_from_output_dir(cls, v: str, values: Dict[str, Any]) -> str:
        """Copy output_output_dir (extracted from ScalePhilParameters) to result."""
        if v == "" and "output_output_dir" in values:
            raw = values["output_output_dir"]
            # At this point it may be a raw str (before extra_fields_to_thirdparty)
            # or a TemplateParameters (if ordering shifts) — handle both
            if hasattr(raw, "params"):
                return str(raw.params)
            if isinstance(raw, str) and raw:
                return raw
        return v


class CompareHKLParameters(ThirdPartyParameters):
    """Parameters for CrystFEL's `compare_hkl` for calculating figures of merit.

    There are many parameters, and many combinations. For more information on
    usage, please refer to the CrystFEL documentation, here:
    https://www.desy.de/~twhite/crystfel/manual-partialator.html
    """

    class Config(ThirdPartyParameters.Config):
        long_flags_use_eq: bool = True
        """Whether long command-line arguments are passed like `--long=arg`."""

        set_result: bool = True
        """Whether the Executor should mark a specified parameter as a result."""

    executable: str = Field(
        "/sdf/group/lcls/ds/tools/crystfel/0.10.2/bin/compare_hkl",
        description="CrystFEL's reflection comparison binary.",
        flag_type="",
    )
    in_files: Optional[str] = Field(
        "",
        description="Path to input HKLs. Space-separated list of 2. Use output of partialator e.g.",
        flag_type="",
    )
    ## Need mechanism to set is_result=True ...
    symmetry: str = Field("", description="Point group symmetry.", flag_type="--")
    cell_file: str = Field(
        "",
        description="Path to a file containing unit cell information (PDB or CrystFEL format).",
        flag_type="-",
        rename_param="p",
    )
    fom: str = Field(
        "Rsplit", description="Specify figure of merit to calculate.", flag_type="--"
    )
    nshells: int = Field(10, description="Use n resolution shells.", flag_type="--")
    # NEED A NEW CASE FOR THIS -> Boolean flag, no arg, one hyphen...
    # fix_unity: bool = Field(
    #    False,
    #    description="Fix scale factors to unity.",
    #    flag_type="-",
    #    rename_param="u",
    # )
    shell_file: str = Field(
        "",
        description="Write the statistics in resolution shells to a file.",
        flag_type="--",
        rename_param="shell-file",
        is_result=True,
    )
    ignore_negs: bool = Field(
        False,
        description="Ignore reflections with negative reflections.",
        flag_type="--",
        rename_param="ignore-negs",
    )
    zero_negs: bool = Field(
        False,
        description="Set negative intensities to 0.",
        flag_type="--",
        rename_param="zero-negs",
    )
    sigma_cutoff: Optional[Union[float, int, str]] = Field(
        # "-infinity",
        description="Discard reflections with I/sigma(I) < n. -infinity means no cutoff.",
        flag_type="--",
        rename_param="sigma-cutoff",
    )
    rmin: Optional[float] = Field(
        description="Low resolution cutoff of 1/d (m-1). Use this or --lowres NOT both.",
        flag_type="--",
    )
    lowres: Optional[float] = Field(
        descirption="Low resolution cutoff in Angstroms. Use this or --rmin NOT both.",
        flag_type="--",
    )
    rmax: Optional[float] = Field(
        description="High resolution cutoff in 1/d (m-1). Use this or --highres NOT both.",
        flag_type="--",
    )
    highres: Optional[float] = Field(
        description="High resolution cutoff in Angstroms. Use this or --rmax NOT both.",
        flag_type="--",
    )

    @validator("in_files", always=True)
    def validate_in_files(cls, in_files: str, values: Dict[str, Any]) -> str:
        if in_files == "":
            partialator_file: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}", "MergePartialator", "out_file"
            )
            if partialator_file:
                hkls: str = f"{partialator_file}1 {partialator_file}2"
                return hkls
        return in_files

    @validator("cell_file", always=True)
    def validate_cell_file(cls, cell_file: str, values: Dict[str, Any]) -> str:
        if cell_file == "":
            idx_cell_file: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}",
                "IndexCrystFEL",
                "cell_file",
                valid_only=False,
            )
            if idx_cell_file:
                return idx_cell_file
        return cell_file

    @validator("symmetry", always=True)
    def validate_symmetry(cls, symmetry: str, values: Dict[str, Any]) -> str:
        if symmetry == "":
            partialator_sym: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}", "MergePartialator", "symmetry"
            )
            if partialator_sym:
                return partialator_sym
        return symmetry

    @validator("shell_file", always=True)
    def validate_shell_file(cls, shell_file: str, values: Dict[str, Any]) -> str:
        if shell_file == "":
            partialator_file: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}", "MergePartialator", "out_file"
            )
            if partialator_file:
                shells_out: str = partialator_file.split(".")[0]
                shells_out = f"{shells_out}_{values['fom']}_n{values['nshells']}.dat"
                return shells_out
        return shell_file


class ManipulateHKLParameters(ThirdPartyParameters):
    """Parameters for CrystFEL's `get_hkl` for manipulating lists of reflections.

    This Task is predominantly used internally to convert `hkl` to `mtz` files.
    Note that performing multiple manipulations is undefined behaviour. Run
    the Task with multiple configurations in explicit separate steps. For more
    information on usage, please refer to the CrystFEL documentation, here:
    https://www.desy.de/~twhite/crystfel/manual-partialator.html
    """

    class Config(ThirdPartyParameters.Config):
        long_flags_use_eq: bool = True
        """Whether long command-line arguments are passed like `--long=arg`."""

        set_result: bool = True
        """Whether the Executor should mark a specified parameter as a result."""

    executable: str = Field(
        "/sdf/group/lcls/ds/tools/crystfel/0.10.2/bin/get_hkl",
        description="CrystFEL's reflection manipulation binary.",
        flag_type="",
    )
    in_file: str = Field(
        "",
        description="Path to input HKL file.",
        flag_type="-",
        rename_param="i",
    )
    out_file: str = Field(
        "",
        description="Path to output file.",
        flag_type="-",
        rename_param="o",
        is_result=True,
    )
    cell_file: str = Field(
        "",
        description="Path to a file containing unit cell information (PDB or CrystFEL format).",
        flag_type="-",
        rename_param="p",
    )
    output_format: str = Field(
        "mtz",
        description="Output format. One of mtz, mtz-bij, or xds. Otherwise CrystFEL format.",
        flag_type="--",
        rename_param="output-format",
    )
    expand: Optional[str] = Field(
        description="Reflections will be expanded to fill asymmetric unit of specified point group.",
        flag_type="--",
    )
    # Reducing reflections to higher symmetry
    twin: Optional[str] = Field(
        description="Reflections equivalent to specified point group will have intensities summed.",
        flag_type="--",
    )
    no_need_all_parts: Optional[bool] = Field(
        description="Use with --twin to allow reflections missing a 'twin mate' to be written out.",
        flag_type="--",
        rename_param="no-need-all-parts",
    )
    # Noise - Add to data
    noise: Optional[bool] = Field(
        description="Generate 10% uniform noise.", flag_type="--"
    )
    poisson: Optional[bool] = Field(
        description="Generate Poisson noise. Intensities assumed to be A.U.",
        flag_type="--",
    )
    adu_per_photon: Optional[int] = Field(
        description="Use with --poisson to convert A.U. to photons.",
        flag_type="--",
        rename_param="adu-per-photon",
    )
    # Remove duplicate reflections
    trim_centrics: Optional[bool] = Field(
        description="Duplicated reflections (according to symmetry) are removed.",
        flag_type="--",
    )
    # Restrict to template file
    template: Optional[str] = Field(
        description="Only reflections which also appear in specified file are written out.",
        flag_type="--",
    )
    # Multiplicity
    multiplicity: Optional[bool] = Field(
        description="Reflections are multiplied by their symmetric multiplicites.",
        flag_type="--",
    )
    # Resolution cutoffs
    cutoff_angstroms: Optional[Union[str, int, float]] = Field(
        description=(
            "Either n, or n1,n2,n3. For n, reflections < n are removed. "
            "For n1,n2,n3 anisotropic trunction performed at separate resolution "
            "limits for a*, b*, c*."
        ),
        flag_type="--",
        rename_param="cutoff-angstroms",
    )
    lowres: Optional[float] = Field(
        description="Remove reflections with d > n", flag_type="--"
    )
    highres: Optional[float] = Field(
        description="Synonym for first form of --cutoff-angstroms"
    )
    reindex: Optional[str] = Field(
        description="Reindex according to specified operator. E.g. k,h,-l.",
        flag_type="--",
    )
    # Override input symmetry
    symmetry: Optional[str] = Field(
        description="Point group symmetry to use to override. Almost always OMIT this option.",
        flag_type="--",
    )

    @validator("in_file", always=True)
    def validate_in_file(cls, in_file: str, values: Dict[str, Any]) -> str:
        if in_file == "":
            partialator_file: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}", "MergePartialator", "out_file"
            )
            if partialator_file:
                return partialator_file
        return in_file

    @validator("out_file", always=True)
    def validate_out_file(cls, out_file: str, values: Dict[str, Any]) -> str:
        if out_file == "":
            partialator_file: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}", "MergePartialator", "out_file"
            )
            if partialator_file:
                mtz_out: str = partialator_file.split(".")[0]
                mtz_out = f"{mtz_out}.mtz"
                return mtz_out
        return out_file

    @validator("cell_file", always=True)
    def validate_cell_file(cls, cell_file: str, values: Dict[str, Any]) -> str:
        if cell_file == "":
            idx_cell_file: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}",
                "IndexCrystFEL",
                "cell_file",
                valid_only=False,
            )
            if idx_cell_file:
                return idx_cell_file
        return cell_file
