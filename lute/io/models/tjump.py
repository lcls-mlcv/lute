"""Parameters for T-Jumpt scattering analysis task.

This module contains the parameter model for the T-Jumpt scattering analysis
that processes smalldata h5 files and produces new h5 files with a matplotlib
summary figure for elog display.
"""

from typing import Optional, Dict, Any, List, Union
from pydantic import Field, validator, root_validator

from lute.io.models.base import ThirdPartyParameters
from lute.io.db import read_latest_db_entry
import os


class TJumpParameters(ThirdPartyParameters):
    """Parameters for TJump analysis task.

    This task processes smalldata h5 files to analyze solvent scattering in
    temperature jump experiments, produces output h5 files along with summary
    plots for elog display.
    """

    class Config(ThirdPartyParameters.Config):
        """Configuration for parameters."""

        long_flags_use_eq: bool = False
        set_result: bool = True
        result_from_params: str = ""

    # ====== BEGIN PENDING ALEX SCRIPTS ======
    # executable: str = Field(
    #     "/path/to/tjump_script.py",
    #     description="Path to the TJump processing script",
    # )

    executable: str = Field("python", description="Python executable.", flag_type="")
    python_script: str = Field(
        description="Path to the TJump processing script",
        flag_type="",
    )

    input_h5: str = Field(
        "",
        description="Path to input smalldata h5 file that contains 1D Azimuthal Integration data",
        flag_type="--",
        rename_param="input",
    )

    event_codes: Union[List[int], str] = Field(
        [],
        description="List of event codes to use, assumes the first code is laser_on, followed by laser_off1, laser_off2, etc.",
        flag_type="--",
        rename_param="event_codes",
    )

    exp: str = Field(
        "",
        description="LCLS experiment identifier",
        flag_type="--",
        rename_param="exp",
    )
    run: int = Field(
        -1,
        description="LCLS run number",
        flag_type="--",
        rename_param="run",
    )

    # Output parameters
    output_dir: str = Field(
        "",
        description="Path to output files, including output h5 file and png files for "
        "intermediate plots",
        flag_type="--",
        rename_param="output",
    )

    output_h5: str = Field(
        "",
        description="name of the output h5 file that contains scaled 1D data, time stamp, "
        "evr code, and all individual filtering masks where columns are "
        "boolean and listed in the ordering of applied filters",
        flag_type="--",
        rename_param="output_h5",
    )
    peakfit: str = Field(
        "",
        description="Peak fitting method: 'simple' or 'spline' or 'two_peak_fit' (default=simple). "
        "Not applicable to sd2qwp1.py script.",
        flag_type="--",
        rename_param="peakfit",
    )
    # # Other potential parameters
    # zscore_threshold: Optional[float] = Field(
    #     2,
    #     description="Z-score threshold for filtering",
    #     flag_type="--",
    # )

    # qmin: Optional[float] = Field(
    #     0.3,
    #     description="Minimum q-range for analysis (optional)",
    #     flag_type="--",
    # )

    # qmax: Optional[float] = Field(
    #     3.2,
    #     description="Maximum q-range for analysis (optional)",
    #     flag_type="--",
    # )

    # ====== END PENDING ALEX SCRIPTS ======
    @validator("exp")
    def validate_exp(cls, exp: str, values: Dict[str, Any]):
        """Validate that the experiment identifier is a valid LCLS experiment identifier."""
        if exp == "":
            exp = values["lute_config"].experiment
        return exp

    @validator("run")
    def validate_run(cls, run: int, values: Dict[str, Any]):
        """Validate that the run number is a valid LCLS run number."""
        if run == -1:
            run = int(values["lute_config"].run)
        return run

    @validator("output_dir")
    def validate_output_dir(cls, output_dir: str, values: Dict[str, Any]):
        """Create output directory if it doesn't exist."""
        if output_dir == "":
            exp: str = values["lute_config"].experiment
            run: int = int(values["lute_config"].run)
            hutch: str = exp[:3]
            output_dir = (
                f"/sdf/data/lcls/ds/{hutch}/{exp}/stats/summary/TJump/{run:04d}"
            )
        import os

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir

    # ====== BEGIN PENDING ALEX SCRIPTS ======
    @validator("event_codes")
    def event_codes_validator(
        cls, event_codes: Union[str, List[int]], values: Dict[str, Any]
    ) -> str:
        print(f"event_codes: {event_codes}", flush=True)
        print(f"type(event_codes): {type(event_codes)}", flush=True)
        if isinstance(event_codes, list):
            for code in event_codes:
                if not isinstance(code, int):
                    raise ValueError(f"Event code {code} is not an integer")
                if code < 0:
                    raise ValueError(f"Event code {code} is negative")
                if code > 287:
                    raise ValueError(f"Event code {code} is greater than 287")
            return " ".join(str(code) for code in event_codes)
        else:
            return event_codes

    @validator("input_h5")
    def validate_smd_path_for_run(cls, smd_path: str, values: Dict[str, Any]) -> str:
        if smd_path == "":
            run: int = int(values["lute_config"].run)
            # Try from database first
            hdf5_path: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}",
                "SubmitSMD",
                "result.payload",
                for_run=run,
            )
            if hdf5_path is not None:
                return hdf5_path
            else:
                exp: str = values["lute_config"].experiment
                hutch: str = exp[:3]
                hdf5_path = f"/sdf/data/lcls/ds/{hutch}/{exp}/hdf5/smalldata/{exp}_Run{run:04d}.h5"
                if os.path.exists(hdf5_path):
                    return hdf5_path
                raise ValueError("No path provided for hdf5 and cannot auto-determine!")
        return smd_path

    @validator("output_h5")
    def validate_output_h5(cls, output_h5: str, values: Dict[str, Any]):
        # if not supplied, create a default name based on the input h5 file
        if output_h5 == "":
            run: int = int(values["lute_config"].run)
            output_h5 = f"run{run:04d}_sd2qwp1.hdf5"
        return output_h5

    @validator("peakfit")
    def validate_peakfit(cls, peakfit: str, values: Dict[str, Any]):
        """Validate that the peak fitting method is valid."""
        if not values["python_script"].endswith("sd2qwp1.py"):
            if peakfit == "":
                peakfit = "simple"
            elif peakfit not in ["simple", "spline", "two_peak_fit"]:
                raise ValueError(f"Invalid peak fitting method: {peakfit}")
        return peakfit

    # ====== END PENDING ALEX SCRIPTS ======

    @root_validator(pre=False)
    def define_result(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # Extract the values of output_dir and out_name
        output_dir: str = values["output_dir"]
        out_name: str = values["output_h5"]
        result: str = f"{output_dir}/{out_name}"
        cls.Config.result_from_params = result
        return values
