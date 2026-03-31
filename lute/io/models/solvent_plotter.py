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


class PlotSolventParameters(ThirdPartyParameters):
    """Parameters for plotting solvent scattering data.

    This task plots the solvent scattering data from input h5 file, produces output
    png file for elog display.
    """

    class Config(ThirdPartyParameters.Config):
        """Configuration for parameters."""

        long_flags_use_eq: bool = False
        set_result: bool = True
        result_from_params: str = ""

    # ====== BEGIN PENDING SCRIPTS ======
    executable: str = Field("python", description="Python executable.", flag_type="")
    python_script: str = Field(
        description="Path to the solvent scattering plotting script",
        flag_type="",
    )

    input_h5: str = Field(
        "",
        description="Path to input smalldata h5 file that contains 1D Azimuthal Integration data",
        flag_type="--",
        rename_param="data-file",
    )

    event_codes: Union[List[int], str] = Field(
        [],
        description="List of event codes to classify",
        flag_type="--",
        rename_param="event-codes",
    )

    output_dir: str = Field(
        "",
        description="Path to output files, including output h5 file and png files for "
        "intermediate plots",
        flag_type="--",
        rename_param="output-dir",
    )

    output_png: str = Field(
        "",
        description="name of the output png file that contains the summary plot",
        flag_type="--",
        rename_param="image",
    )

    medfilt_window: int = Field(
        13,
        description="Window size for median filter, must be odd",
        flag_type="--",
        rename_param="medfilt-window",
    )

    water_q_params: Union[List[float], str, None] = Field(
        None,
        description="(low_q, high_q, ratio) for filtering based on water peak shapes. A good starting point is (1.0, 1.93, 1.25).",
        flag_type="--",
        rename_param="water-q",
    )

    water_peak_sigma: float = Field(
        None,
        description="Sigma threshold for water peak filtering. A good starting point is 2.0.",
        flag_type="--",
        rename_param="water-peak-sigma",
    )

    laser_on_event_code: int = Field(
        None,
        description="Laser on event code. Default is None. Default is the first event code.",
        flag_type="--",
        rename_param="laser-on-event-code",
    )

    reference_event_code: int = Field(
        None,
        description="Reference event code for difference subtraction. Default is the last event code.",
        flag_type="--",
        rename_param="reference-event-code",
    )

    # ====== END PENDING SCRIPTS ======
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

    @validator("water_q_params")
    def water_q_params_validator(
        cls, water_q_params: Union[str, List[float], None], values: Dict[str, Any]
    ) -> str:
        print(f"water q params: {water_q_params}", flush=True)
        print(f"type(water_q_params): {type(water_q_params)}", flush=True)
        if water_q_params is None:
            return None
        if isinstance(water_q_params, list):
            if len(water_q_params) != 3:
                raise ValueError(
                    "Water q params must be a list of 3 floats: low_q, high_q, ratio"
                )
            for param in water_q_params:
                if not isinstance(param, float):
                    raise ValueError(f"Water q param {param} is not a float")
            return " ".join(str(param) for param in water_q_params)
        else:
            return water_q_params

    @validator("output_dir")
    def validate_output_dir(cls, output_dir: str, values: Dict[str, Any]):
        """Create output directory if it doesn't exist."""
        if output_dir == "":
            exp: str = values["lute_config"].experiment
            run: int = int(values["lute_config"].run)
            hutch: str = exp[:3]
            output_dir = f"/sdf/data/lcls/ds/{hutch}/{exp}/stats/summary/SolventScattering/{run:04d}"
        import os

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir

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

    @validator("output_png")
    def validate_output_png(cls, output_png: str, values: Dict[str, Any]):
        if output_png == "":
            run: int = int(values["lute_config"].run)
            output_png = f"run{run:04d}_averages_by_event_codes.png"
        return output_png

    @root_validator(pre=False)
    def define_result(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # Extract the values of output_dir and out_name
        output_dir: str = values["output_dir"]
        out_name: str = values["output_png"]
        result: str = f"{output_dir}/{out_name}"
        cls.Config.result_from_params = result
        return values
