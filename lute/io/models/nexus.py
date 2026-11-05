"""Models for NeXuS related Tasks.

Classes:
    ConvertSMDToNexusParameters(TaskParameters): Parameters to convert
        smalldata_tools HDF5 file to CCTBX compatible NeXuS file.
"""

__all__ = ["ConvertSMDToNexusParameters"]
__author__ = "Fred Poitevin and Gabriel Dorlhiac"

from typing import Any, Dict, Union

import pydantic

from lute.io.models.base import TaskParameters

pydantic_ver: int = int(pydantic.version.VERSION[0])
if pydantic_ver > 1:
    from pydantic.v1 import Field, validator  # type: ignore
else:
    from pydantic import Field, validator


class ConvertSMDToNexusParameters(TaskParameters):
    """Parameters for running a conversion of smalldata HDF5 to NEXUS HDF5."""

    class Config(TaskParameters.Config):
        """Identical to super-class Config but includes a result."""

        set_result: bool = True
        """Whether the Executor should mark a specified parameter as a result."""

    input: str = Field(
        "",
        description="Path to input smd .h5 file.",
    )
    geom: str = Field(description="Detector geometry file (.expt).")
    output: str = Field(
        "",
        description="Path to output Nexus (.h5) file.",
        is_result=True,
    )
    flip: bool = Field(False, description="Flag to flip.")

    @validator("input", always=True)
    def validate_input_path(cls, input: str, values: Dict[str, Any]) -> str:
        if input == "":
            exp: str = values["lute_config"].experiment
            hutch: str = exp[:3]
            run: Union[str, int] = values["lute_config"].run
            base_path: str = f"/sdf/data/lcls/ds/{hutch}/{exp}/hdf5/cctbx"
            path: str
            path = f"{base_path}/{exp}_Run{int(run):04d}.h5"
            return path
        return input

    @validator("output", always=True)
    def validate_output_path(cls, output: str, values: Dict[str, Any]) -> str:
        if output == "":
            exp: str = values["lute_config"].experiment
            hutch: str = exp[:3]
            run: Union[str, int] = values["lute_config"].run
            base_path: str = f"/sdf/data/lcls/ds/{hutch}/{exp}/hdf5/cctbx"
            path: str
            path = f"{base_path}/{exp}_Run{int(run):04d}_nexus.h5"
            return path
        return output
