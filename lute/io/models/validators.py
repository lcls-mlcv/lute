"""Generic/reusable validators for parameter models.

Functions:
    template_parameter_validator: Prepare a model which accepts template
        parameters for validation.
"""

__all__ = ["template_parameter_validator", "validate_smd_path"]
__author__ = ["Gabriel Dorlhiac"]

import os
from typing import Dict, Any, Optional

import pydantic

from lute.io.db import read_latest_db_entry

pydantic_ver: int = int(pydantic.version.VERSION[0])
if pydantic_ver > 1:
    from pydantic.v1 import validator, BaseModel  # type: ignore
else:
    from pydantic import validator, BaseModel


def template_parameter_validator(template_params_name: str):
    """Populates a TaskParameters model with a set of validated TemplateParameters.

    This validator is intended for use with third-party Task's which use a
    templated configuration file. The validated template parameters are passed
    as a dictionary through a single parameter in the TaskParameters model. This
    dictionary is typically actually a separate pydantic BaseModel defined either
    within the TaskParameter class or externally. The other model provides the
    initial validation of each individual template parameter.

    This validator populates the TaskParameter model with the template parameters.
    It then returns `None` for the initial parameter that held these parameters.
    Returning None ensures that no attempt is made to pass the parameters on the
    command-line/when launching the third-party Task.
    """

    def _template_parameter_validator(
        cls, template_params: Optional[Any], values: Dict[str, Any]
    ) -> None:
        if template_params is not None:
            for param, value in template_params:
                if isinstance(value, BaseModel):
                    # Simplify Jinja templating by converting to dicts
                    values[param] = value.dict()
                elif value is not None:
                    values[param] = value
        return None

    return validator(template_params_name, always=True, allow_reuse=True)(
        _template_parameter_validator
    )


def validate_smd_path(smd_path_name: str):
    """Finds the path to a valid Smalldata file or raises an error."""

    def _validate_smd_path(cls, smd_path: str, values: Dict[str, Any]) -> str:
        if smd_path == "":
            # Try from database first
            hdf5_path: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}", "SubmitSMD", "result.payload"
            )
            if hdf5_path is not None:
                return hdf5_path
            else:
                exp: str = values["lute_config"].experiment
                run: int = int(values["lute_config"].run)
                hutch: str = exp[:3]
                hdf5_path = f"/sdf/data/lcls/ds/{hutch}/{exp}/hdf5/smalldata/{exp}_Run{run:04d}.h5"
                if os.path.exists(hdf5_path):
                    return hdf5_path
                raise ValueError("No path provided for hdf5 and cannot auto-determine!")

        return smd_path

    return validator(smd_path_name, always=True, allow_reuse=True)(_validate_smd_path)
