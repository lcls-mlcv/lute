"""Models for optimizing detector geometry using PyFAI and Bayesian optimization.

Classes:
    - BayFAIParameters:
        Parameters for running BayFAI
"""

__all__ = ["BayFAIParameters"]
__author__ = "Louis Conreux"

import os
from typing import Any, Dict, List, Tuple, Optional
from pydantic import BaseModel, Field, validator

from lute.io.models.base import TaskParameters
from lute.io.models.validators import (
    validate_smd_path,
)


def validate_geometry_path(output_path_name: str):
    """Dynamically generates the output geometry path for the optimization results."""

    def _validate_geometry_path(cls, output_path: str, values: Dict[str, Any]) -> str:
        if output_path == "":
            work_dir = values["lute_config"].work_dir
            run = int(values["lute_config"].run)
            detname = values["detname"]
            geom_dir = os.path.join(work_dir, "geom")
            os.makedirs(geom_dir, exist_ok=True)
            output_run_path = os.path.join(geom_dir, f"{run}-end_{detname}.data")
            return output_run_path
        return output_path

    return validator(output_path_name, always=True)(_validate_geometry_path)


class BayFAIParameters(TaskParameters):
    """Parameters for optimizing detector geometry using PyFAI and Bayesian optimization.

    BayFAI has default hyperparameters that can be overriden by the user.
    """

    class Config(TaskParameters.Config):
        set_result: bool = True
        """Whether the Executor should mark a specified parameter as a result."""

    class BayFAIHyperparameters(BaseModel):
        """BayFAI hyperparameters."""

        n_samples: int = Field(
            default=20,
            description="Number of samples to initialize the Gaussian Process.",
        )

        n_iterations: int = Field(
            default=80,
            description="Number of iterations of Bayesian Optimization",
        )

        max_rings: int = Field(
            default=8,
            description="Maximum number of rings to search for Bragg peaks.",
        )

        pts_per_deg: float = Field(
            default=2.0,
            description="Number of Bragg peaks to extract per azimuthal degree.",
        )

        Imin: float = Field(
            default=95,
            description="Minimum intensity percentile threshold for Bragg peak detection.",
        )

        prior: bool = Field(
            default=True,
            description="Whether to sample initial points around the center of search space or randomly.",
        )

        beta: float = Field(
            default=1.96,
            description="Exploration-exploitation trade-off hyperparameter for Upper Confidence Bound acquisition function.",
        )

        step: int = Field(
            default=5,
            description="Size of the refinement space around best parameters.",
        )

        lbda: float = Field(
            default=0.1,
            description="Penalty weight in final scoring to select winner geometry.",
        )

        seed: Optional[int] = Field(
            default=None,
            description="Random seed for reproducibility.",
        )

        n_cand: Optional[int] = Field(
            default=8192,
            description="Number of random grid points scored by UCB per BO step (0 or None = whole grid).",
        )

    parallelized: Optional[str] = Field(
        None,
        description="Name of the geometry parameter to distribute across MPI ranks as a "
        "sliding window (e.g. 'dist'). If None, all ranks instead run BO over the same space.",
    )

    tiling: Optional[Dict[str, Any]] = Field(
        None,
        description="Per-rank tiling of the parallelized 'dist' axis, e.g. "
        "{mode: geometric, min: 0.0348, max: 0.4002}: rank r searches [b_r, b_r+1] with "
        "b = min*(max/min)**(r/size). Replaces dist center/bounds/resolutions. None = linear grid.",
    )

    fixed: List[str] = Field(
        ["rot3"],
        description="List of fixed parameters for the optimization (rot3 is fixed by default) "
        "because diffraction rings are invariance by rotation around the beam axis.",
    )

    center: Dict[str, float] = Field(
        {
            "dist": 0.1,
            "poni1": 0.0,
            "poni2": 0.0,
            "rot1": 0.0,
            "rot2": 0.0,
            "rot3": 0.0,
        },
        description="Center of the search space for the detector geometry parameters.",
    )

    bounds: Dict[str, Tuple[float, float]] = Field(
        {
            "dist": (-0.01, 0.01),
            "poni1": (-0.0025, 0.0025),
            "poni2": (-0.0025, 0.0025),
            "rot1": (-0.1, 0.1),
            "rot2": (-0.1, 0.1),
            "rot3": (-0.1, 0.1),
        },
        description="Bounds of the search space for the detector geometry parameters.",
    )

    resolutions: Dict[str, float] = Field(
        {
            "dist": 0.001,
            "poni1": 0.0001,
            "poni2": 0.0001,
            "rot1": 0.02,
            "rot2": 0.02,
            "rot3": 0.02,
        },
        description="Resolution of the search space for the detector geometry parameters.",
    )

    detname: str = Field(
        "",
        description="Detector name",
    )

    calibrant: str = Field(
        "",
        description="Calibrant used for the calibration supported by pyFAI: https://github.com/silx-kit/pyFAI/tree/main/src/pyFAI/resources/calibration, \n e.g. Silver Behenate 'AgBh', LaB6 'CeO2', etc.",
    )

    wavelength: float = Field(
        1e-10,
        description=(
            "Wavelength in meters. If provided (non-default), it takes precedence "
            "over the mean photon energy read from the h5 file."
        ),
    )

    h5: str = Field(
        "",
        description="Smalldata hdf5 file path to be used for the calibration.",
    )

    out_file: str = Field(
        "",
        description="Path to the output .data file that will contain the optimized detector geometry.",
        is_result=True,
    )

    bayfai_params: BayFAIHyperparameters = Field(
        BayFAIHyperparameters(),
        description="BayFAI hyperparameters.",
    )

    _find_smd_path = validate_smd_path("h5")

    _find_out_file_path = validate_geometry_path("out_file")
