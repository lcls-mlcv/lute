"""Models for geometry and geometry optimization Tasks.

Classes:
    OptimizeAgBhGeometryExhaustiveParameters(TaskParameters): Parameter model for
        the OptimizeAgBhGeometryExhaustive Task. Used to optimize detector distance
        and beam center after acquiring Ag Behenate data.
"""

__all__ = ["OptimizeAgBhGeometryExhaustiveParameters"]
__author__ = "Gabriel Dorlhiac"

from typing import Optional, Tuple

import pydantic

from lute.io.models.base import TaskParameters
from lute.io.models.validators import validate_smd_path

pydantic_ver: int = int(pydantic.version.VERSION[0])
if pydantic_ver > 1:
    from pydantic.v1 import Field  # type: ignore
else:
    from pydantic import Field


class OptimizeAgBhGeometryExhaustiveParameters(TaskParameters):
    """TaskParameter model for OptimizeAgBhGeometryExhaustive Task.

    This Task does geometry optimization of detector distance and beam center
    based on a powder image produced from acquiring a run of Ag Behenate.
    """

    _find_smd_path = validate_smd_path("powder")

    detname: str = Field(description="Name of the detector to optimize geometry for.")

    powder: str = Field(
        "", description="Path to the powder image, or file containing it."
    )

    geom_out_dir: str = Field(
        "", description="Directory to write new geometry files to."
    )

    mask: Optional[str] = Field(
        None, description="Path to a detector mask, or file containing it."
    )

    n_peaks: int = Field(4, description="")

    n_iterations: int = Field(
        5, description="Number of optimization iterations. Per MPI rank."
    )

    threshold: float = Field(
        1e6,
        description=(
            "Pixels in the powder image with an intensity above this threshold "
            "are set to 0."
        ),
    )

    dx: Tuple[float, float, int] = Field(
        (-6, 6, 5),
        description=(
            "Defines the search radius for beam center x position as offsets from "
            "the image center. Format: (left, right, num_steps). In units of pixels."
        ),
    )

    dy: Tuple[float, float, int] = Field(
        (-6, 6, 5),
        description=(
            "Defines the search radius for beam center y position as offsets from "
            "the image center. Format: (up, down, num_steps). In units of pixels."
        ),
    )

    center_guess: Optional[Tuple[float, float]] = Field(
        None, description=("Provide an optional starting guess for the beam center.")
    )

    distance_guess: Optional[float] = Field(
        None,
        description="Provide an optional starting guess for the detector distance (mm).",
    )
