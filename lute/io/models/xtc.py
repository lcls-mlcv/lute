"""Models for the Xtc converter Task.

Classes:
    Xtc1to2(TaskParameters): Parameter model for the Xtc1to2 converter tool,
    which converst lcls1 style xtc files to lcls2 style.
"""

from typing import Dict, List, Optional, Tuple, Union

import pydantic

from lute.io.models.base import TaskParameters

pydantic_ver: int = int(pydantic.version.VERSION[0])
if pydantic_ver > 1:
    from pydantic.v1 import BaseModel, Field  # type: ignore
else:
    from pydantic import BaseModel, Field


class ConversionSpecification(BaseModel):
    xtc2_attr_name: str = Field(
        description="The name this field will have in the XTC2 file."
    )
    object_name: str = Field(
        description=(
            "The name used to access the object in psana1. E.g. `epix10k2M` if you "
            "would create the object as `psana.Detector('epix10k2M')`"
        )
    )
    object_type: str = Field(
        description="The psana1 object type used. E.g. `psana.Detector`"
    )
    object_field_name: Union[str, Tuple[str, str]] = Field(
        description=(
            "The field (or fields) on the constructed psana1 object used to get "
            "the per event data. E.g. `'calib'` if using `det.calib(evt)`. "
            "Or `('get', 'ebeamPhotonEnergy')` if using "
            "`det.get(evt).ebeamPhotonEnergy()`"
        )
    )


class ReadXtc1Parameters(TaskParameters):
    """Parameters for the xtc conversion Task."""

    eventfile: str = Field(
        default="",
        description="CSV file with event numbers. Otherwise will process all events.",
    )
    nevents: Optional[int] = Field(
        default=None,
        description=(
            "Optionally specify the number of events to use. If providing eventfile "
            "as well, that option will supercede this one."
        ),
    )
    xtc1_access_pattern: Dict[str, List[ConversionSpecification]] = Field(
        description=(
            "Provides information for how to access the data in XTC1. The top level "
            "keys will be used as the detector names in the XTC2 data."
        ),
    )


class WriteXtc2Parameters(TaskParameters):
    """Parameters for the xtc conversion Task."""

    node_id: str = Field(default="1", description="Node ID for the detector")
    output_file: str = Field(
        description="Where to write the output XTC2 file.", is_result=True
    )
    xtc1_access_pattern: Dict[str, List[ConversionSpecification]] = Field(
        description=(
            "Provides information for how to access the data in XTC1. The top level "
            "keys will be used as the detector names in the XTC2 data."
        ),
    )
