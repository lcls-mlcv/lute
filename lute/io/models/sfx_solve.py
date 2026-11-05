"""Models for structure solution in serial femtosecond crystallography.

Classes:
    DimpleSolveParameters(ThirdPartyParameters): Perform structure solution
        using CCP4's dimple (molecular replacement).
"""

__all__ = ["DimpleSolveParameters", "RunSHELXCParameters"]
__author__ = "Gabriel Dorlhiac"

import os
from typing import Union, Optional, Dict, Any

import pydantic

from lute.io.models.base import ThirdPartyParameters
from lute.io.db import read_latest_db_entry

pydantic_ver: int = int(pydantic.version.VERSION[0])
if pydantic_ver > 1:
    from pydantic.v1 import Field, validator, PositiveFloat, PositiveInt, root_validator  # type: ignore
else:
    from pydantic import Field, validator, PositiveFloat, PositiveInt, root_validator


class DimpleSolveParameters(ThirdPartyParameters):
    """Parameters for CCP4's dimple program.

    There are many parameters. For more information on
    usage, please refer to the CCP4 documentation, here:
    https://ccp4.github.io/dimple/
    """

    class Config(ThirdPartyParameters.Config):
        """Identical to super-class Config but includes a result."""

        set_result: bool = True
        """Whether the Executor should mark a specified parameter as a result."""

        result_from_params: str = ""
        """Defines a result from the parameters. Use a validator to do so."""

    executable: str = Field(
        "/sdf/group/lcls/ds/tools/ccp4-8.0/bin/dimple",
        description="CCP4 Dimple for solving structures with MR.",
        flag_type="",
    )
    # Positional requirements - all required.
    in_file: str = Field(
        "",
        description="Path to input mtz.",
        flag_type="",
    )
    pdb: str = Field("", description="Path to a PDB.", flag_type="")
    out_dir: str = Field("", description="Output DIRECTORY.", flag_type="")
    # Most used options
    mr_thresh: PositiveFloat = Field(
        0.4,
        description="Threshold for molecular replacement.",
        flag_type="--",
        rename_param="mr-when-r",
    )
    slow: Optional[bool] = Field(
        False, description="Perform more refinement.", flag_type="--"
    )
    # Other options (IO)
    hklout: str = Field(
        "final.mtz", description="Output mtz file name.", flag_type="--"
    )
    xyzout: str = Field(
        "final.pdb", description="Output PDB file name.", flag_type="--"
    )
    icolumn: Optional[str] = Field(
        # "IMEAN",
        description="Name for the I column.",
        flag_type="--",
    )
    sigicolumn: Optional[str] = Field(
        # "SIG<ICOL>",
        description="Name for the Sig<I> column.",
        flag_type="--",
    )
    fcolumn: Optional[str] = Field(
        # "F",
        description="Name for the F column.",
        flag_type="--",
    )
    sigfcolumn: Optional[str] = Field(
        # "F",
        description="Name for the Sig<F> column.",
        flag_type="--",
    )
    libin: Optional[str] = Field(
        description="Ligand descriptions for refmac (LIBIN).", flag_type="--"
    )
    refmac_key: Optional[str] = Field(
        description="Extra Refmac keywords to use in refinement.",
        flag_type="--",
        rename_param="refmac-key",
    )
    free_r_flags: Optional[str] = Field(
        description="Path to a mtz file with freeR flags.",
        flag_type="--",
        rename_param="free-r-flags",
    )
    freecolumn: Optional[Union[int, float]] = Field(
        # 0,
        description="Refree column with an optional value.",
        flag_type="--",
    )
    img_format: Optional[str] = Field(
        description="Format of generated images. (png, jpeg, none).",
        flag_type="-",
        rename_param="f",
    )
    white_bg: bool = Field(
        False,
        description="Use a white background in Coot and in images.",
        flag_type="--",
        rename_param="white-bg",
    )
    no_cleanup: bool = Field(
        False,
        description="Retain intermediate files.",
        flag_type="--",
        rename_param="no-cleanup",
    )
    # Calculations
    no_blob_search: bool = Field(
        False,
        description="Do not search for unmodelled blobs.",
        flag_type="--",
        rename_param="no-blob-search",
    )
    anode: bool = Field(
        False,
        description="Use SHELX/AnoDe to find peaks in the anomalous map.",
        flag_type="--",
    )
    # Run customization
    no_hetatm: bool = Field(
        False,
        description="Remove heteroatoms from the given model.",
        flag_type="--",
        rename_param="no-hetatm",
    )
    rigid_cycles: Optional[PositiveInt] = Field(
        # 10,
        description="Number of cycles of rigid-body refinement to perform.",
        flag_type="--",
        rename_param="rigid-cycles",
    )
    jelly: Optional[PositiveInt] = Field(
        # 4,
        description="Number of cycles of jelly-body refinement to perform.",
        flag_type="--",
    )
    restr_cycles: Optional[PositiveInt] = Field(
        # 8,
        description="Number of cycles of refmac final refinement to perform.",
        flag_type="--",
        rename_param="restr-cycles",
    )
    lim_resolution: Optional[PositiveFloat] = Field(
        description="Limit the final resolution.", flag_type="--", rename_param="reso"
    )
    weight: Optional[str] = Field(
        # "auto-weight",
        description="The refmac matrix weight.",
        flag_type="--",
    )
    mr_prog: Optional[str] = Field(
        # "phaser",
        description="Molecular replacement program. phaser or molrep.",
        flag_type="--",
        rename_param="mr-prog",
    )
    mr_num: Optional[Union[str, int]] = Field(
        # "auto",
        description="Number of molecules to use for molecular replacement.",
        flag_type="--",
        rename_param="mr-num",
    )
    mr_reso: Optional[PositiveFloat] = Field(
        # 3.25,
        description="High resolution for molecular replacement. If >10 interpreted as eLLG.",
        flag_type="--",
        rename_param="mr-reso",
    )
    itof_prog: Optional[str] = Field(
        description="Program to calculate amplitudes. truncate, or ctruncate.",
        flag_type="--",
        rename_param="ItoF-prog",
    )

    @validator("in_file", always=True)
    def validate_in_file(cls, in_file: str, values: Dict[str, Any]) -> str:
        if in_file == "":
            get_hkl_file: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}", "ManipulateHKL", "out_file"
            )
            if get_hkl_file:
                return get_hkl_file
        return in_file

    @validator("out_dir", always=True)
    def validate_out_dir(cls, out_dir: str, values: Dict[str, Any]) -> str:
        if out_dir == "":
            get_hkl_file: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}", "ManipulateHKL", "out_file"
            )
            if get_hkl_file:
                return os.path.dirname(get_hkl_file)
        return out_dir

    @root_validator(pre=False)
    def define_result(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        out_dir: str = values["out_dir"]
        result: str
        if out_dir != "":
            result = f"{out_dir}/final.mtz;{out_dir}/final.pdb"
        else:
            result = ""
        cls.Config.result_from_params = result
        return values


class RunSHELXCParameters(ThirdPartyParameters):
    """Parameters for CCP4's SHELXC program.

    SHELXC prepares files for SHELXD and SHELXE.

    For more information please refer to the official documentation:
    https://www.ccp4.ac.uk/html/crank.html
    """

    executable: str = Field(
        "/sdf/group/lcls/ds/tools/ccp4-8.0/bin/shelxc",
        description="CCP4 SHELXC. Generates input files for SHELXD/SHELXE.",
        flag_type="",
    )
    placeholder: str = Field(
        "xx", description="Placeholder filename stem.", flag_type=""
    )
    in_file: str = Field(
        "",
        description="Input file for SHELXC with reflections AND proper records.",
        flag_type="",
    )

    @validator("in_file", always=True)
    def validate_in_file(cls, in_file: str, values: Dict[str, Any]) -> str:
        if in_file == "":
            # get_hkl needed to be run to produce an XDS format file...
            xds_format_file: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}", "ManipulateHKL", "out_file"
            )
            if xds_format_file:
                in_file = xds_format_file
        if in_file[0] != "<":
            # Need to add a redirection for this program
            # Runs like `shelxc xx <input_file.xds`
            in_file = f"<{in_file}"
        return in_file
