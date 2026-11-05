"""Models for all test Tasks.

Classes:
    TestParameters(TaskParameters): Model for most basic test case. Single
        core first-party Task. Uses only communication via pipes.

    TestBinaryParameters(ThirdPartyParameters): Parameters for a simple multi-
        threaded binary executable.

    TestSocketParameters(TaskParameters): Model for first-party test requiring
        communication via socket.

    TestWriteOutputParameters(TaskParameters): Model for test Task which writes
        an output file. Location of file is recorded in database.

    TestReadOutputParameters(TaskParameters): Model for test Task which locates
        an output file based on an entry in the database, if no path is provided.
"""

__all__ = [
    "TestParameters",
    "TestBinaryParameters",
    "TestBinaryErrParameters",
    "TestSocketParameters",
    "TestWriteOutputParameters",
    "TestReadOutputParameters",
    "TestRequestParameters",
    "TestUnresponsiveParameters",
]
__author__ = "Gabriel Dorlhiac"

from typing import Dict, Any, Optional

import pydantic

from lute.io.models.base import TaskParameters, ThirdPartyParameters
from lute.io.db import read_latest_db_entry

pydantic_ver: int = int(pydantic.version.VERSION[0])
if pydantic_ver > 1:
    from pydantic.v1 import BaseModel, Field, validator  # type: ignore
else:
    from pydantic import BaseModel, Field, validator


class TestParameters(TaskParameters):
    """Parameters for the test Task `Test`."""

    float_var: float = Field(0.01, description="A floating point number.")
    str_var: str = Field("test", description="A string.")

    class CompoundVar(BaseModel):
        int_var: int = 1
        dict_var: Dict[str, str] = {"a": "b"}

    compound_var: CompoundVar = Field(
        description=(
            "A compound parameter - consists of a `int_var` (int) and `dict_var`"
            " (Dict[str, str])."
        )
    )
    throw_error: bool = Field(
        False, description="If `True`, raise an exception to test error handling."
    )


class TestBinaryParameters(ThirdPartyParameters):
    executable: str = Field(
        "/sdf/home/d/dorlhiac/test_tasks/test_threads",
        description="Multi-threaded test binary.",
    )
    p_arg1: int = Field(1, descriptions="Number of threads.")


class TestBinaryErrParameters(ThirdPartyParameters):
    """Same as TestBinary, but exits with non-zero code."""

    executable: str = Field(
        "/sdf/home/d/dorlhiac/test_tasks/test_threads_err",
        description="Multi-threaded tes tbinary with non-zero exit code.",
    )
    p_arg1: int = Field(1, description="Number of threads.")


class TestSocketParameters(TaskParameters):
    array_size: int = Field(
        10000, description="Size of an array to send (number of values) via socket."
    )
    num_arrays: int = Field(10, description="Number of arrays to send via socket.")


class TestWriteOutputParameters(TaskParameters):
    class Config(TaskParameters.Config):
        set_result: bool = True

    outfile_name: str = Field(
        "test_output.txt", description="Outfile name without full path.", is_result=True
    )
    num_vals: int = Field(100, description='Number of values to "process"')


class TestReadOutputParameters(TaskParameters):
    in_file: str = Field("", description="File to read in. (Full path)")

    @validator("in_file", always=True)
    def validate_in_file(cls, in_file: str, values: Dict[str, Any]) -> str:
        if in_file == "":
            filename: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}", "TestWriteOutput", "outfile_name"
            )
            if filename is not None:
                return f"{values['lute_config'].work_dir}/{filename}"
        return in_file


class TestRequestParameters(TaskParameters):
    placeholder: str = Field("Test", description="This is just a placeholder.")


class TestUnresponsiveParameters(TaskParameters):
    placeholder: str = Field("Test", description="This is just a placeholder.")
