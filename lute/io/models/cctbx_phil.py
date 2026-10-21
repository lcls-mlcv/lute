"""Helpers for building CCTBX phil files from nested TaskParameters models.

A CCTBX phil file is a tree of scopes:

    filter {
      unit_cell {
        value {
          relative_length_tolerance = 0.03
        }
      }
    }

The phil parameters of the CCTBX Tasks are written in the YAML with the same
nesting, and modelled by nested `PhilScope` models whose field names are the
phil names:

    phil_parameters:
      filter:
        unit_cell:
          value:
            relative_length_tolerance: 0.03

Only the commonly used parameters are modelled (typed, documented). Every
scope accepts additional keys, which are passed to CCTBX as-is, so any CCTBX
parameter can be set without changing LUTE. `None` values and empty scopes are
never written, so CCTBX falls back to its own defaults for anything unset.

Functions:
    dict_to_phil(tree): Render a nested dict as phil text.

    phil_parameter_validator(name): Validator for a ThirdPartyParameters model
        which renders its phil parameters into the `phil_text` template
        variable.

Classes:
    PhilScope(BaseModel): Base model for a phil scope.
"""

__all__ = ["PhilScope", "dict_to_phil", "phil_parameter_validator"]

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, validator

logger: logging.Logger = logging.getLogger(__name__)


def _phil_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def dict_to_phil(tree: Dict[str, Any], indent: int = 0) -> str:
    """Render a nested dict as phil text.

    Mappings become scopes (`key { ... }`), lists become repeated definitions
    (`key = item` once per item, e.g. for `input.path`) and anything else
    becomes a single definition (`key = value`). `None` values and empty
    scopes are skipped.

    Args:
        tree (Dict[str, Any]): Nested dictionary of phil scopes and values.

        indent (int): Indentation level (2 spaces per level).

    Returns:
        phil (str): The phil text.
    """
    pad: str = "  " * indent
    lines: List[str] = []
    for key, value in tree.items():
        if value is None:
            continue
        if isinstance(value, dict):
            body: str = dict_to_phil(value, indent + 1)
            if body:
                lines.append(f"{pad}{key} {{\n{body}\n{pad}}}")
        elif isinstance(value, (list, tuple)):
            lines.extend(f"{pad}{key} = {_phil_value(item)}" for item in value)
        else:
            lines.append(f"{pad}{key} = {_phil_value(value)}")
    return "\n".join(lines)


class PhilScope(BaseModel):
    """A phil scope. Field names are phil names; sub-scopes are PhilScopes.

    Keys that are not modelled are accepted and passed through to CCTBX
    unvalidated.
    """

    class Config(BaseModel.Config):  # type: ignore
        extra: str = "allow"

    def unmodelled_paths(self, prefix: str = "") -> List[str]:
        """Phil paths that were passed through without validation."""
        paths: List[str] = []
        for key, value in self.__dict__.items():
            if key not in self.__fields__:
                paths.append(f"{prefix}{key}")
            elif isinstance(value, PhilScope):
                paths.extend(value.unmodelled_paths(prefix=f"{prefix}{key}."))
        return paths

    def to_phil(self) -> str:
        """Render this scope (and everything below it) as phil text."""
        return dict_to_phil(self.dict())


def phil_parameter_validator(phil_params_name: str):
    """Render a PhilScope model into the `phil_text` template variable.

    The nested parameters are also stored as `phil_tree` so that other
    validators of the TaskParameters model can read them (e.g. to set a
    result). `None` is returned for the original parameter so that it is not
    passed on the command-line.
    """

    def _phil_parameter_validator(
        cls, phil_params: Optional[PhilScope], values: Dict[str, Any]
    ) -> None:
        if phil_params is not None:
            unmodelled: List[str] = phil_params.unmodelled_paths()
            if unmodelled:
                logger.info(
                    "Phil parameters passed to CCTBX without validation "
                    f"(check spelling): {unmodelled}"
                )
            values["phil_tree"] = phil_params.dict()
            values["phil_text"] = phil_params.to_phil()
        return None

    return validator(phil_params_name, always=True, allow_reuse=True)(
        _phil_parameter_validator
    )
