"""Machinary for the IO of configuration YAML files and their validation.

Functions:
    parse_config(taskname: str, config_path: str) -> TaskParameters: Parse a
        configuration file and return a TaskParameters object of validated
        parameters for a specific Task. Raises an exception if the provided
        configuration does not match the expected model.

Exceptions:
    ValidationError: Error raised by pydantic during data validation. (From
        Pydantic)
"""

# flake8: noqa: F403,F405

__all__ = ["parse_config"]
__author__ = "Gabriel Dorlhiac"

import os
import re
import warnings
from typing import List, Dict, Iterator, Any, Union, Optional

import pprint
import yaml

from lute.io.models import *
from lute.execution.debug_utils import LUTE_DEBUG_EXIT


def _isfloat(string: str) -> bool:
    try:
        float(string)
        return True
    except ValueError:
        return False


def _check_str_numeric(string: str) -> Union[str, int, float]:
    """Check if a string is an integer or float and return it as such.

    Args:
        string (str): Input string to check.

    Returns:
        output (str | int | float): Returns an int or float if the string can be
            converted to one. Otherwise returns the original string.
    """
    if string.isnumeric():
        return int(string)
    elif _isfloat(string):
        return float(string)
    else:
        return string


def substitute_variables(
    header: Dict[str, Any], config: Dict[str, Any], curr_key: Optional[str] = None
) -> None:
    """Performs variable substitutions on a dictionary read from config YAML file.

    Can be used to define input parameters in terms of other input parameters.
    This is similar to functionality employed by validators for parameters in
    the specific Task models, but is intended to be more accessible to users.
    Variable substitutions are defined using a minimal syntax from Jinja:
                               {{ experiment }}
    defines a substitution of the variable `experiment`. The characters `{{ }}`
    can be escaped if the literal symbols are needed in place.

    For example, a path to a file can be defined in terms of experiment and run
    values in the config file:
        MyTask:
          experiment: myexp
          run: 2
          special_file: /path/to/{{ experiment }}/{{ run }}/file.inp

    Acceptable variables for substitutions are values defined elsewhere in the
    YAML file. Environment variables can also be used if prefaced with a `$`
    character. E.g. to get the experiment from an environment variable:
        MyTask:
          run: 2
          special_file: /path/to/{{ $EXPERIMENT }}/{{ run }}/file.inp

    Args:
        config (Dict[str, Any]):  A dictionary of parsed configuration.

        curr_key (Optional[str]): Used to keep track of recursion level when scanning
            through iterable items in the config dictionary.

    Returns:
        subbed_config (Dict[str, Any]): The config dictionary after substitutions
            have been made. May be identical to the input if no substitutions are
            needed.
    """
    _sub_pattern = r"\{\{[^}{]*\}\}"
    iterable: Dict[str, Any] = config
    if curr_key is not None:
        # Need to handle nested levels by interpreting curr_key
        keys_by_level: List[str] = curr_key.split(".")
        for key in keys_by_level:
            iterable = iterable[key]
    else:
        ...
        # iterable = config
    for param, value in iterable.items():
        if isinstance(value, dict):
            new_key: str
            if curr_key is None:
                new_key = param
            else:
                new_key = f"{curr_key}.{param}"
            substitute_variables(header, config, curr_key=new_key)
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                if not isinstance(item, str):
                    continue
                matches_list: List[str] = re.findall(_sub_pattern, item)
                for m in matches_list:
                    item_key_to_sub_maybe_with_fmt = m[2:-2].strip().split(":")
                    item_key_to_sub = item_key_to_sub_maybe_with_fmt[0]
                    item_fmt = None
                    if len(item_key_to_sub_maybe_with_fmt) == 2:
                        item_fmt = item_key_to_sub_maybe_with_fmt[1]
                    item_sub: Any
                    if item_key_to_sub[0] == "$":
                        item_sub = os.getenv(item_key_to_sub[1:], None)
                        if item_sub is None:
                            # Check if we use a different env - substitution
                            # happens before environment reset
                            item_sub = os.getenv(f"LUTE_TENV_{item_key_to_sub[1:]}")
                        if item_sub is None:
                            print(
                                f"Environment variable {item_key_to_sub[1:]} not found! Cannot substitute in YAML config!",
                                flush=True,
                            )
                            continue
                        # substitutions from env vars will be strings, so
                        # convert back to numeric for formatting (e.g. {var:04d})
                        item_sub = _check_str_numeric(item_sub)
                    else:
                        try:
                            item_sub = config
                            for key in item_key_to_sub.split("."):
                                item_sub = item_sub[key]
                        except KeyError:
                            item_sub = header[item_key_to_sub]
                    item_pattern = (
                        m.replace("{{", r"\{\{")
                        .replace("}}", r"\}\}")
                        .replace("$", r"\$")
                    )
                    if item_fmt is not None:
                        item_sub = f"{item_sub:{item_fmt}}"
                    else:
                        item_sub = f"{item_sub}"
                    item = re.sub(item_pattern, item_sub, item)
                iterable[param][idx] = _check_str_numeric(item)
        # Scalars str - we skip numeric types
        elif isinstance(value, str):
            matches: List[str] = re.findall(_sub_pattern, value)
            for m in matches:
                key_to_sub_maybe_with_fmt: List[str] = m[2:-2].strip().split(":")
                key_to_sub: str = key_to_sub_maybe_with_fmt[0]
                fmt: Optional[str] = None
                if len(key_to_sub_maybe_with_fmt) == 2:
                    fmt = key_to_sub_maybe_with_fmt[1]
                sub: Any
                if key_to_sub[0] == "$":
                    sub = os.getenv(key_to_sub[1:], None)
                    if sub is None:
                        # Check if we use a different env - substitution happens
                        # before environment reset
                        sub = os.getenv(f"LUTE_TENV_{key_to_sub[1:]}")
                    if sub is None:
                        print(
                            f"Environment variable {key_to_sub[1:]} not found! Cannot substitute in YAML config!",
                            flush=True,
                        )
                        continue
                    # substitutions from env vars will be strings, so convert back
                    # to numeric in order to perform formatting later on (e.g. {var:04d})
                    sub = _check_str_numeric(sub)
                else:
                    try:
                        sub = config
                        for key in key_to_sub.split("."):
                            sub = sub[key]
                    except KeyError:
                        sub = header[key_to_sub]
                pattern: str = (
                    m.replace("{{", r"\{\{").replace("}}", r"\}\}").replace("$", r"\$")
                )
                if fmt is not None:
                    sub = f"{sub:{fmt}}"
                else:
                    sub = f"{sub}"
                iterable[param] = re.sub(pattern, sub, iterable[param])
            # Reconvert back to numeric values if needed...
            iterable[param] = _check_str_numeric(iterable[param])


def parse_config(task_name: str = "test", config_path: str = "") -> TaskParameters:
    """Parse a configuration file and validate the contents.

    Args:
        task_name (str): Name of the specific task that will be run.

        config_path (str): Path to the configuration file.

    Returns:
        params (TaskParameters): A TaskParameters object of validated
            task-specific parameters. Parameters are accessed with "dot"
            notation. E.g. `params.param1`.

    Raises:
        ValidationError: Raised if there are problems with the configuration
            file. Passed through from Pydantic.
    """
    cleaned_task_name: str = re.sub(r"_\d+$", "", task_name)
    task_config_name: str = f"{cleaned_task_name}Parameters"

    with open(config_path, "r") as f:
        docs: Iterator[Dict[str, Any]] = yaml.load_all(stream=f, Loader=yaml.FullLoader)
        header: Dict[str, Any] = next(docs)
        config: Dict[str, Any] = next(docs)
    substitute_variables(header, header)
    substitute_variables(header, config)
    LUTE_DEBUG_EXIT("LUTE_DEBUG_EXIT_AT_YAML", pprint.pformat(config))
    lute_config: Dict[str, AnalysisHeader] = {"lute_config": AnalysisHeader(**header)}
    try:
        task_config: Dict[str, Any] = dict(config[task_name])
        lute_config.update(task_config)
    except KeyError:
        warnings.warn(
            (
                f"{task_name} has no parameter definitions in YAML file."
                " Attempting default parameter initialization."
            )
        )
    parsed_parameters: TaskParameters = globals()[task_config_name](**lute_config)

    # Determine and record Task version information dynamically
    version_specifier: Optional[int] = getattr(
        parsed_parameters.Config, "version_specifier", None
    )
    if version_specifier is not None and version_specifier > 0:
        import json

        from lute.io.version_utils import record_version

        location: Optional[str] = getattr(
            parsed_parameters.Config, "version_location", None
        )
        diff_args: Optional[List[str]] = getattr(
            parsed_parameters.Config, "version_diff_args", None
        )
        version_info: Dict[str, str] = record_version(
            version_specifier=version_specifier, location=location, diff_args=diff_args
        )

        # Store where version information was taken from
        version_info["version-location"] = json.dumps(location)
        version_info["version-diff-args"] = json.dumps(diff_args)

        if version_info:
            parsed_parameters.Config.task_version = json.dumps(version_info)

    return parsed_parameters
