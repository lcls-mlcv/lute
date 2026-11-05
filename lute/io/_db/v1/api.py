"""Tools for working with the LUTE parameter and configuration database.

The current implementation relies on a sqlite backend database. In the future
this may change - therefore relatively few high-level API function calls are
intended to be public. These abstract away the details of the database interface
and work exclusively on LUTE objects.

Functions:
    record_analysis_db(cfg: DescribedAnalysis) -> None: Writes the configuration
        to the backend database.

    read_latest_db_entry(db_dir: str, task_name: str, param: str) -> Any: Retrieve
        the most recent entry from a database for a specific Task.

Exceptions:
    DatabaseError: Generic exception raised for LUTE database errors.
"""

__all__ = ["record_analysis_db", "read_latest_db_entry"]
__author__ = "Gabriel Dorlhiac"

import logging
import os
from typing import List, Dict, Any, Tuple, Optional, Union

import pydantic

from lute.io._db.common_sqlite import DatabaseError
from lute.execution.logging import get_logger
from lute.io.models.base import TaskParameters, TemplateParameters
from lute.tasks.dataclasses import TaskResult, TaskStatus, DescribedAnalysis

pydantic_ver: int = int(pydantic.version.VERSION[0])
if pydantic_ver > 1:
    from pydantic.v1 import BaseModel  # type: ignore
else:
    from pydantic import BaseModel

if __debug__:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

logger: logging.Logger = get_logger(__name__, is_task=False)


def _cfg_to_exec_entry_cols(
    cfg: DescribedAnalysis,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Converts AnalysisConfig to be entered into a exec_cfg table.

    Args:
        entries (Dict[str, Any]): Converted {name:value} dictionary.

        columns (Dict[str, str]): Converted {name:type} dictionary.
    """
    selected_env_vars: Dict[str, str] = {
        key: cfg.task_env[key]
        for key in cfg.task_env
        if ("LUTE_" in key and "_TENV_" not in key) or "SLURM_" in key
    }
    entry: Dict[str, Any] = {
        "env": ";".join(f"{key}={value}" for key, value in selected_env_vars.items()),
        "poll_interval": cfg.poll_interval,
        "communicator_desc": ";".join(desc for desc in cfg.communicator_desc),
    }
    columns: Dict[str, str] = {
        "env": "TEXT",
        "poll_interval": "REAL",
        "communicator_desc": "TEXT",
    }

    return entry, columns


def _params_to_entry_cols(
    params: TaskParameters,
) -> Tuple[
    Dict[str, Any],
    Dict[str, str],
    Dict[str, Any],
    Dict[str, str],
]:
    """Adapts a TaskParameters object to be entered into a table.

    Extracts the appropriate names and types from a TaskParameters object.
    Compound types (e.g. dicts) are recursively extracted and are given names
    where subparameters are delimited by ".". E.g. a parameter such as:
        my_param = {
            "a": 1,
            "b": 0.1,
        }
    will be converted into the following entries:
        ("my_param.a", "INTEGER"), ("my_param.b", "REAL").

    The `lute_config` analysis header is separated out and returned as a separate
    set of entries and columns. This particular field of the `TaskParameters`
    object contains shared configuration between `Task`s which is stored in a
    separated table.

    Args:
        params (TaskParameters): The TaskParameters object to convert to columns.

    Returns:
        entries (Dict[str, Any]): Converted {name:value} dictionary for Task
            specific parameters.

        columns (Dict[str, str]): Converted {name:type} dictionary for Task
            specific parameters.

        gen_entries (Dict[str, Any]): General configuration entry dictionary.

        gen_columns (Dict[str, str]): General configuration type
            information dictionary.
    """
    gen_entry: Dict[str, Any]
    gen_columns: Dict[str, str]
    entry: Dict[str, Any]
    columns: Dict[str, str]
    gen_entry, gen_columns = _dict_to_flatdicts(params.lute_config.dict())
    del params.lute_config
    # Note: We now store the Config options in the schema, but this is not added
    # to the output of `dict()` so no need to check/remove it.
    entry, columns = _dict_to_flatdicts(params.dict())

    return (
        entry,
        columns,
        gen_entry,
        gen_columns,
    )


def _result_to_entry_cols(
    result: TaskResult,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Adapts the required fields from a TaskResult to be entered into a table.

    Args:
        entries (Dict[str, Any]): Converted {name:value} dictionary.

        columns (Dict[str, str]): Converted {name:type} dictionary.
    """
    entry: Dict[str, Any] = {
        "result.task_status": str(result.task_status).split(".")[1],
        "result.summary": result.summary,
        "result.payload": result.payload,
        "result.impl_schemas": result.impl_schemas,
    }
    columns: Dict[str, str] = {
        "result.task_status": "TEXT",
        "result.summary": "TEXT",
        "result.payload": "BLOB",
        "result.impl_schemas": "TEXT",
    }

    return entry, columns


def _check_type(value: Any) -> str:
    """Return SQL type for a value."""
    if isinstance(value, int):
        return "INTEGER"
    elif isinstance(value, float):
        return "REAL"
    elif isinstance(value, str):
        return "TEXT"
    else:
        return "BLOB"


def _list_to_flatlists(
    l_entry: Union[List[Any], Tuple], curr_key: str = ""
) -> Tuple[List[Tuple[str, Any]], List[Tuple[str, str]]]:
    """Flatten lists for database storage.

    Indexes entries in the list using "[idx]" notation.
    Nested lists are handled with multiple indices [x][y][...]. This
    function is called recursively to handle nesting.

    Args:
        l_entry (List[Any]): Dictionary to flatten.

        curr_key (str): Current flattened key. Base key for indexing.

    Returns:
        flattened_params (List[Tuple[str, Any]]): List of (indexed_key, value) pairs.

        flattened_types (List[Tuple[str, str]]): List of (indexed_key, type) pairs.
            Types are one of TEXT, INTEGER, REAL
    """
    param_list: List[Tuple[str, Any]] = []
    type_list: List[Tuple[str, str]] = []
    idx: int
    indexed_value: Any
    for idx, indexed_value in enumerate(l_entry):
        indexed_curr_key: str = f"{curr_key}[{idx}]"
        if isinstance(indexed_value, tuple) or isinstance(indexed_value, list):
            x, y = _list_to_flatlists(indexed_value, indexed_curr_key)
            param_list.extend(x)
            type_list.extend(y)
        else:
            param_list.append((indexed_curr_key, indexed_value))
            type_list.append((indexed_curr_key, _check_type(indexed_value)))
    return param_list, type_list


def _dict_to_flatdicts(
    d: Dict[str, Any], curr_key: str = ""
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Flattens a dictionary delimiting parameters with a '.'.

    Works recursively. Also returns the type of each flattened value.
    Tuples/lists are handled flattened as well, using an indexing scheme. E.g.
    a["b"] in the nested dictionaries below:
                          {
                              "a": {
                                  "b": (1, 2),
                              },
                              # ...
                          }
    will be addressed as:
                          a.b[0] and a.b[1]
    for the values of 1 and 2, respectively.

    Args:
        d (Dict[str, Any]): Dictionary to flatten.

        curr_key (str): Current flattened key.

    Returns:
        flattened_params (Dict[str, Any]): Dictionary of flatkey:value pairs.

        flattened_types (Dict[str, str]): Dictionary of flatkey:type(value) pairs.
            Types are one of TEXT, INTEGER, REAL.
    """
    param_list: List[Tuple[str, Any]] = []
    type_list: List[Tuple[str, str]] = []
    for key, value in d.items():
        flat_key: str
        if curr_key == "":
            flat_key = key
        else:
            flat_key = f"{curr_key}.{key}"
        corrected_value: Any = value
        if isinstance(corrected_value, TemplateParameters):
            corrected_value = value.params
        if isinstance(corrected_value, BaseModel):
            corrected_value = corrected_value.dict()
        x: Union[Dict[str, Any], List[Tuple[str, Any]]]
        y: Union[Dict[str, str], List[Tuple[str, str]]]
        if isinstance(corrected_value, dict):
            x, y = _dict_to_flatdicts(corrected_value, curr_key=flat_key)
            param_list.extend(x.items())
            type_list.extend(y.items())
        elif isinstance(corrected_value, tuple) or isinstance(corrected_value, list):
            x, y = _list_to_flatlists(corrected_value, flat_key)
            param_list.extend(x)
            type_list.extend(y)
        else:
            param_list.append((flat_key, corrected_value))
            type_list.append((flat_key, _check_type(corrected_value)))

    return dict(param_list), dict(type_list)


def record_analysis_db(cfg: DescribedAnalysis) -> None:
    """Write an DescribedAnalysis object to the database.

    The DescribedAnalysis object is maintained by the Executor and contains all
    information necessary to fully describe a single `Task` execution. The
    contained fields are split across multiple tables within the database as
    some of the information can be shared across multiple Tasks. Refer to
    `docs/design/database.md` for more information on the database specification.
    """
    import sqlite3
    from ._sqlite import (
        _make_shared_table,
        _make_task_table,
        _add_row_no_duplicate,
        _add_task_entry,
    )

    assert isinstance(cfg.task_parameters, TaskParameters)
    try:
        work_dir: str = cfg.task_parameters.lute_config.work_dir
    except AttributeError:
        logger.error(
            (
                "Unable to access TaskParameters object. Likely wasn't created. "
                "Cannot store result."
            )
        )
        return
    del cfg.task_parameters.lute_config.work_dir

    exec_entry, exec_columns = _cfg_to_exec_entry_cols(cfg)
    task_name: str = cfg.task_result.task_name
    # All `Task`s have an AnalysisHeader, but this info can be shared so is
    # split into a different table
    (
        task_entry,  # Dict[str, Any]
        task_columns,  # Dict[str, str]
        gen_entry,  # Dict[str, Any]
        gen_columns,  # Dict[str, str]
    ) = _params_to_entry_cols(cfg.task_parameters)
    x, y = _result_to_entry_cols(cfg.task_result)
    task_entry.update(x)
    task_columns.update(y)

    db_path: str = f"{work_dir}/lute.db"
    con: sqlite3.Connection = sqlite3.Connection(db_path)
    with con:
        try:
            # --- Table Creation ---#
            if not _make_shared_table(con, "gen_cfg", gen_columns):
                raise DatabaseError("Could not make general configuration table!")
            if not _make_shared_table(con, "exec_cfg", exec_columns):
                raise DatabaseError("Could not make Executor configuration table!")
            if not _make_task_table(con, task_name, task_columns):
                raise DatabaseError(f"Could not make Task table for: {task_name}!")

            # --- Row Addition ---#
            gen_id: int = _add_row_no_duplicate(con, "gen_cfg", gen_entry)
            exec_id: int = _add_row_no_duplicate(con, "exec_cfg", exec_entry)

            full_task_entry: Dict[str, Any] = {
                "gen_cfg_id": gen_id,
                "exec_cfg_id": exec_id,
            }
            full_task_entry.update(task_entry)
            # Prepare flag to indicate whether the task entry is valid or not
            # By default we say it is assuming proper completion
            valid_flag: int = (
                1 if cfg.task_result.task_status == TaskStatus.COMPLETED else 0
            )
            full_task_entry.update({"valid_flag": valid_flag})

            _add_task_entry(con, task_name, full_task_entry)
        except sqlite3.OperationalError as err:
            logger.error(f"Database storage error: {err}")
    try:
        os.chmod(db_path, 0o664)
    except Exception:
        logger.error("Cannot setup permissions on database!")


def read_latest_db_entry(
    db_dir: str,
    task_name: str,
    param: str,
    valid_only: bool = True,
    for_run: Optional[Union[str, int]] = os.getenv("RUN"),
) -> Optional[Any]:
    """Read most recent value entered into the database for a Task parameter.

    (Will be updated for schema compliance as well as Task name.)

    Args:
        db_dir (str): Database location.

        task_name (str): The name of the Task to check the database for.

        param (str): The parameter name for the Task that we want to retrieve.

        valid_only (bool): Whether to consider only valid results or not. E.g.
            An input file may be useful even if the Task result is invalid
            (Failed). Default = True.

        for_run (Optional[str | int]): Only consider latest entries from the
            specific experiment run provided.

    Returns:
        val (Any): The most recently entered value for `param` of `task_name`
            that can be found in the database. Returns None if nothing found.
    """
    import sqlite3
    from ._sqlite import _select_from_db

    db_path: str = f"{db_dir}/lute.db"
    con: sqlite3.Connection = sqlite3.Connection(db_path)
    with con:
        try:
            cond: Dict[str, str] = {}
            if valid_only:
                cond = {"valid_flag": "1"}
            entries: Any = _select_from_db(
                con, task_name, f'gen_cfg_id,"{param}"', cond
            )
            if for_run is not None:
                gen_cfg_entries: Any = _select_from_db(
                    con, "gen_cfg", "id", {"run": str(for_run)}
                )
                task_entries_for_run: List[Any] = [
                    entry[1] for entry in entries if (entry[0],) in gen_cfg_entries
                ]
                if task_entries_for_run:
                    return task_entries_for_run[-1]
                return None
            else:
                return entries[-1][1]
        except sqlite3.OperationalError as err:
            logger.error(f"Cannot retrieve value {param} due to: {err}")
            return None
