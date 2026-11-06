import os
import sqlite3

import pytest

import lute.io._db.v3.api as api
from lute.io._db.common_sqlite import DatabaseError
from lute.io.models.base import AnalysisHeader
from lute.io.models.tests import TestWriteOutputParameters
from lute.tasks.dataclasses import DescribedAnalysis, TaskResult, TaskStatus


def _params(work_dir, run):
    params = TestWriteOutputParameters(
        lute_config=AnalysisHeader(experiment="mfxtest", run=run, work_dir=work_dir)
    )
    # `add_execution` deletes these from the Config class after each write
    params.Config.version_specifier = None
    params.Config.task_version = None
    return params


def _described(params):
    return DescribedAnalysis(
        task_result=TaskResult(
            task_name="TestWriteOutput",
            task_status=TaskStatus.COMPLETED,
            summary="",
            payload="",
        ),
        task_parameters=params,
        task_env={},
        executor_name="Executor",
        poll_interval=0.05,
        communicator_desc=["PipeCommunicator: pipe"],
    )


def _rows(db_dir, sql):
    con = sqlite3.connect(os.path.join(db_dir, "lute.db"))
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def _placeholder_then_update(work_dir, run):
    """Task stores a placeholder, then the Executor completes it (no-pickle mode)."""
    from lute.execution.executor import construct_task_parameters

    row_ids = api.record_parameters_db(_params(work_dir, run))  # Task side
    # Executor side: rebuild the parameters from the database, as `task_started`
    definition, values = api.get_task_parameters_defn_and_params(work_dir, row_ids)
    params = construct_task_parameters(schema=definition, values=values)
    params.lute_config.work_dir = work_dir
    api.update_analysis_db(_described(params), row_ids)


EXEC_UNITS = (
    "SELECT e.id, u.key FROM executions e "
    "JOIN execution_units eu ON eu.execution_id = e.id "
    "JOIN units u ON u.id = eu.unit_id ORDER BY e.id"
)


def test_fixture_executions_linked_to_their_run(v3_db_dir):
    assert _rows(v3_db_dir, EXEC_UNITS) == [
        (1, "experiment=mfxtest;run=36"),
        (2, "experiment=mfxtest;run=36"),
        (3, "experiment=mfxtest;run=36"),
        (4, "experiment=mfxtest;run=37"),
        (5, "experiment=mfxtest;run=37"),
    ]
    # One link per execution, one unit per run (shared by Tasks and retries)
    assert _rows(v3_db_dir, "SELECT COUNT(*) FROM execution_units") == [(5,)]
    assert _rows(v3_db_dir, "SELECT COUNT(*) FROM units") == [(2,)]


def test_one_execution_gives_one_link(tmp_path):
    work_dir = str(tmp_path)
    api.record_analysis_db(_described(_params(work_dir, "0036")))
    assert _rows(work_dir, EXEC_UNITS) == [(1, "experiment=mfxtest;run=36")]


def test_placeholder_then_update_gives_one_link(tmp_path):
    work_dir = str(tmp_path)
    _placeholder_then_update(work_dir, "0036")
    assert _rows(work_dir, "SELECT COUNT(*) FROM executions") == [(1,)]
    assert _rows(work_dir, EXEC_UNITS) == [(1, "experiment=mfxtest;run=36")]

    # Same run in another format: a second execution, same unit
    _placeholder_then_update(work_dir, 36)
    assert _rows(work_dir, EXEC_UNITS) == [
        (1, "experiment=mfxtest;run=36"),
        (2, "experiment=mfxtest;run=36"),
    ]
    assert _rows(work_dir, "SELECT COUNT(*) FROM units") == [(1,)]


def test_both_write_paths_share_units(tmp_path):
    work_dir = str(tmp_path)
    api.record_analysis_db(_described(_params(work_dir, 36)))
    _placeholder_then_update(work_dir, "36")
    api.record_analysis_db(_described(_params(work_dir, 37)))
    assert _rows(work_dir, "SELECT unit_id FROM execution_units ORDER BY 1") == [
        (1,),
        (1,),
        (2,),
    ]


@pytest.fixture
def no_run_env(monkeypatch):
    # AnalysisHeader takes the run from RUN_NUM when the header has none
    monkeypatch.delenv("RUN_NUM", raising=False)


def test_execution_without_run_refused(tmp_path, no_run_env):
    work_dir = str(tmp_path)
    with pytest.raises(DatabaseError, match="must have a run"):
        api.record_analysis_db(_described(_params(work_dir, "")))
    # Rolled back: nothing written
    for table in ("executions", "config", "results", "parameters", "units"):
        assert _rows(work_dir, f"SELECT COUNT(*) FROM {table}") == [(0,)]


def test_placeholder_without_run_refused(tmp_path, no_run_env):
    work_dir = str(tmp_path)
    with pytest.raises(DatabaseError, match="must have a run"):
        api.record_parameters_db(_params(work_dir, ""))
    for table in ("config", "parameters", "units"):
        assert _rows(work_dir, f"SELECT COUNT(*) FROM {table}") == [(0,)]
