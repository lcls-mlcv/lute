import os
import shutil
import sqlite3

import pytest

import lute.io._db.v3.api as api
from lute.io._db.v3._units import create_group, get_or_create_unit
from lute.io._db.v3._versioning import open_db

from test_db_v2 import BASELINE_LATEST
from test_db_v3_write import _described, _params


def _latest(db_dir, param="outfile_name", task="TestWriteOutput", **kwargs):
    return api.read_latest_db_entry(db_dir, task, param, **kwargs)


def test_config_has_no_experiment_or_run(v3_db_dir):
    con = sqlite3.connect(os.path.join(v3_db_dir, "lute.db"))
    cols = [r[1] for r in con.execute("PRAGMA table_info(config)")]
    n_config = con.execute("SELECT COUNT(*) FROM config").fetchone()[0]
    con.close()
    assert "experiment" not in cols
    assert "run" not in cols
    # Runs 36 and 37 share the same header fields: one row
    assert n_config == 1


def test_same_answers_as_v2(v2_db_dir, v3_db_dir):
    """The read path through units gives the v2 answers on the same data."""
    for for_run, valid_only, param, _ in BASELINE_LATEST:
        kwargs = dict(param=param, valid_only=valid_only, for_run=for_run)
        assert _latest(v3_db_dir, **kwargs) == _latest_v2(v2_db_dir, **kwargs)


def _latest_v2(db_dir, param, **kwargs):
    import lute.io._db.v2.api as v2_api

    return v2_api.read_latest_db_entry(db_dir, "TestWriteOutput", param, **kwargs)


@pytest.mark.parametrize("run", [36, "36", "0036", " 36 "])
def test_run_in_any_format(v3_db_dir, run):
    # v2 returns None for "0036"
    assert _latest(v3_db_dir, for_run=run) == "r36_b.txt"


def test_empty_run_means_any_run(v3_db_dir):
    assert _latest(v3_db_dir, for_run="", valid_only=False) == "r37_fail.txt"


def test_same_second_tie_break(tmp_path):
    """Executions within the same second: the last one written is the latest."""
    work_dir = str(tmp_path)
    for name in ("first.txt", "second.txt", "third.txt"):
        params = _params(work_dir, 36)
        params.outfile_name = name
        api.record_analysis_db(_described(params))
    stamps = sqlite3.connect(f"{work_dir}/lute.db").execute(
        "SELECT COUNT(DISTINCT timestamp) FROM executions"
    )
    if stamps.fetchone()[0] != 1:
        pytest.skip("Writes did not fall within the same second")
    assert _latest(work_dir, for_run=36) == "third.txt"
    assert _latest(work_dir, for_run=None) == "third.txt"


@pytest.mark.parametrize(
    "unit, valid_only, expected",
    [
        ("experiment=mfxtest;run=36", True, "r36_b.txt"),
        ("experiment=mfxtest;run=37", True, "r37_a.txt"),
        ("experiment=mfxtest;run=37", False, "r37_fail.txt"),
        ("experiment=mfxtest;run=38", True, None),
        ("experiment=other;run=36", True, None),
    ],
)
def test_for_unit(v3_db_dir, unit, valid_only, expected):
    value = _latest(v3_db_dir, for_run=None, for_unit=unit, valid_only=valid_only)
    assert value == expected


def test_filters_combine(v3_db_dir):
    unit37 = "experiment=mfxtest;run=37"
    assert _latest(v3_db_dir, for_run=37, for_unit=unit37) == "r37_a.txt"
    assert _latest(v3_db_dir, for_run=36, for_unit=unit37) is None


def test_for_group(tmp_path, v3_db_dir):
    db_path = str(tmp_path / "lute.db")
    shutil.copy(os.path.join(v3_db_dir, "lute.db"), db_path)
    con = open_db(db_path)
    with con:
        units = [get_or_create_unit(con, "mfxtest", run) for run in (36, 37)]
        create_group(con, "both", units)
        # Executions 2 (r36_b) and 4 (r37_a) were launched under the group
        con.executemany(
            "INSERT INTO execution_groups (execution_id, group_id) VALUES (?, 1)",
            [(2,), (4,)],
        )
    con.close()
    work_dir = str(tmp_path)

    # Execution 5 (r37_fail) is newer, but not launched under the group
    assert _latest(work_dir, for_run=None, for_group="both", valid_only=False) == (
        "r37_a.txt"
    )
    assert _latest(work_dir, for_run=36, for_group="both") == "r36_b.txt"
    assert _latest(work_dir, for_run=None, for_group="missing") is None


def test_run_env_read_at_call_time(v3_db_dir, monkeypatch):
    """`for_run` defaults to RUN when called, not when the module was imported."""
    monkeypatch.setenv("RUN", "36")
    assert _latest(v3_db_dir) == "r36_b.txt"
    monkeypatch.setenv("RUN", "37")
    assert _latest(v3_db_dir) == "r37_a.txt"
    monkeypatch.delenv("RUN")
    assert _latest(v3_db_dir, valid_only=False) == "r37_fail.txt"
