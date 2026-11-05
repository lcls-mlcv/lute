import os
import sqlite3
import subprocess
import sys

import pytest

from lute.io._db.v3._sqlite import create_tables
from lute.io.db import import_function

from test_db_v2 import BASELINE_LATEST


def test_create_tables_v3(tmp_path):
    con = sqlite3.connect(str(tmp_path / "lute.db"))
    create_tables(con)

    res = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in res.fetchall()]
    assert "executions" in tables
    assert "tasks" in tables
    assert "parameters" in tables
    assert "results" in tables
    con.close()


@pytest.mark.parametrize("for_run, valid_only, param, expected", BASELINE_LATEST)
def test_read_latest_db_entry_v3(v3_db_dir, for_run, valid_only, param, expected):
    from lute.io._db.v3.api import read_latest_db_entry

    value = read_latest_db_entry(
        v3_db_dir,
        "TestWriteOutput",
        param,
        valid_only=valid_only,
        for_run=for_run,
    )
    assert value == expected


def test_read_latest_db_entry_v3_filters_task(v3_db_dir):
    from lute.io._db.v3.api import read_latest_db_entry

    assert read_latest_db_entry(v3_db_dir, "Test", "str_var", for_run=36) == "s36"
    assert read_latest_db_entry(v3_db_dir, "Test", "str_var", for_run=37) is None


def test_import_function_v3():
    func = import_function("read_latest_db_entry", 3)
    assert func.__module__ == "lute.io._db.v3.api"


def test_env_var_selects_v3():
    """`LUTE_DB_SPEC_VERSION` is read at import, so check it in a fresh process."""
    code = (
        "from lute.io.db import LUTE_DB_SPEC_VERSION, import_function;"
        "print(LUTE_DB_SPEC_VERSION,"
        " import_function('update_analysis_db', LUTE_DB_SPEC_VERSION).__module__)"
    )
    env = dict(os.environ, LUTE_DB_SPEC_VERSION="3")
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert out[-2:] == ["3", "lute.io._db.v3.api"]
