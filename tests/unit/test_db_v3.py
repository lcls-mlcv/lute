import multiprocessing
import os
import shutil
import sqlite3
import subprocess
import sys

import pytest

import lute.io._db.v3._versioning as versioning
from lute.io._db.common_sqlite import DatabaseError, get_user_version
from lute.io._db.v3._sqlite import create_tables
from lute.io._db.v3._versioning import open_db
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


# Versioning and migrations (step 2)


def _user_version(db_path):
    con = sqlite3.connect(db_path)
    try:
        return get_user_version(con)
    finally:
        con.close()


def _tables(db_path):
    con = sqlite3.connect(db_path)
    try:
        res = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return {r[0] for r in res.fetchall()}
    finally:
        con.close()


def test_open_db_new_file_is_latest(tmp_path):
    db_path = str(tmp_path / "lute.db")
    con = open_db(db_path)
    n_triggers = con.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger'"
    ).fetchone()[0]
    n_base_schema = con.execute("SELECT COUNT(*) FROM base_schema").fetchone()[0]
    con.close()

    assert _user_version(db_path) == versioning.latest_user_version() == 300
    assert {"executions", "config", "parameters", "results"} <= _tables(db_path)
    assert n_triggers == 6
    assert n_base_schema == 2


def test_open_db_twice_is_noop(tmp_path):
    db_path = str(tmp_path / "lute.db")
    open_db(db_path).close()
    open_db(db_path).close()
    assert _user_version(db_path) == 300
    assert not os.path.exists(f"{db_path}.bak-v300")


def test_fixture_written_with_v3_is_latest(v3_db_dir):
    assert _user_version(os.path.join(v3_db_dir, "lute.db")) == 300


def test_migration_applies_with_backup(tmp_path, v3_db_dir, monkeypatch):
    db_path = str(tmp_path / "lute.db")
    shutil.copy(os.path.join(v3_db_dir, "lute.db"), db_path)

    def add_table(con):
        con.execute("CREATE TABLE new_table (id INTEGER PRIMARY KEY)")

    monkeypatch.setattr(
        versioning, "MIGRATIONS", versioning.MIGRATIONS + [(300, 301, add_table)]
    )
    open_db(db_path).close()

    assert _user_version(db_path) == 301
    assert "new_table" in _tables(db_path)
    backup = f"{db_path}.bak-v300"
    assert _user_version(backup) == 300
    assert "new_table" not in _tables(backup)


def test_failed_migration_leaves_file_intact(tmp_path, v3_db_dir, monkeypatch):
    db_path = str(tmp_path / "lute.db")
    shutil.copy(os.path.join(v3_db_dir, "lute.db"), db_path)

    def broken(con):
        con.execute("CREATE TABLE half_done (id INTEGER PRIMARY KEY)")
        con.execute("UPDATE results SET valid_flag = 0")
        raise RuntimeError("migration failed")

    monkeypatch.setattr(
        versioning, "MIGRATIONS", versioning.MIGRATIONS + [(300, 301, broken)]
    )
    with pytest.raises(RuntimeError):
        open_db(db_path)

    assert _user_version(db_path) == 300
    assert "half_done" not in _tables(db_path)
    con = sqlite3.connect(db_path)
    n_valid = con.execute("SELECT SUM(valid_flag) FROM results").fetchone()[0]
    con.close()
    assert n_valid == 4  # 5 executions, 1 failed


def test_missing_migration_raises(tmp_path, monkeypatch):
    db_path = str(tmp_path / "lute.db")
    open_db(db_path).close()
    monkeypatch.setattr(
        versioning,
        "MIGRATIONS",
        versioning.MIGRATIONS + [(301, 302, lambda con: None)],
    )
    with pytest.raises(DatabaseError, match="No migration"):
        open_db(db_path)
    assert _user_version(db_path) == 300


def _open_after_barrier(db_path, barrier, results):
    barrier.wait()
    try:
        open_db(db_path).close()
        results.put("ok")
    except Exception as err:  # Report to the parent instead of dying silently
        results.put(repr(err))


def test_concurrent_openers_migrate_once(tmp_path):
    db_path = str(tmp_path / "lute.db")
    n_procs = 6
    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(n_procs)
    results = ctx.Queue()
    procs = [
        ctx.Process(target=_open_after_barrier, args=(db_path, barrier, results))
        for _ in range(n_procs)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=120)

    assert [results.get(timeout=5) for _ in procs] == ["ok"] * n_procs
    assert _user_version(db_path) == 300
    con = sqlite3.connect(db_path)
    assert con.execute("SELECT COUNT(*) FROM base_schema").fetchone()[0] == 2
    con.close()


def test_v3_refuses_v2_file(v2_db_dir):
    db_path = os.path.join(v2_db_dir, "lute.db")
    with pytest.raises(DatabaseError, match="LUTE_DB_SPEC_VERSION=2"):
        open_db(db_path)
    assert _user_version(db_path) == 0


def test_v3_api_refuses_v2_file(v2_db_dir):
    from lute.io._db.v3.api import read_latest_db_entry

    with pytest.raises(DatabaseError):
        read_latest_db_entry(v2_db_dir, "TestWriteOutput", "outfile_name")


def test_v3_refuses_newer_file(tmp_path):
    db_path = str(tmp_path / "lute.db")
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA user_version = 999")
    con.close()
    with pytest.raises(DatabaseError, match="newer"):
        open_db(db_path)


def test_v2_refuses_v3_file(v3_db_dir):
    from lute.io._db.v2.api import read_latest_db_entry

    with pytest.raises(DatabaseError, match="LUTE_DB_SPEC_VERSION=3"):
        read_latest_db_entry(v3_db_dir, "TestWriteOutput", "outfile_name")


# Labels, units and groups tables (step 3)

NEW_TABLES = {
    "label_kinds",
    "labels",
    "units",
    "unit_labels",
    "run_groups",
    "group_members",
    "execution_units",
    "execution_groups",
}


@pytest.fixture
def v3_con(tmp_path, v3_db_dir):
    """Connection to a copy of the v3 fixture (5 executions), FKs enforced.

    The write path created unit 1 (run 36, labels 1-2) and linked execution 1
    to it. Add a label, a group and a group link to build on.
    """
    db_path = str(tmp_path / "lute.db")
    shutil.copy(os.path.join(v3_db_dir, "lute.db"), db_path)
    con = open_db(db_path)
    con.execute(
        "INSERT INTO labels (kind_id, value) "
        "SELECT id, 'Lyso' FROM label_kinds WHERE name = 'elog_sample'"
    )
    con.execute(
        "INSERT INTO unit_labels (unit_id, label_id) "
        "SELECT 1, id FROM labels WHERE value = 'Lyso'"
    )
    con.execute("INSERT INTO run_groups (name) VALUES ('lyso')")
    con.execute("INSERT INTO group_members (group_id, unit_id) VALUES (1, 1)")
    con.execute("INSERT INTO execution_groups (execution_id, group_id) VALUES (1, 1)")
    con.commit()
    yield con
    con.close()


def test_v3_new_tables_and_indices_exist(tmp_path):
    db_path = str(tmp_path / "lute.db")
    con = open_db(db_path)
    kinds = con.execute(
        "SELECT name, is_identity FROM label_kinds ORDER BY id"
    ).fetchall()
    indices = {
        r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='index'")
    }
    con.close()

    assert NEW_TABLES <= _tables(db_path)
    assert kinds == [
        ("experiment", 1),
        ("run", 1),
        ("elog_sample", 0),
        ("elog_tag", 0),
    ]
    assert {
        "parameters_index_0",
        "execution_units_index_0",
        "unit_labels_index_0",
        "group_members_index_0",
    } <= indices


def test_v3_valid_rows_accepted(v3_con):
    linked = v3_con.execute(
        "SELECT u.key, l.value, g.name FROM execution_units eu "
        "JOIN units u ON u.id = eu.unit_id "
        "JOIN unit_labels ul ON ul.unit_id = u.id "
        "JOIN labels l ON l.id = ul.label_id "
        "JOIN execution_groups eg ON eg.execution_id = eu.execution_id "
        "JOIN run_groups g ON g.id = eg.group_id "
        "WHERE l.value = 'Lyso'"
    ).fetchall()
    assert linked == [("experiment=mfxtest;run=36", "Lyso", "lyso")]


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO label_kinds (name) VALUES ('run')",
        "INSERT INTO labels (kind_id, value) VALUES (3, 'Lyso')",
        "INSERT INTO units (key) VALUES ('experiment=mfxtest;run=36')",
        "INSERT INTO unit_labels (unit_id, label_id) VALUES (1, 1)",
        "INSERT INTO run_groups (name) VALUES ('lyso')",
        "INSERT INTO group_members (group_id, unit_id) VALUES (1, 1)",
        "INSERT INTO execution_units (execution_id, unit_id) VALUES (1, 1)",
        # An execution is launched under at most one group
        "INSERT INTO execution_groups (execution_id, group_id) VALUES (1, 1)",
    ],
)
def test_v3_duplicates_refused(v3_con, statement):
    with pytest.raises(sqlite3.IntegrityError):
        v3_con.execute(statement)


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO labels (kind_id, value) VALUES (99, 'x')",
        "INSERT INTO unit_labels (unit_id, label_id) VALUES (99, 1)",
        "INSERT INTO unit_labels (unit_id, label_id) VALUES (1, 99)",
        "INSERT INTO group_members (group_id, unit_id) VALUES (99, 1)",
        "INSERT INTO group_members (group_id, unit_id) VALUES (1, 99)",
        "INSERT INTO execution_units (execution_id, unit_id) VALUES (99, 1)",
        "INSERT INTO execution_units (execution_id, unit_id) VALUES (2, 99)",
        "INSERT INTO execution_groups (execution_id, group_id) VALUES (99, 1)",
        "INSERT INTO execution_groups (execution_id, group_id) VALUES (2, 99)",
        # A unit still referenced by executions cannot be deleted
        "DELETE FROM units WHERE id = 1",
    ],
)
def test_v3_foreign_keys_enforced(v3_con, statement):
    with pytest.raises(sqlite3.IntegrityError):
        v3_con.execute(statement)


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO units (key) VALUES (NULL)",
        "INSERT INTO labels (kind_id, value) VALUES (1, NULL)",
        "INSERT INTO run_groups (name) VALUES (NULL)",
        "INSERT INTO label_kinds (name, is_identity) VALUES ('x', 2)",
    ],
)
def test_v3_not_null_and_checks(v3_con, statement):
    with pytest.raises(sqlite3.IntegrityError):
        v3_con.execute(statement)
