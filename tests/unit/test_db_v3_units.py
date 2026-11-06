import os
import shutil

import pytest

from lute.io._db.common_sqlite import DatabaseError
from lute.io._db.v3._units import (
    add_label_kind,
    add_to_group,
    add_unit_labels,
    create_group,
    find_units,
    get_group_units,
    get_or_create_unit,
    get_unit_labels,
    normalize_run,
    remove_from_group,
)
from lute.io._db.v3._versioning import open_db


@pytest.fixture
def con(tmp_path):
    con = open_db(str(tmp_path / "lute.db"))
    yield con
    con.close()


def _count(con, table):
    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


@pytest.mark.parametrize(
    "run, expected",
    [
        (36, "36"),
        ("36", "36"),
        ("0036", "36"),
        (" 36 ", "36"),
        (0, "0"),
        ("000", "0"),
        ("r36_test", "r36_test"),
    ],
)
def test_normalize_run(run, expected):
    assert normalize_run(run) == expected


@pytest.mark.parametrize("run", [None, "", "   "])
def test_normalize_run_empty_refused(run):
    with pytest.raises(DatabaseError, match="must have a run"):
        normalize_run(run)


def test_same_run_any_format_gives_one_unit(con):
    ids = {get_or_create_unit(con, "mfxtest", run) for run in (36, "36", "0036")}
    assert len(ids) == 1
    unit_id = ids.pop()
    key = con.execute("SELECT key FROM units WHERE id = ?", (unit_id,)).fetchone()[0]
    assert key == "experiment=mfxtest;run=36"
    assert get_unit_labels(con, unit_id) == {"experiment": ["mfxtest"], "run": ["36"]}
    assert _count(con, "units") == 1
    assert _count(con, "unit_labels") == 2


def test_different_runs_and_experiments_give_different_units(con):
    a = get_or_create_unit(con, "mfxtest", 36)
    b = get_or_create_unit(con, "mfxtest", 37)
    c = get_or_create_unit(con, "mfxother", 36)
    assert len({a, b, c}) == 3
    # Shared labels are stored once
    assert _count(con, "labels") == 4  # 2 experiments + 2 runs


@pytest.mark.parametrize("experiment, run", [("", 36), ("mfxtest", ""), ("x", None)])
def test_empty_experiment_or_run_refused(con, experiment, run):
    with pytest.raises(DatabaseError):
        get_or_create_unit(con, experiment, run)
    assert _count(con, "units") == 0


def test_helpers_do_not_commit(con):
    get_or_create_unit(con, "mfxtest", 36)
    con.rollback()
    assert _count(con, "units") == 0
    assert _count(con, "labels") == 0


def test_add_unit_labels(con):
    unit_id = get_or_create_unit(con, "mfxtest", 36)
    add_unit_labels(con, unit_id, {"elog_sample": "Lyso", "elog_tag": ["good", 2]})
    add_unit_labels(con, unit_id, {"elog_tag": "good"})  # Already there: kept once
    assert get_unit_labels(con, unit_id) == {
        "experiment": ["mfxtest"],
        "run": ["36"],
        "elog_sample": ["Lyso"],
        "elog_tag": ["2", "good"],
    }


@pytest.mark.parametrize(
    "labels, match",
    [
        ({"run": 37}, "identity"),
        ({"experiment": "other"}, "identity"),
        ({"no_such_kind": "x"}, "Unknown label kind"),
        ({"elog_sample": " "}, "Empty value"),
    ],
)
def test_add_unit_labels_refused(con, labels, match):
    unit_id = get_or_create_unit(con, "mfxtest", 36)
    with pytest.raises(DatabaseError, match=match):
        add_unit_labels(con, unit_id, labels)


def test_add_label_kind(con):
    unit_id = get_or_create_unit(con, "mfxtest", 36)
    add_label_kind(con, "detector")
    add_label_kind(con, "detector")  # Idempotent
    add_unit_labels(con, unit_id, {"detector": "jungfrau4M"})
    assert find_units(con, detector="jungfrau4M") == [unit_id]


@pytest.fixture
def labelled(con):
    """Runs 36-38 of mfxtest and run 36 of mfxother, with eLog labels."""
    units = {
        36: get_or_create_unit(con, "mfxtest", 36),
        37: get_or_create_unit(con, "mfxtest", 37),
        38: get_or_create_unit(con, "mfxtest", 38),
        "other": get_or_create_unit(con, "mfxother", 36),
    }
    add_unit_labels(con, units[36], {"elog_sample": "Lyso", "elog_tag": "good"})
    add_unit_labels(con, units[37], {"elog_sample": "Lyso"})
    add_unit_labels(con, units[38], {"elog_sample": "Thau", "elog_tag": "good"})
    add_unit_labels(con, units["other"], {"elog_sample": "Lyso"})
    return units


def test_find_units_by_labels(con, labelled):
    u = labelled
    assert find_units(con) == sorted(u.values())
    assert find_units(con, experiment="mfxtest") == [u[36], u[37], u[38]]
    assert find_units(con, elog_sample="Lyso") == [u[36], u[37], u["other"]]
    assert find_units(con, experiment="mfxtest", elog_sample="Lyso") == [u[36], u[37]]
    assert find_units(con, elog_sample="Lyso", elog_tag="good") == [u[36]]
    assert find_units(con, run="0036") == [u[36], u["other"]]
    assert find_units(con, experiment="mfxtest", run=36) == [u[36]]
    assert find_units(con, elog_sample="Nope") == []


def test_find_units_unknown_kind_refused(con, labelled):
    with pytest.raises(DatabaseError, match="Unknown label kind"):
        find_units(con, sample="Lyso")


def test_groups(con, labelled):
    u = labelled
    create_group(con, "lyso", find_units(con, experiment="mfxtest", elog_sample="Lyso"))
    assert get_group_units(con, "lyso") == [u[36], u[37]]

    add_to_group(con, "lyso", [u[37], u[38]])  # 37 already in: kept once
    assert get_group_units(con, "lyso") == [u[36], u[37], u[38]]

    remove_from_group(con, "lyso", [u[38]])
    assert get_group_units(con, "lyso") == [u[36], u[37]]

    create_group(con, "empty")
    assert get_group_units(con, "empty") == []


def test_group_errors(con, labelled):
    create_group(con, "lyso")
    with pytest.raises(DatabaseError, match="already exists"):
        create_group(con, "lyso")
    with pytest.raises(DatabaseError, match="No group"):
        add_to_group(con, "missing", [1])
    with pytest.raises(DatabaseError, match="No group"):
        get_group_units(con, "missing")


def test_changing_group_keeps_execution_units(tmp_path, v3_db_dir):
    db_path = str(tmp_path / "lute.db")
    shutil.copy(os.path.join(v3_db_dir, "lute.db"), db_path)
    con = open_db(db_path)
    u36 = get_or_create_unit(con, "mfxtest", 36)
    u37 = get_or_create_unit(con, "mfxtest", 37)
    # Link the fixture executions to their run's unit (step 5 will do this)
    con.execute(
        "INSERT INTO execution_units (execution_id, unit_id) "
        "SELECT e.id, CASE c.run WHEN '36' THEN ? ELSE ? END "
        "FROM executions e JOIN config c ON c.id = e.config_id",
        (u36, u37),
    )
    create_group(con, "both", [u36, u37])
    before = con.execute("SELECT * FROM execution_units ORDER BY 1, 2").fetchall()

    remove_from_group(con, "both", [u37])
    add_to_group(con, "both", [get_or_create_unit(con, "mfxtest", 38)])

    after = con.execute("SELECT * FROM execution_units ORDER BY 1, 2").fetchall()
    con.close()
    assert len(before) == 5
    assert after == before
