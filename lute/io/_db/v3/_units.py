"""Units, labels and groups for the v3 database specification.

A unit is the thing LUTE processes, currently one (experiment, run). Labels are
(kind, value) pairs attached to units, e.g. (elog_sample, Lyso). The identity
labels (experiment, run) make up the unit's key, e.g.
`experiment=mfx101624926;run=36`. Groups are named sets of units.

None of these functions commit. The caller owns the transaction, e.g. with
`with con:`, so a unit, its labels and the execution using it are written
together or not at all.

Functions:
    normalize_run(run) -> str: Canonical text form of a run number.

    get_or_create_unit(con, experiment, run) -> int: Unit id for a run.

    add_unit_labels(con, unit_id, labels) -> None: Attach metadata labels.

    find_units(con, **labels) -> List[int]: Units having all the given labels.

    create_group / add_to_group / remove_from_group / get_group_units: Manage
        named groups of units.
"""

from typing import List

__all__: List[str] = [
    "normalize_run",
    "get_or_create_unit",
    "get_unit_labels",
    "add_label_kind",
    "add_unit_labels",
    "find_units",
    "create_group",
    "add_to_group",
    "remove_from_group",
    "get_group_units",
]

import sqlite3
from typing import Dict, Iterable, Optional, Tuple, Union

from lute.io._db.common_sqlite import DatabaseError

from ._sqlite import LABEL_KINDS

# Identity kinds, in the order they appear in a unit key
IDENTITY_KINDS: Tuple[str, ...] = tuple(
    name for name, is_identity in LABEL_KINDS if is_identity
)

LabelValues = Union[str, int, Iterable[Union[str, int]]]


def normalize_run(run: Optional[Union[str, int]]) -> str:
    """Return the canonical text form of a run, so "0036", "36" and 36 match.

    Args:
        run (Optional[str | int]): The run as given by the user or the header.

    Returns:
        run (str): Numeric runs without leading zeros. Other values are kept
            as given (stripped of surrounding whitespace).

    Raises:
        DatabaseError: If the run is empty. Every execution must be for a run.
    """
    if run is None or str(run).strip() == "":
        raise DatabaseError("An execution must have a run, but the run is empty.")
    text: str = str(run).strip()
    if text.isdigit():
        return str(int(text))
    return text


def _normalize_value(kind: str, value: Union[str, int]) -> str:
    if kind == "run":
        return normalize_run(value)
    text: str = str(value).strip()
    if text == "":
        raise DatabaseError(f"Empty value for label kind {kind}.")
    return text


def _get_kind_id(con: sqlite3.Connection, kind: str) -> Tuple[int, bool]:
    row: Optional[Tuple[int, int]] = con.execute(
        "SELECT id, is_identity FROM label_kinds WHERE name = ?", (kind,)
    ).fetchone()
    if row is None:
        raise DatabaseError(
            f"Unknown label kind {kind}. Add it first with `add_label_kind`."
        )
    return row[0], bool(row[1])


def _get_or_create_label(con: sqlite3.Connection, kind: str, value: str) -> int:
    kind_id: int
    kind_id, _ = _get_kind_id(con, kind)
    con.execute(
        "INSERT OR IGNORE INTO labels (kind_id, value) VALUES (?, ?)", (kind_id, value)
    )
    return con.execute(
        "SELECT id FROM labels WHERE kind_id = ? AND value = ?", (kind_id, value)
    ).fetchone()[0]


def _attach_label(con: sqlite3.Connection, unit_id: int, label_id: int) -> None:
    con.execute(
        "INSERT OR IGNORE INTO unit_labels (unit_id, label_id) VALUES (?, ?)",
        (unit_id, label_id),
    )


def _unit_key(identity: Dict[str, str]) -> str:
    return ";".join(f"{kind}={identity[kind]}" for kind in IDENTITY_KINDS)


def get_or_create_unit(
    con: sqlite3.Connection, experiment: str, run: Union[str, int]
) -> int:
    """Return the id of the unit for (experiment, run), creating it if needed.

    The unit gets its identity labels (`experiment`, `run`). The run is
    normalized, so the same run in any format gives the same unit.

    Args:
        con (sqlite3.Connection): A connection to the database.

        experiment (str): The experiment, e.g. mfx101624926.

        run (str | int): The run, e.g. 36 or "0036".

    Returns:
        unit_id (int): The id in the `units` table.

    Raises:
        DatabaseError: If the experiment or the run is empty.
    """
    identity: Dict[str, str] = {
        "experiment": _normalize_value("experiment", experiment),
        "run": normalize_run(run),
    }
    key: str = _unit_key(identity)
    con.execute("INSERT OR IGNORE INTO units (key) VALUES (?)", (key,))
    row: Tuple[int] = con.execute(
        "SELECT id FROM units WHERE key = ?", (key,)
    ).fetchone()
    unit_id: int = row[0]
    for kind, value in identity.items():
        _attach_label(con, unit_id, _get_or_create_label(con, kind, value))
    return unit_id


def get_unit_labels(con: sqlite3.Connection, unit_id: int) -> Dict[str, List[str]]:
    """Return the labels of a unit as {kind: [values]}, values sorted."""
    labels: Dict[str, List[str]] = {}
    res: sqlite3.Cursor = con.execute(
        "SELECT k.name, l.value FROM unit_labels ul "
        "JOIN labels l ON l.id = ul.label_id "
        "JOIN label_kinds k ON k.id = l.kind_id "
        "WHERE ul.unit_id = ? ORDER BY k.id, l.value",
        (unit_id,),
    )
    for kind, value in res.fetchall():
        labels.setdefault(kind, []).append(value)
    return labels


def add_label_kind(con: sqlite3.Connection, name: str) -> int:
    """Add a metadata label kind (not an identity kind). Returns its id."""
    con.execute(
        "INSERT OR IGNORE INTO label_kinds (name, is_identity) VALUES (?, 0)", (name,)
    )
    return _get_kind_id(con, name)[0]


def add_unit_labels(
    con: sqlite3.Connection, unit_id: int, labels: Dict[str, LabelValues]
) -> None:
    """Attach metadata labels to a unit, e.g. {"elog_tag": ["good", "lyso"]}.

    Labels already attached are kept. Identity labels cannot be added: they are
    set by `get_or_create_unit` and define the unit.

    Args:
        con (sqlite3.Connection): A connection to the database.

        unit_id (int): The unit to label.

        labels (Dict[str, str | int | Iterable]): Kind to one value or several.

    Raises:
        DatabaseError: For an unknown or identity kind, or an empty value.
    """
    for kind, values in labels.items():
        is_identity: bool
        _, is_identity = _get_kind_id(con, kind)
        if is_identity:
            raise DatabaseError(
                f"{kind} is an identity label; it is set when the unit is created."
            )
        if isinstance(values, (str, int)):
            values = [values]
        for value in values:
            label_id: int = _get_or_create_label(
                con, kind, _normalize_value(kind, value)
            )
            _attach_label(con, unit_id, label_id)


def find_units(con: sqlite3.Connection, **labels: Union[str, int]) -> List[int]:
    """Return the ids of the units having all the given labels.

    Example: `find_units(con, experiment="mfx101624926", elog_sample="Lyso")`.
    With no labels, returns every unit.

    Args:
        con (sqlite3.Connection): A connection to the database.

        **labels (str | int): Kind=value pairs. All must match.

    Returns:
        unit_ids (List[int]): Sorted unit ids.
    """
    if not labels:
        return [r[0] for r in con.execute("SELECT id FROM units ORDER BY id")]
    conditions: List[str] = []
    params: List[str] = []
    for kind, value in labels.items():
        _get_kind_id(con, kind)  # Raise on unknown kinds instead of matching none
        conditions.append("(k.name = ? AND l.value = ?)")
        params.extend([kind, _normalize_value(kind, value)])
    query: str = (
        "SELECT ul.unit_id FROM unit_labels ul "
        "JOIN labels l ON l.id = ul.label_id "
        "JOIN label_kinds k ON k.id = l.kind_id "
        f"WHERE {' OR '.join(conditions)} "
        "GROUP BY ul.unit_id HAVING COUNT(*) = ? ORDER BY ul.unit_id"
    )
    res: sqlite3.Cursor = con.execute(query, [*params, len(labels)])
    return [r[0] for r in res.fetchall()]


def _get_group_id(con: sqlite3.Connection, name: str) -> int:
    row: Optional[Tuple[int]] = con.execute(
        "SELECT id FROM run_groups WHERE name = ?", (name,)
    ).fetchone()
    if row is None:
        raise DatabaseError(f"No group named {name}.")
    return row[0]


def create_group(
    con: sqlite3.Connection, name: str, unit_ids: Iterable[int] = ()
) -> int:
    """Create a named group of units. Returns its id.

    Raises:
        DatabaseError: If a group with this name already exists.
    """
    try:
        group_id: Optional[int] = con.execute(
            "INSERT INTO run_groups (name) VALUES (?)", (name,)
        ).lastrowid
    except sqlite3.IntegrityError as err:
        raise DatabaseError(f"Group {name} already exists.") from err
    assert group_id is not None
    add_to_group(con, name, unit_ids)
    return group_id


def add_to_group(con: sqlite3.Connection, name: str, unit_ids: Iterable[int]) -> None:
    """Add units to a group. Units already in it are kept."""
    group_id: int = _get_group_id(con, name)
    con.executemany(
        "INSERT OR IGNORE INTO group_members (group_id, unit_id) VALUES (?, ?)",
        [(group_id, unit_id) for unit_id in unit_ids],
    )


def remove_from_group(
    con: sqlite3.Connection, name: str, unit_ids: Iterable[int]
) -> None:
    """Remove units from a group.

    Past executions keep their units: `execution_units` is not changed.
    """
    group_id: int = _get_group_id(con, name)
    con.executemany(
        "DELETE FROM group_members WHERE group_id = ? AND unit_id = ?",
        [(group_id, unit_id) for unit_id in unit_ids],
    )


def get_group_units(con: sqlite3.Connection, name: str) -> List[int]:
    """Return the ids of the units currently in a group, sorted."""
    group_id: int = _get_group_id(con, name)
    res: sqlite3.Cursor = con.execute(
        "SELECT unit_id FROM group_members WHERE group_id = ? ORDER BY unit_id",
        (group_id,),
    )
    return [r[0] for r in res.fetchall()]
