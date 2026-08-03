"""Partition a LUTE workflow DAG YAML into a run-dependent and a
non-run-dependent subgraph, based on a per-node `run_dependent` key.

Used by `--tag` submissions (see `tagged_launch.py`): a workflow author marks
each DAG node `run_dependent: true` (default, if omitted - matches today's
implicit single-run behavior) or `run_dependent: false`. `--tag` submissions
resolve a set of runs from an eLog tag, submit the run-dependent subgraph
once per resolved run, and the non-run-dependent subgraph once, after the
run-dependent stage completes.

Only supports one contiguous run-dependent prefix feeding one contiguous
non-run-dependent suffix - the DAG/JobStep execution engine (a compiled
extension, see `maestro._maestro._maestro`) has no native concept of mixed
run-dependence within a single workflow run, so this module works entirely
at the raw YAML level, before `maestro.parser.load_lute_dag` ever sees the
file, and hands back two ordinary `!LUTE_DAG` documents for that unmodified
loader to parse as usual.
"""

from __future__ import annotations

import io
from typing import List, Optional, Tuple

import yaml

__all__ = ["DagPartitionError", "partition_workflow_yaml"]


class DagPartitionError(Exception):
    """Raised when a workflow DAG cannot be partitioned by run-dependence."""


_UNSUPPORTED_TAG_PREFIXES = ("!branch", "!param_sweep")


def _get_task_name(node: yaml.MappingNode) -> str:
    for key_node, val_node in node.value:
        if key_node.value == "task_name":
            return val_node.value
    return "<unknown>"


def _find_value_node(node: yaml.MappingNode, key: str) -> Optional[yaml.Node]:
    for key_node, val_node in node.value:
        if key_node.value == key:
            return val_node
    return None


def _is_run_dependent(node: yaml.MappingNode) -> bool:
    # No `run_dependent` key -> defaults to run-dependent, so unannotated
    # workflows behave exactly as they do today.
    val_node = _find_value_node(node, "run_dependent")
    if val_node is None:
        return True
    return val_node.value.strip().lower() not in ("false", "no", "0")


def _check_supported(node: yaml.MappingNode) -> None:
    if any(node.tag.startswith(p) for p in _UNSUPPORTED_TAG_PREFIXES):
        raise DagPartitionError(
            f"Task node '{_get_task_name(node)}' uses tag '{node.tag}', which "
            "is not supported in a --tag submission (v1 only supports plain "
            "run_dependent/non-run_dependent chains, not !branch_*/!param_sweep)."
        )


def _rebuild_next(
    pairs: List[Tuple[yaml.Node, yaml.Node]],
    new_next_children: List[yaml.MappingNode],
) -> List[Tuple[yaml.Node, yaml.Node]]:
    new_next_node = yaml.SequenceNode("tag:yaml.org,2002:seq", list(new_next_children))
    out: List[Tuple[yaml.Node, yaml.Node]] = []
    replaced = False
    for key_node, val_node in pairs:
        if key_node.value == "next":
            out.append((key_node, new_next_node))
            replaced = True
        else:
            out.append((key_node, val_node))
    if not replaced and new_next_children:
        out.append((yaml.ScalarNode("tag:yaml.org,2002:str", "next"), new_next_node))
    return out


def _walk(
    node: yaml.MappingNode,
    keep_dependent: bool,
    boundary_roots: List[yaml.MappingNode],
) -> yaml.MappingNode:
    """Rebuild `node` keeping only descendants matching `keep_dependent`.

    A child whose run-dependence differs from its parent is either a valid
    prefix->suffix boundary (dependent parent, non-dependent child -
    recorded into `boundary_roots` so the caller can attach it as a new root
    of the non-dependent subgraph) or an unsupported non-dependent->dependent
    edge (hard error).
    """
    _check_supported(node)
    next_node = _find_value_node(node, "next")
    kept_children: List[yaml.MappingNode] = []
    if isinstance(next_node, yaml.SequenceNode):
        for child in next_node.value:
            if not isinstance(child, yaml.MappingNode):
                raise DagPartitionError(
                    "--tag submission requires every DAG node to be a plain "
                    "mapping (task_name/slurm_params/next/run_dependent); "
                    f"encountered an unsupported node type under "
                    f"'{_get_task_name(node)}'."
                )
            _check_supported(child)
            child_dependent = _is_run_dependent(child)
            if child_dependent == keep_dependent:
                kept_children.append(_walk(child, keep_dependent, boundary_roots))
            elif keep_dependent and not child_dependent:
                boundary_roots.append(child)
            else:
                raise DagPartitionError(
                    f"Task '{_get_task_name(node)}' is marked run_dependent: "
                    f"false but has a run_dependent: true child "
                    f"('{_get_task_name(child)}'). v1 only supports one "
                    "run-dependent prefix feeding one non-run-dependent "
                    "suffix - split this DAG by hand for this topology."
                )
    new_pairs = _rebuild_next(node.value, kept_children)
    return yaml.MappingNode(node.tag, new_pairs, flow_style=node.flow_style)


def _roots_of(document: yaml.Node) -> List[yaml.MappingNode]:
    if isinstance(document, yaml.MappingNode):
        return [document]
    if isinstance(document, yaml.SequenceNode):
        for item in document.value:
            if not isinstance(item, yaml.MappingNode):
                raise DagPartitionError(
                    "Top-level workflow document must be a !LUTE_DAG mapping "
                    "or a list of them."
                )
        return list(document.value)
    raise DagPartitionError("Unrecognized top-level workflow document structure.")


def _serialize(node_list: List[yaml.MappingNode]) -> Optional[str]:
    if not node_list:
        return None
    top: yaml.Node
    if len(node_list) == 1:
        top = node_list[0]
    else:
        top = yaml.SequenceNode("tag:yaml.org,2002:seq", list(node_list))
    top.tag = "!LUTE_DAG"
    buf = io.StringIO()
    yaml.serialize(top, buf, Dumper=yaml.SafeDumper)
    return buf.getvalue()


def partition_workflow_yaml(workflow_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Split a LUTE workflow DAG YAML file into a run-dependent subgraph and
    a non-run-dependent subgraph.

    Returns (dependent_yaml, non_dependent_yaml) as YAML strings; either is
    None if that subgraph is empty. Both, when present, are ordinary
    `!LUTE_DAG`-tagged documents loadable via the existing, unmodified
    `maestro.parser.load_lute_dag_str`.

    Raises DagPartitionError if the DAG doesn't fit the "one run-dependent
    prefix feeding one non-run-dependent suffix" shape, or uses
    !branch_*/!param_sweep tags (unsupported in v1).
    """
    with open(workflow_path, "r") as f:
        document = yaml.compose(f, Loader=yaml.SafeLoader)

    if document is None:
        raise DagPartitionError(f"Workflow file '{workflow_path}' is empty.")

    roots = _roots_of(document)
    boundary_roots: List[yaml.MappingNode] = []

    dep_roots: List[yaml.MappingNode] = []
    non_dep_top_roots: List[yaml.MappingNode] = []
    for root in roots:
        _check_supported(root)
        if _is_run_dependent(root):
            dep_roots.append(_walk(root, True, boundary_roots))
        else:
            non_dep_top_roots.append(_walk(root, False, boundary_roots))

    # Snapshot before the second walk: keep_dependent=False below can never
    # append new boundary roots (that only happens when keep_dependent=True),
    # but snapshotting avoids relying on that invariant while iterating.
    non_dep_roots = non_dep_top_roots + [
        _walk(b, False, boundary_roots) for b in list(boundary_roots)
    ]

    return _serialize(dep_roots), _serialize(non_dep_roots)
