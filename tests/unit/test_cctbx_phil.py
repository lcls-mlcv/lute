from typing import Any, Dict

import pytest
from pydantic import ValidationError

from lute.io.models.cctbx_phil import dict_to_phil
from lute.io.models.sfx_merge import ScaleCCTBXXFELParameters

ScalePhil = ScaleCCTBXXFELParameters.ScalePhilParameters

BASE: Dict[str, Any] = {
    "input": {"path": "/data/r0036/index"},
    "output": {"output_dir": "/data/r0036/scale"},
}


def test_dict_to_phil_nesting_lists_and_none() -> None:
    tree: Dict[str, Any] = {
        "input": {"path": ["/a", "/b"]},
        "filter": {"unit_cell": {"value": {"relative_length_tolerance": 0.03}}},
        "select": {"algorithm": None, "significance_filter": {"sigma": None}},
        "merging": {"merge_anomalous": True},
    }
    assert dict_to_phil(tree) == (
        "input {\n"
        "  path = /a\n"
        "  path = /b\n"
        "}\n"
        "filter {\n"
        "  unit_cell {\n"
        "    value {\n"
        "      relative_length_tolerance = 0.03\n"
        "    }\n"
        "  }\n"
        "}\n"
        "merging {\n"
        "  merge_anomalous = True\n"
        "}"
    )


def test_nested_yaml_renders_same_tree() -> None:
    phil: str = ScalePhil(
        **BASE,
        filter={"unit_cell": {"value": {"relative_length_tolerance": 0.03}}},
    ).to_phil()
    assert (
        "filter {\n  unit_cell {\n    value {\n      relative_length_tolerance = 0.03\n"
        in phil
    )
    # Unset optional fields and empty scopes are left out entirely
    assert "\nselect {" not in phil
    assert "\nstatistics {" not in phil
    assert "cluster" not in phil
    # Required values are always written
    assert "  d_min = 3.0\n" in phil
    assert "  save_experiments_and_reflections = True\n" in phil


def test_multi_run_input_path_is_repeated() -> None:
    phil: str = ScalePhil(
        **{**BASE, "input": {"path": ["/data/r0035/index", "/data/r0036/index"]}}
    ).to_phil()
    assert "  path = /data/r0035/index\n  path = /data/r0036/index\n" in phil


def test_unmodelled_parameters_pass_through() -> None:
    model = ScalePhil(
        **BASE,
        filter={"outlier": {"min_corr": -1.0, "mad_thresh": 5}},
        postrefinement={"enable": True},
    )
    phil: str = model.to_phil()
    assert "  outlier {\n    min_corr = -1.0\n    mad_thresh = 5\n  }" in phil
    assert "postrefinement {\n  enable = True\n}" in phil
    assert sorted(model.unmodelled_paths()) == [
        "filter.outlier.mad_thresh",
        "postrefinement",
    ]


def test_cluster_scope() -> None:
    phil: str = ScalePhil(
        **BASE,
        filter={
            "unit_cell": {
                "algorithm": "cluster",
                "cluster": {"covariance": {"file": "/cov.pickle", "mahalanobis": 4.0}},
            }
        },
    ).to_phil()
    assert (
        "    cluster {\n      covariance {\n        file = /cov.pickle\n"
        "        mahalanobis = 4.0\n      }\n    }" in phil
    )


def test_output_dir_defaults_to_work_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUTE_WORK_DIR", "/work")
    phil: str = ScalePhil(input={"path": "/data"}).to_phil()
    assert "  output_dir = /work\n" in phil


def test_wrong_type_still_rejected() -> None:
    with pytest.raises(ValidationError):
        ScalePhil(**BASE, filter={"outlier": {"min_corr": "not-a-number"}})
