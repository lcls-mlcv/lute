"""Config-time validation of the IndexGLINT task parameters.

These validators decide whether a GLINT run does what its YAML says. Each one guards a failure that
would otherwise be silent: a job that "succeeds" and hands StreamFileConcatenator nothing usable.
The last group checks the command line the Task would exec, because that is where the two GLINT
programs (glint.glint_cli, glint_xtc.py) differ.
"""

import os
import tempfile
from typing import Any, Dict, List

import pytest

from lute.io.models.base import AnalysisHeader
from lute.io.models.glint_index import (
    GLINT_CLI,
    GLINT_ROOT_DEFAULT,
    GLINT_XTC_RELPATH,
    IndexGLINTParameters,
    glint_root,
)

HEADER = AnalysisHeader(experiment="test_exp", run=1, work_dir=tempfile.gettempdir())
XTC: Dict[str, Any] = dict(exp="mfxx49820", run=16, zdist=0.1027, out="o.stream")


def P(**kw: Any) -> IndexGLINTParameters:
    return IndexGLINTParameters(lute_config=HEADER, **kw)


def bad(**kw: Any) -> str:
    """Assert the config is rejected and return the message, so a test can check which rule fired."""
    with pytest.raises(Exception) as e:
        P(**kw)
    return str(e.value)


def argv(params: IndexGLINTParameters) -> List[str]:
    """The argument list ThirdPartyTask would exec, built by its own `_pre_run`.

    `Task.__init__` arms a timer, pins CPU affinity and touches stdin, none of which a unit test
    wants, so the instance is assembled by hand with the four attributes `_pre_run` reads.
    """
    from lute.tasks.task import ThirdPartyTask

    task = ThirdPartyTask.__new__(ThirdPartyTask)
    task._task_parameters = params
    task._cmd = params.executable
    task._args_list = [task._cmd]
    task._template_context = {}
    task._pre_run()
    return task._args_list


# The interpreter comes from the managed Task's environment; the program from the frame source
def test_executable_defaults_to_the_environment_python() -> None:
    assert P(peaks="p.stream", out="o.stream").executable == "python"
    assert P(
        peaks="p.stream", out="o.stream", executable="/env/bin/python"
    ).executable == ("/env/bin/python")


def test_program_follows_the_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LUTE_GLINT_ROOT", raising=False)
    assert P(peaks="p.stream", out="o.stream").program == GLINT_CLI
    assert P(images="i.cxi", out="o.stream").program == GLINT_CLI
    assert P(**XTC).program == f"{GLINT_ROOT_DEFAULT}/{GLINT_XTC_RELPATH}"
    # a value in the YAML is overridden, never forwarded
    assert P(**XTC, program="something_else.py").program.endswith(GLINT_XTC_RELPATH)


def test_glint_root_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUTE_GLINT_ROOT", "/my/checkout")
    assert glint_root() == "/my/checkout"
    assert P(**XTC).program == f"/my/checkout/{GLINT_XTC_RELPATH}"
    monkeypatch.delenv("LUTE_GLINT_ROOT")
    assert glint_root() == GLINT_ROOT_DEFAULT


# Exactly one frame source
def test_no_source_rejected() -> None:
    assert "frame source is required" in bad(out="o.stream")


@pytest.mark.parametrize(
    "a,b",
    [
        ({"peaks": "p.stream"}, {"images": "i.cxi"}),
        ({"peaks": "p.stream"}, {"exp": "mfxx49820", "run": 16, "zdist": 0.1}),
        ({"images": "i.cxi"}, {"exp": "mfxx49820", "run": 16, "zdist": 0.1}),
    ],
)
def test_two_sources_rejected(a: Dict[str, Any], b: Dict[str, Any]) -> None:
    assert "exactly ONE frame source" in bad(out="o.stream", **a, **b)


@pytest.mark.parametrize(
    "src",
    [
        {"peaks": "p.stream"},
        {"images": "i.cxi"},
        {"exp": "mfxx49820", "run": 16, "zdist": 0.1027},
    ],
)
def test_one_source_accepted(src: Dict[str, Any]) -> None:
    assert P(out="o.stream", **src) is not None


# Raw xtc needs a run and a refined distance
@pytest.mark.parametrize("missing", ["run", "zdist"])
def test_xtc_requires_run_and_zdist(missing: str) -> None:
    kw = dict(XTC)
    kw.pop(missing)
    assert f"`{missing}` is required with `exp`" in bad(**kw)


# Peak-finder validity is per source
def test_peakfinder_defaults_per_source() -> None:
    assert P(**XTC).peakfinder == "v4"
    assert P(images="i.cxi", out="o.stream").peakfinder == "stored"


@pytest.mark.parametrize("pf", ["v4", "pf8", "pf8-panel"])
def test_peakfinder_valid_on_xtc(pf: str) -> None:
    assert P(peakfinder=pf, **XTC).peakfinder == pf


@pytest.mark.parametrize("pf", ["stored", "pf9"])
def test_peakfinder_rejected_on_xtc(pf: str) -> None:
    assert "not available on the `exp`" in bad(peakfinder=pf, **XTC)


@pytest.mark.parametrize("pf", ["pf8", "pf8-panel"])
def test_pf8_rejected_on_images(pf: str) -> None:
    assert "only wired on the `exp`" in bad(
        peakfinder=pf, images="i.cxi", out="o.stream"
    )


# Knobs that only the raw-xtc program reads (glint.glint_cli has no such flags)
XTC_ONLY = [
    ("det", "MfxEndstation.0:Epix10ka2M.0"),
    ("psana", "1"),
    ("calib_dir", "/tmp/calib"),
    ("min_pix", 4),
    ("son_min", 12.0),
    ("thr_high", 8.0),
    ("thr_low", 4.0),
    ("pf8_min_snr", 12.0),
    ("max_events", 500),
]


@pytest.mark.parametrize("field,value", XTC_ONLY)
def test_xtc_only_knobs_rejected_elsewhere(field: str, value: Any) -> None:
    assert "applies only to the `exp`" in bad(
        peaks="p.stream", out="o.stream", **{field: value}
    )
    assert "applies only to the `exp`" in bad(
        images="i.cxi", out="o.stream", **{field: value}
    )


@pytest.mark.parametrize("field,value", XTC_ONLY)
def test_xtc_only_knobs_accepted_on_xtc(field: str, value: Any) -> None:
    assert getattr(P(**XTC, **{field: value}), field) == value


# Knobs that only glint.glint_cli reads (glint_xtc.py would exit at argparse)
@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", "sparse"),
        ("device", "cpu"),
        ("lattice", "tPc"),
        ("tofile", "sol.txt"),
        ("cascade", "/opt/ffbidx"),
        ("top_peaks", 100),
        ("image_dir", "/data"),
        ("event_axis", "event"),
        ("n", 10),
    ],
)
def test_cli_only_knobs_rejected_on_xtc(field: str, value: Any) -> None:
    msg = bad(**XTC, **{field: value})
    assert "not options of the raw-xtc program" in msg
    assert field in msg


def test_n_on_xtc_points_at_max_events() -> None:
    assert "`max_events`" in bad(**XTC, n=10)


def test_gate_rejected_on_xtc() -> None:
    assert "has no gate" in bad(**XTC, gate="strict")


def test_wavelength_is_accepted_on_every_source() -> None:
    assert P(peaks="p.stream", out="o.stream", wavelength=1.29) is not None
    assert P(images="i.cxi", out="o.stream", wavelength=1.29) is not None
    assert P(**XTC, wavelength=1.29) is not None


def test_top_peaks_only_with_images() -> None:
    assert "applies only to `images`" in bad(
        peaks="p.stream", out="o.stream", top_peaks=100
    )
    assert P(images="i.cxi", out="o.stream", top_peaks=100).top_peaks == 100


# Outputs the downstream Tasks depend on
def test_out_is_required() -> None:
    assert "`out` is required" in bad(peaks="p.stream", out="")


def test_integrate_with_peaks_needs_image_dir() -> None:
    assert "requires `image_dir`" in bad(
        peaks="p.stream", out="o.stream", integrate=True
    )
    assert (
        P(peaks="p.stream", out="o.stream", integrate=True, image_dir="/data")
        is not None
    )


def test_legacy_fromfile_is_renamed_not_dropped() -> None:
    assert P(peaks="p.stream", out="o.stream", fromfile="sol.txt").tofile == "sol.txt"
    assert "not both" in bad(
        peaks="p.stream", out="o.stream", fromfile="a.txt", tofile="b.txt"
    )


@pytest.mark.parametrize(
    "field,flag,values",
    [
        ("event_axis", "event-axis", ("auto", "event", "panel")),
        ("bg_mode", "bg-mode", ("clipmean", "median", "mean")),
        ("gate", "gate", ("none", "strict")),
    ],
)
def test_enum_fields_render_as_cli_flags(field: str, flag: str, values: tuple) -> None:
    f = IndexGLINTParameters.__fields__[field]
    assert f.field_info.extra["rename_param"] == flag
    assert f.field_info.extra["flag_type"] == "--"
    base = dict(peaks="p.stream", out="o.stream")
    assert getattr(P(**base), field) is None  # unset defers to GLINT's own default
    for v in values:
        assert getattr(P(**dict(base, **{field: v})), field) == v
    with pytest.raises(Exception):  # a typo must not silently mean "default"
        P(**dict(base, **{field: "typo"}))


def test_floor_gate_requires_floor() -> None:
    base = dict(peaks="p.stream", out="o.stream")
    assert P(**base, gate="floor", floor="cxidb17").floor == "cxidb17"
    assert P(images="i.cxi", out="o.stream", gate="floor", floor="1.5,0.2").floor == (
        "1.5,0.2"
    )
    assert "`gate: floor` requires `floor`" in bad(**base, gate="floor")
    assert "`floor` is used only with `gate: floor`" in bad(**base, floor="cxidb17")
    assert "`floor` is used only with `gate: floor`" in bad(
        **base, gate="strict", floor="cxidb17"
    )


# The command line, as ThirdPartyTask builds it
def test_argv_images_route_is_the_launcher_command_line() -> None:
    """The r51 test config, minus the launcher: python -m glint.glint_cli, then exactly the flags the
    launcher script used to receive (the three defaulted options included)."""
    p = P(
        images="i.list",
        peakfinder="stored",
        geom="g.geom",
        out="o.stream",
        tofile="s.sol",
        lattice="tPc",
        nbest=3,
        gate="strict",
        device="auto",
        cell="79.17 79.17 37.96 90.00 90.00 90.00",
    )
    assert argv(p) == [
        "python",
        "-m",
        "glint.glint_cli",
        "--images",
        "i.list",
        "--geom",
        "g.geom",
        "--peakfinder",
        "stored",
        "--out",
        "o.stream",
        "--cell",
        "79.17",
        "79.17",
        "37.96",
        "90.00",
        "90.00",
        "90.00",
        "--mode",
        "auto",
        "--nbest",
        "3",
        "--min-peaks",
        "6",
        "--device",
        "auto",
        "--tofile",
        "s.sol",
        "--lattice",
        "tPc",
        "--int-tol",
        "0.002",
        "--gate",
        "strict",
    ]


def test_argv_xtc_route_runs_the_reader_with_its_own_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LUTE_GLINT_ROOT", raising=False)
    a = argv(P(**XTC, det="jungfrau", peakfinder="pf8", max_events=200))
    assert a[:2] == ["python", f"{GLINT_ROOT_DEFAULT}/{GLINT_XTC_RELPATH}"]
    for flag in (
        "--exp",
        "--run",
        "--det",
        "--zdist",
        "--peakfinder",
        "--max-events",
        "--out",
    ):
        assert flag in a
    for flag in ("--mode", "--device", "--lattice", "-N", "--gate", "--tofile"):
        assert flag not in a


def test_argv_executable_override_is_the_first_entry() -> None:
    a = argv(P(peaks="p.stream", out="o.stream", executable="/env/bin/python"))
    assert a[:3] == ["/env/bin/python", "-m", "glint.glint_cli"]


# The managed Task's environment (lute/managed_tasks.py: shell_source psconda.sh + this)
def test_managed_task_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from lute.tasks.util.environment import (
        GLINT_ANA_ENV,
        GLINT_ANA_ENV_ALT,
        GLINT_CONDA1_ENVS,
        setup_glint_env,
        setup_glint_env_ana59,
    )

    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("LUTE_GLINT_ROOT", "/my/checkout")
    env = setup_glint_env()
    prefix = f"{GLINT_CONDA1_ENVS}/{GLINT_ANA_ENV}"
    assert env["PATH"] == f"{prefix}/bin:/usr/bin"
    assert env["PYTHONPATH"] == "/my/checkout"  # the checkout alone, nothing inherited
    assert env["CUDA_PATH"] == env["CONDA_PREFIX"] == prefix
    assert env["CONDA_DEFAULT_ENV"] == GLINT_ANA_ENV
    alt = setup_glint_env_ana59()
    assert alt["CONDA_DEFAULT_ENV"] == GLINT_ANA_ENV_ALT
    assert alt["PATH"].startswith(f"{GLINT_CONDA1_ENVS}/{GLINT_ANA_ENV_ALT}/bin:")
    monkeypatch.delenv("LUTE_GLINT_ROOT")
    assert setup_glint_env()["PYTHONPATH"] == GLINT_ROOT_DEFAULT
    assert "LUTE_GLINT_ROOT" not in os.environ
