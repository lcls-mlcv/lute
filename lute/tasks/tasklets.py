"""Tasklet functions to be run before or after a Task.

"Tasklets" are simple processing steps which are intended to be run by an
Executor after the main Task has finished. These functions help provide a way to
handle some common operations which may need to be performed, such as examining
an output text file, or doing simple file conversions. These operations can be
also be imported for use in first-party Tasks. In particular for
`ThirdPartyTask`s, it is not easy to append operations to a Task which is why
the Executor mechanism is provided.

Functions:
    git_clone(repo: str, location: str, permissions: str): Clone a git repo.

    clone_smalldata(producer_location: str): Clone smalldata_tools based on the
        location of the producer file.

    concat_files(location: str, in_files_glob: str, out_file: str): Concatenate
        a group of files into a single output file.

    grep(match_str: str, in_file: str) -> str | List[str]: grep for text in a
        specific file. Returns the results.

    indexamajig_summary_indexing_rate(stream_file: str) -> Dict[str, str]: Parse
        an output stream file to determine indexed patterns/indexing rate.

    compare_hkl_fom_summary(shell_file: str, figure_display_name: str) ->
        Tuple[Dict[str, str], Optional[ElogSummaryPlots]]: Extract the figure of
        merit and produce a plot of figure of merit/resolution ring.

    setup_dimple_uglymol(final_mtz: str, experiment: str, density_display_name: str
        ) -> None: Set's up the javascript and HTML files needed to display electron
        density in the eLog using UglyMol and the output from dimple.

    solvent_scatter_summary(output_dir: str, figure_display_name: str) ->
        Tuple[Dict[str, str], Optional[ElogSummaryPlots]]: Process SolventScatter
        results and create summary plots for elog display.

Usage:
    As tasklets are just functions they can be imported and used within Task
    code normally if needed.

    However, tasklets can also be managed through the Executor in a similar way
    to environment changes. E.g., to add a tasklet to an Executor instance one
    would:

    # First create Executor instance as normal to run the Task
    MyTaskRunner: Executor = Executor("RunMyTask")
    #MyTaskRunner.update_environment(...) # if needed
    MyTaskRunner.add_tasklet(
        tasklet, args, when="before", set_result=False, set_summary=False
    )

    A special substitution syntax can be used in args if specific values
    from a `TaskParameters` object will be needed to run the Tasklet:
        args=("{{ param_to_sub }}", ...)
"""

__all__ = [
    "grep",
    "git_clone",
    "indexamajig_summary_indexing_rate",
    "compare_hkl_fom_summary",
    "setup_dimple_uglymol",
    "tjump_analysis_summary",
]
__author__ = "Gabriel Dorlhiac"

import logging
import os
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

from lute.tasks.dataclasses import ElogSummaryPlots
from lute.execution.logging import get_logger

if __debug__:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

logger: logging.Logger = get_logger(__name__, is_task=False)


def concat_files(location: str, in_files_glob: str, out_file: str) -> None:
    """Concatenate a series of files into a single output file.

    Args:
        location (str): Path to the files to concatenate.

        in_files_glob (str): A glob to match a series of files at the specified
            path. These will all be concatenated.

        out_file (str): Name of the concatenated output.
    """
    from pathlib import Path
    from typing import BinaryIO

    in_file_path: Path = Path(f"{location}")
    in_file_list: List[Path] = list(in_file_path.rglob(f"{in_files_glob}"))

    wf: BinaryIO
    with open(out_file, "wb") as wf:
        for in_file in in_file_list:
            rf: BinaryIO
            with open(in_file, "rb") as rf:
                shutil.copyfileobj(rf, wf)


def modify_permissions(path: str, permissions: int) -> None:
    """Recursively set permissions for a path."""
    os.chmod(path, permissions)
    for root, dirs, files in os.walk(path):
        for d in dirs:
            os.chmod(os.path.join(root, d), permissions)

        for f in files:
            os.chmod(os.path.join(root, f), permissions)


def git_clone(
    repo: str, location: str, permissions: int, patch: Optional[str] = None
) -> None:
    """Clone a git repository.

    Will not overwrite a directory of there is already a folder at the specified
    location.

    Args:
        repo (str): Name of the repository to clone. Should be specified as:
            "<user_or_organization>/<repository_name>"

        location (str): Path to the location to clone to.

        permissions (str): Permissions to set on the repository.

        patch (Optional[str]): The path to a patch to apply (if desired).
    """
    repo_only: str = repo.split("/")[1]
    if os.path.exists(f"{location}/{repo_only}"):
        logger.debug(
            f"Repository {repo} already exists at {location}. Will not overwrite."
        )
        return
    cmd: List[str] = [
        "git",
        "clone",
        f"https://github.com/{repo}.git",
        f"{location}/{repo_only}",
    ]
    out: str
    out, _ = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    ).communicate()

    # Apply patch if requested
    if patch is not None:
        if not os.path.exists(patch):
            logger.warning(f"Patch requested, but path does not exist: {patch}")
        else:
            cwd: str = os.getcwd()
            os.chdir(f"{location}/{repo_only}")
            patch_cmd: List[str] = [
                "git",
                "apply",
                patch,
            ]
            subprocess.Popen(
                patch_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            ).communicate()
            logger.debug(f"Applied patch ({patch}) to {location}/{repo_only}.")
            os.chdir(cwd)

    modify_permissions(f"{location}/{repo_only}", permissions)


def clone_smalldata(producer_location: str, patch: Optional[str] = None) -> None:
    """Clone smalldata_tools based on producer location.

    Args:
        producer_location (str): Full path to the producer to be used.

        patch (Optional[str]): The path to a patch to apply (if desired).
    """
    from pathlib import Path

    repo: str = "slac-lcls/smalldata_tools"
    location: str = str(Path(producer_location).parent.parent.parent)
    git_clone(repo, location, 0o777, patch)


def grep(match_str: str, in_file: str) -> List[str]:
    """Grep for specific lines of text output.

    Args:
        match_str (str): String to search for.

        in_file (str): File to search.

    Returns:
        lines (List[str]): The matches. It may be a list with just an empty
            string if nothing is found.
    """
    cmd: List[str] = ["grep", match_str, in_file]
    out: str
    out, _ = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    ).communicate()

    lines: List[str] = out.split("\n")
    return lines


def wget(url: str, out_dir: Optional[str] = None) -> None:
    """Pull down some resource.

    Args:
        url (str): URL of the resource.

        out_dir (Optional[str]): Path of a directory to write the resource to.
            If None, will write to the current working directory, which is
            likely user scratch if running from the eLog.
    """
    cmd: List[str] = ["wget", url]
    if out_dir is not None:
        cmd.extend(["-P", out_dir])

    out, _ = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    ).communicate()


def indexamajig_summary_indexing_rate(stream_file: str) -> Dict[str, str]:
    """Return indexing rate from indexamajig output.

    Args:
        stream_file (str): Input stream file.
    """
    res: List[str] = grep("Cell parameters", stream_file)
    n_indexed: int
    if res:
        n_indexed = len(res[:-1])
    else:
        n_indexed = 0
    res = grep("End chunk", stream_file)
    n_hits: int
    rate: float
    if res:
        n_hits = len(res[:-1])
        rate = n_indexed / n_hits
    else:
        n_hits = 0
        rate = 0.0
    return {
        "Number of lattices indexed": str(n_indexed),
        "Indexing rate": f"{rate:.2f}",
    }


def compare_hkl_fom_summary(
    shell_file: str, figure_display_name: str
) -> Tuple[Dict[str, str], Optional[ElogSummaryPlots]]:
    """Analyze information produced by CrystFEL's compare_hkl.

    Extracts figures of merit information and produces text summary and plots.

    Args:
        shell_file (str): Path to output `shell-file` containing FOM information.

        figure_display_name (str): Display name of the figure in the eLog.
    """
    import numpy as np
    import numpy.typing as npt
    import holoviews as hv  # type: ignore
    import panel as pn

    with open(shell_file, "r") as f:
        lines: List[str] = f.readlines()

    header: str = lines[0]
    fom: str = header.split()[2]
    shells_arr: npt.NDArray[np.float64] = np.loadtxt(lines[1:])
    run_params: Dict[str, str] = {fom: str(shells_arr[1])}
    if shells_arr.ndim == 1:
        return run_params, None

    hv.extension("bokeh")
    pn.extension()
    xdim: hv.core.dimension.Dimension = hv.Dimension(
        ("Resolution (A)", "Resolution (A)")
    )
    ydim: hv.core.dimension.Dimension = hv.Dimension((fom, fom))

    angs_bins: npt.NDArray[np.float64] = 10.0 / shells_arr[:, 0]
    pts: hv.Points = hv.Points((angs_bins, shells_arr[:, 1]), kdims=[xdim, ydim])
    grid: pn.GridSpec = pn.GridSpec(name="Figures of Merit")
    grid[:2, :2] = pts
    tabs = pn.Tabs(grid)
    return run_params, ElogSummaryPlots(figure_display_name, tabs)


def setup_dimple_uglymol(
    final_mtz: str, experiment: str, density_display_name: str
) -> None:
    """Setup uglymol so electron density can be explored in eLog.

    Args:
        final_mtz (str): Path to the output MTZ file after running dimple.

        experiment (str): Experiment name.

        density_display_name (str): Name of the tabbed navigation to find the
            electron density in the eLog.
    """
    # We will allow passing a file name to try and guess the display name to use
    display_name: str
    if "." in density_display_name:
        display_name = density_display_name.split(".")[0]
        if "/" in display_name:
            # If it's a path we only want to keep last part of /path/to/filename
            # The standard workflow will name the stream and downstream files
            # <tag>.stream, <tag>.hkl, ...
            display_name = display_name.split("/")[-1]
        else:
            display_name = "density"
    else:
        display_name = density_display_name

    output_path: str = (
        f"/sdf/data/lcls/ds/{experiment[:3]}/{experiment}/stats/summary/{display_name}"
    )
    wasm_path: str = f"{output_path}/wasm"
    logger.debug(f"Electron density path: {output_path}")
    if not os.path.exists(output_path):
        os.makedirs(wasm_path)

    final_pdb: str = f"{final_mtz.split('.')[0]}.pdb"
    uglymol_js_url: str = (
        "https://raw.githubusercontent.com/uglymol/uglymol/master/uglymol.js"
    )
    mtz_js_url: str = (
        "https://raw.githubusercontent.com/uglymol/uglymol.github.io/master/wasm/mtz.js"
    )
    mtz_wasm_url: str = (
        "https://github.com/uglymol/uglymol.github.io/raw/master/wasm/mtz.wasm"
    )

    wget(uglymol_js_url, output_path)
    wget(mtz_js_url, wasm_path)
    wget(mtz_wasm_url, wasm_path)
    shutil.copyfile(final_mtz, f"{output_path}/final.mtz")
    shutil.copyfile(final_pdb, f"{output_path}/final.pdb")

    with open(f"{output_path}/report.html", "w") as f:
        from lute.tasks.util.html import DIMPLE_HTML

        f.write(DIMPLE_HTML)


# ====== BEGIN PENDING ALEX SCRIPTS ======
def tjump_analysis_summary(output_dir: str, run: int) -> ElogSummaryPlots:
    """Process T-Jump results and create summary plots for elog display.

    This tasklet loads the figures generated from T-Jump analysis and creates
    matplotlib plots for display in the elog. It will displate the intermediate
    plots (as tabs) and the summary plot side by side. Intermediate figure names
    are currently hard-coded because the naming is not consistent and not fully
    contained in the output_h5 file. Summary plot is hard-coded to be
    "water-peak-vs-laser-status.png" which is fixed in the T-Jump analysis script.

    Args:
        output_dir (str): Path to the output directory containing the T-Jump analysis results.
        run (int): Run number of the T-Jump analysis.

    Returns:
        ElogSummaryPlots: Plot object for elog display.
    """
    import panel as pn

    pn.extension()

    # The current visual display is not great...but that's the best Doris can do for now.
    # col_widths = [700, 900]
    row_widths = [600, 600, 1000]

    # Create panel tabs for intermediate plots loaded from png files generated by tjump analysis script.
    # Hard coding figure/filter names for now as they are not consistent.
    filter_mask_keys = [
        "scattering_falloff_filter_pass",
        "Scaled_Smoothed_AirSubtracted",
        "water-to-air_filter_pass",
        "water-peak-2sigma_filter_pass",
    ]
    tabs_list = []
    for filter_mask_key in filter_mask_keys:
        png_file = f"{output_dir}/run{run:04d}_{filter_mask_key}.png"
        if os.path.exists(png_file):
            png_pane = pn.pane.PNG(png_file, width=row_widths[0])
            tabs_list.append((filter_mask_key, png_pane))
        else:
            logger.warning(f"PNG file {png_file} not found")
    intermediate_tabs = (
        pn.Tabs(*tabs_list)
        if tabs_list
        else pn.pane.Markdown("No intermediate plots found")
    )

    # Load the scan plot
    scan_keys = [
        "lens-v-scan",
        "lens-h-scan",
    ]
    scan_tabs_list = []
    for scan_key in scan_keys:
        png_file = f"{output_dir}/run{run:04d}_{scan_key}.png"
        if os.path.exists(png_file):
            scan_pane = pn.pane.PNG(png_file, width=row_widths[1])
            scan_tabs_list.append((scan_key, scan_pane))
        else:
            logger.warning(f"PNG file {png_file} not found")
    scan_tabs = (
        pn.Tabs(*scan_tabs_list)
        if scan_tabs_list
        else pn.pane.Markdown("No scan plots found")
    )

    # Load the summary plot
    summary_png_file = f"{output_dir}/run{run:04d}_water-peak-vs-laser-status.png"
    if os.path.exists(summary_png_file):
        summary_plot = pn.pane.PNG(
            summary_png_file, name="summary_plot", width=row_widths[2]
        )
    else:
        logger.warning(f"Summary PNG file {summary_png_file} not found")
        summary_plot = pn.pane.Markdown("Summary plot not found")

    # Load intermediate and summary plots side by side, note that ElogSummaryPlots
    # expects the name to be a relative path within .../stats/summary/
    # combined_layout = pn.Row(intermediate_tabs, summary_plot)
    combined_layout = pn.Column(intermediate_tabs, scan_tabs, summary_plot)
    if "/stats/summary/" in output_dir:
        summary_folder_name = output_dir.split("/stats/summary/")[1]
    else:
        summary_folder_name = output_dir
    logger.info(f"Summary folder name in stats/summary/: {summary_folder_name}")
    combined_plots = ElogSummaryPlots(summary_folder_name, combined_layout)
    return combined_plots


# ====== END PENDING ALEX SCRIPTS ======


def classify_tjump_summary(output_dir: str, output_png: str) -> ElogSummaryPlots:
    """Process T-Jump classification results and create summary plots for elog display.

    Args:
        output_dir (str): Path to the output directory containing the T-Jump classification results.
        output_png (str): Relative path of the output png file that contains the summary plot.

    Returns:
        ElogSummaryPlots: Plot object for elog display.
    """
    import panel as pn

    pn.extension()

    # Load the summary plot
    summary_png_file = os.path.join(output_dir, output_png)
    if os.path.exists(summary_png_file):
        summary_plot = pn.pane.PNG(summary_png_file, name="summary_plot", width=900)
    else:
        logger.warning(f"Summary PNG file {summary_png_file} not found")
        summary_plot = pn.pane.Markdown("Summary plot not found")

    # ElogSummaryPlots expects the name to be consistent with a relative path within .../stats/summary/
    if "/stats/summary/" in output_dir:
        summary_folder_name = output_dir.split("/stats/summary/")[1]
    else:
        summary_folder_name = output_dir
    logger.info(f"Summary folder name in stats/summary/: {summary_folder_name}")
    return ElogSummaryPlots(summary_folder_name, summary_plot)


def solvent_scatter_summary(output_dir: str, output_png: str) -> ElogSummaryPlots:
    """Process Solvent Scattering results and create summary plots for elog display.

    Args:
        output_dir (str): Path to the output directory containing the Solvent Scattering results.
        output_png (str): Relative path of the output png file that contains the summary plot.
    """
    import panel as pn

    pn.extension()

    # Load the event code traces plot
    processed_output_png = output_png.split(".png")[0] + "_processed.png"
    event_code_traces_tabs_list = []
    for event_code_traces_key in [output_png, processed_output_png]:
        png_file = os.path.join(output_dir, event_code_traces_key)
        if os.path.exists(png_file):
            event_code_traces_pane = pn.pane.PNG(png_file, width=1000)
            event_code_traces_tabs_list.append(
                (event_code_traces_key, event_code_traces_pane)
            )
        else:
            logger.warning(f"PNG file {png_file} not found")
    event_code_traces_tabs = (
        pn.Tabs(*event_code_traces_tabs_list)
        if event_code_traces_tabs_list
        else pn.pane.Markdown("No event code traces plots found")
    )
    # ElogSummaryPlots expects the name to be consistent with a relative path within .../stats/summary/
    if "/stats/summary/" in output_dir:
        summary_folder_name = output_dir.split("/stats/summary/")[1]
    else:
        summary_folder_name = output_dir
    logger.info(f"Summary folder name in stats/summary/: {summary_folder_name}")
    return ElogSummaryPlots(summary_folder_name, event_code_traces_tabs)
