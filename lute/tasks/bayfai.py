"""
Classes for geometry optimization tasks.

Classes:
    BayFAI: optimize detector geometry using PyFAI coupled with Bayesian Optimization

"""

__all__ = ["BayFAI"]
__author__ = "Louis Conreux"

from lute.tasks._bayfai import BayFAIOpt, _ph_add, _ph_emit, _proc_start_epoch
from lute.io.models.bayfai import BayFAIParameters

from lute.tasks.task import Task
from lute.tasks.dataclasses import TaskStatus, ElogSummaryPlots
from lute.execution.logging import get_logger

import os
import logging
import time  # type: ignore

logger: logging.Logger = get_logger(__name__)


class BayFAI(Task):
    """Optimize detector geometry using PyFAI coupled with Bayesian Optimization."""

    def __init__(
        self, *, params: BayFAIParameters, use_mpi: bool = True, row_ids=None
    ) -> None:
        self._task_parameters: BayFAIParameters
        super().__init__(params=params, use_mpi=use_mpi, row_ids=row_ids)

    def _run(self) -> None:
        start_time = time.time()
        optimizer: BayFAIOpt = BayFAIOpt(
            exp=self._task_parameters.lute_config.experiment,
            run=int(self._task_parameters.lute_config.run),
        )
        optimizer.setup(
            detname=self._task_parameters.detname,
            h5=self._task_parameters.h5,
            Imin=self._task_parameters.bayfai_params.Imin,
            calibrant=self._task_parameters.calibrant,
            fixed=self._task_parameters.fixed,
            parallelized=self._task_parameters.parallelized,
            wavelength=self._task_parameters.wavelength,
        )
        _ph_add("startup", _proc_start_epoch(), start_time)
        _ph_add("setup", start_time, time.time())
        _ph_emit(optimizer.rank)
        bayfai_hyperparams = {
            "n_samples": self._task_parameters.bayfai_params.n_samples,
            "n_iterations": self._task_parameters.bayfai_params.n_iterations,
            "max_rings": self._task_parameters.bayfai_params.max_rings,
            "pts_per_deg": self._task_parameters.bayfai_params.pts_per_deg,
            "Imin": optimizer.Imin,
            "prior": self._task_parameters.bayfai_params.prior,
            "beta": self._task_parameters.bayfai_params.beta,
            "step": self._task_parameters.bayfai_params.step,
            "lbda": self._task_parameters.bayfai_params.lbda,
            "seed": self._task_parameters.bayfai_params.seed,
            "n_cand": self._task_parameters.bayfai_params.n_cand,
        }
        optimizer.bayfai_opt(
            center=self._task_parameters.center,
            bounds=self._task_parameters.bounds,
            res=self._task_parameters.resolutions,
            **bayfai_hyperparams,
        )
        t_post0 = time.time()
        if optimizer.rank == 0:
            logger.info("Optimization complete")
            logger.info(f"Elapsed time: {time.time() - start_time:.2f} s")
            params = optimizer.params
            score = optimizer.neglog_score
            distance = params[0]
            cy = params[1]
            cx = params[2]
            logger.info(f"Detector Distance to Sample: {distance:.6f}")
            logger.info(f"Beam center ({cx:.6f}, {cy:.6f})")
            logger.info(
                f"Rotations: \u03b8x = ({params[3]:.2e}, \u03b8y = {params[4]:.2e}, \u03b8z = {params[5]:.2e})"
            )
            logger.info(f"Best Score: {score:.2e}")
            fig_folder = os.path.join(
                self._task_parameters.lute_config.work_dir, "figs"
            )
            os.makedirs(fig_folder, exist_ok=True)
            plot = f"{fig_folder}/bayFAI_summary_{optimizer.exp}_r{optimizer.run:0>4}_{self._task_parameters.detname}.png"
            t = time.time()
            optimizer.update_geometry(
                self._task_parameters.out_file, self._task_parameters.detname
            )
            t_plots = time.time()
            _ph_add("post.update_geometry", t, t_plots)
            powder_plot, qs, resolutions = optimizer.create_interactive_powder()
            diagnostics_plot = optimizer.create_diagnostics_panel()
            _ = optimizer.create_summary_plot(plot=plot)
            _ph_add("post.plots", t_plots, time.time())
            import panel as pn  # type: ignore

            pn.extension("matplotlib", "bokeh")
            plots = pn.Row(
                pn.pane.Matplotlib(diagnostics_plot, sizing_mode="fixed"),
                powder_plot,
            )
            content = pn.Column(
                pn.pane.Markdown(
                    "### Detector Geometry Optimization Summary",
                    styles={
                        "font-size": "2em",
                        "font-weight": "bold",
                        "text-align": "center",
                        "margin": "0 auto",
                        "display": "block",
                    },
                ),
                plots,
                sizing_mode="stretch_width",
            )
            t = time.time()
            content.save(
                f"{fig_folder}/bayFAI_summary_{optimizer.exp}_r{optimizer.run:0>4}_{self._task_parameters.detname}.html",
                embed=True,
            )
            _ph_add("post.html_save", t, time.time())
            plots = pn.Tabs(content)
            self._result.summary = []
            self._result.summary.append(
                {
                    "Detector distance (m)": f"{params[0]:.6f}",
                    "Detector center (pix)": (
                        f"{cx / optimizer.detector.pixel_size:.3f}",
                        f"{cy / optimizer.detector.pixel_size:.3f}",
                    ),
                    "Lowest q": f"{qs['closest']:.3f} \u00c5-1 | {resolutions['closest']:.3f} \u00c5",
                    "Highest q": f"{qs['furthest']:.3f} \u00c5-1 | {resolutions['furthest']:.3f} \u00c5 (detector corner)",
                }
            )
            logger.info(
                f">>> Lowest q : {qs['closest']:.3f} \u00c5-1 | {resolutions['closest']:.3f} \u00c5"
            )
            logger.info(
                f">>> Highest q : {qs['furthest']:.3f} \u00c5-1 | {resolutions['furthest']:.3f} \u00c5 (detector corner)"
            )
            self._result.summary.append(
                ElogSummaryPlots(
                    f"Geometry_Fit/r{self._task_parameters.lute_config.run:0>4}/{self._task_parameters.detname}",
                    plots,
                )
            )
            self._result.task_status = TaskStatus.COMPLETED
            _ph_add("post", t_post0, time.time())
        t = time.time()
        _ph_add("run_end", t, t)
        _ph_emit(optimizer.rank)
