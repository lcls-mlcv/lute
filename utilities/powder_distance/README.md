# powder_distance: detector distance and poni from a calibrant run

A single MPI script that goes from an LCLS-II calibrant run (xtc2, psana2) to:

- the sample–detector distance;
- the beam centre and two small detector tilts;
- figures, a `SUMMARY.md`, a `geometry.json`;
- a pyFAI `.poni` in the LCLSGeom frame, which can go straight into the SmallData `azav_lclsgeom` `poni_file:`.

It is meant as a fast (about 2 min end to end on the eLog), independent cross-check of BayFAI. A small wrapper lets it run as a one-click eLog workflow that posts a report to the eLog **Summaries** tab.

Nothing in it is specific to one experiment, hutch or detector. It has been used with the jungfrau 16M and LaB6, and any pyFAI calibrant should work.

| File | Role |
|---|---|
| `powder_distance.py` | The fit (powder → rings → geometry → report) |
| `submit_powder_distance.sh` | eLog workflow entry point; only calls `sbatch` |
| `powder_distance_job.sh` | Slurm job body; activates psana and runs the fit under `mpirun` |
| `set_poni.sh` | Points a SmallData config at a new poni, with backups |

## Requirements

- The psana2 environment: `source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh`.
- [LCLSGeom](https://github.com/lcls-mlcv/LCLSGeom) on `PYTHONPATH`, for the `.poni`. On S3DF: `export PYTHONPATH=/sdf/group/lcls/ds/tools/LCLSGeom/src:$PYTHONPATH`. Without it, everything except the poni is still written.
- pyFAI (for the calibrant d-spacings), scipy, matplotlib, h5py and mpi4py. All are in the psana2 environment.
- numba (in the psana2 environment). It is optional: without it the fit falls back to numpy and is several times slower.

## Quick start (interactive)

```bash
source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
export PYTHONPATH=/sdf/group/lcls/ds/tools/LCLSGeom/src:$PYTHONPATH

mpirun -n 12 python powder_distance.py -e <experiment> -r <run> -c LaB6 --expected <lo_mm> <hi_mm>
```

The powder is saved, so you can refit it without reading the xtc2 again, for example with different options:

```bash
python powder_distance.py -e <experiment> -r <run> --powder <exp>_r<run>_<det>/<exp>_r<run>_<det>_powder.h5
```

`--powder` also accepts a LUTE/SmallData h5 that has `Sums/<det>_calib(_max)` and `UserDataCfg/<det>/{x,y,ix,iy,mask}`.

## Outputs

Outputs go to `./<exp>_r<run>_<det>/`, or to the `-o` directory if given.

| File | What it is |
|---|---|
| `SUMMARY.md` | Distance, beam centre, tilts, wavelength, rings matched, warnings |
| `geometry.json` | The same numbers, machine-readable |
| `r<run>_<det>.poni` | pyFAI poni in the LCLSGeom frame, for SmallData `azav_lclsgeom` (`poni_file:`) |
| `*_rings.png` | Powder image with the predicted rings drawn on: the main visual check |
| `*_profile.png` | Background-subtracted intensity vs 2θ, with the calibrant positions dashed |
| `*_score_vs_distance.png` | Fit agreement vs distance; there should be one sharp peak |
| `*_powder.h5` | Mean and max calib image plus pixel coordinates (reuse it with `--powder`) |
| `fit.log` | Everything printed during the fit, including per-phase `[time]` lines |

## As an eLog workflow

The eLog runs workflow executables in a pod with a read-only `/sdf/data` and no Kerberos. So `submit_powder_distance.sh` only submits `powder_distance_job.sh` to Slurm, and the job does the work. The defaults are 16 ranks on `milano`, account `lcls:<exp>` and a 1 h limit. A run takes about 2 min for 1500 events. Most of that is the psana calibration-constant fetch at run start.

The job writes its report to `/sdf/data/lcls/ds/<hutch>/<exp>/stats/summary/Detector_Distance/r<run>_<det>/report.html`, which the eLog shows under **Summaries**. The report has the figures inlined. The job also posts the key numbers to the run's workflow status through `$JID_UPDATE_COUNTERS`. Neither step needs Kerberos.

**One-time setup per experiment:** create the log directory, because the pod cannot create it and the wrapper refuses to submit without it:

```bash
mkdir -p /sdf/data/lcls/ds/<hutch>/<exp>/results/$USER/geometry/logs
```

**Register** it in eLog → Workflow → Definitions:

| Field | Value |
|---|---|
| Executable | `<path to this folder>/submit_powder_distance.sh` |
| Parameters | Passed to `powder_distance.py`, e.g. `-c LaB6 --expected <lo_mm> <hi_mm>`. Add `--reservation NAME` for a Slurm reservation (handled by the wrapper and not passed on). |
| Trigger | **Manual**. Fire it only on calibrant runs, because a data run has no rings. |
| Location / As user | S3DF / the account that owns the output directory |

The following environment variables can be set by hand, e.g. when running the wrapper outside the eLog:

| Variable | Default | Meaning |
|---|---|---|
| `EXPERIMENT`, `RUN_NUM` | set by the eLog | Required |
| `OUT_BASE` | `/sdf/data/lcls/ds/<hutch>/<exp>/results/$USER/geometry` | Results and `logs/` |
| `NTASKS` | 16 | MPI ranks |
| `SLURM_RESERVATION` | none | Same as `--reservation` |
| `PD_SCRIPT` | `powder_distance.py` | Script to run (for testing a modified copy) |

For example:

```bash
EXPERIMENT=<exp> RUN_NUM=<run> ./submit_powder_distance.sh -c LaB6 --expected 140 160
```

## Options (`powder_distance.py`)

| Option | Default | Use |
|---|---|---|
| `-e/--exp`, `-r/--run` | required | Experiment and run |
| `-d/--det` | `jungfrau` | Area detector name |
| `-c/--calibrant` | `LaB6` | Any pyFAI calibrant: `CeO2`, `Si`, `AgBh`, `LaB6_SRM660c`, ... |
| `-n/--events` | 1500 | Shots in the powder |
| `-o/--out` | `./<exp>_r<run>_<det>` | Output directory |
| `--expected LO HI` | none | Expected distance band (mm). It is shaded in the plot, and a warning is given if D falls outside it. |
| `--dist-range MIN MAX` | 40 600 | Distance search range (mm) |
| `--centre-search` | 6 | ± beam-centre search about the psana origin (mm) |
| `--beamstop` | 25 | Ignore pixels within this radius of the origin (mm) |
| `--max-rings` | 15 | Number of calibrant rings used |
| `--ring-frac` | 2.5 | Top % of high-passed pixels taken as ring candidates |
| `--energy-eV` / `--wavelength-A` | from ebeam | Override the wavelength, e.g. when there is no ebeam data |
| `--ebeam` | `ebeamh` | psana detector carrying `ebeamPhotonEnergy` |
| `--powder FILE` | none | Skip the xtc2 step and reuse a powder h5 |
| `--datasource-kw` | none | Extra `DataSource` kwargs, e.g. `dir=/path/xtc` |
| `--no-tilt` | off | Fit distance and centre only |
| `--no-figures` | off | Skip the figures (speed tests only) |
| `--elog-summary [NAME]` | off (on in the eLog job) | Publish to Summaries under `NAME` (default `Detector_Distance`) and post the run status |

## Is the result good? Check these

1. **`*_rings.png`:** the circles sit on the rings all the way round, not just on one side.
2. **`*_score_vs_distance.png`:** there is one sharp peak. A second peak of similar height means the result is ambiguous.
3. **Warnings in `SUMMARY.md` (and the eLog page):** there should be none. The script warns if:
   - D or the beam centre is at the edge of its search range;
   - another distance scores more than 60 % of the best;
   - the two half-split distances differ by more than 0.5 mm;
   - fewer than 3 rings matched;
   - D is outside `--expected`.
4. **Half-split ΔD** is well under 0.1 mm for a good powder.

If the result disagrees strongly with BayFAI or with the expected motor position, check the detector motor position for that run before using either number.

## Applying the poni to SmallData

After the checks above:

```bash
./set_poni.sh <OUT_BASE>/<exp>_r<run>_<det>/r<run>_<det>.poni \
    <smd>/lute_output/<config>.yaml [<smd>/<config>.yaml.j2 ...]
```

This sets `poni_file:` in each given config after taking a timestamped backup. It refuses if a config still has an `ai_kwargs` block, because that would override the poni. The change only applies to runs fired afterwards.

The fit deliberately doesn't update SmallData automatically: a bad fit would then silently set the geometry for every following run. A poni is only valid for runs at the same detector position.

## How it works

1. **Powder (MPI):** mean and max of `<det>.raw.calib` over N events, plus pixel coordinates, mask and photon energy. λ is the median of `ebeamPhotonEnergy`.
2. **Ring pixels:** a per-panel `median_filter` high-pass. The top 2.5 % of pixels outside the beamstop radius are taken as ring candidates, and single-crystal spots are capped.
3. **Fit** of D, the beam centre (cx, cy) and two small tilts. The fit maximizes how close each candidate pixel's 2θ is to the nearest calibrant cone, in two stages:
   - a coarse search over D (log grid) × beam centre (1 mm grid);
   - Nelder-Mead refinement while the ring width narrows from 1.5 to 0.3 mrad.
4. **Checks:** a refit on two disjoint halves of the pixels, the score vs D, and per-ring residuals.
5. **poni:** the psana → LCLSGeom pyFAI mapping is fitted from the pixel coordinates, and must be rigid. The beam centre is converted with it, and rotations are written as 0.

### Performance

- The scoring kernel is numba: it fuses 2θ, nearest-cone and score, is multithreaded, and is batched over geometries for the grid scans.
- The per-panel median filter runs in a thread pool.
- The LCLSGeom conversion runs in a background thread during the fit.
- After the powder step, rank 0 takes over the CPU cores of all ranks. The other ranks wait in a sleeping barrier, so their allocation stays alive. This matters under Slurm, which pins each rank to one core.
- The powder accumulates in float32.

With these, the fit phase takes about 30 s. The rest of the job is dominated by psana start-up (calibration constants) and the event loop. Using fewer than ~1500 events saves little time but measurably moves D (by up to ~0.1 mm), so 1500 is the default.

## Caveats

- The psana panel geometry is taken as exact; there is no per-panel refinement. BayFAI and pyFAI-calib can refine more.
- The tilt model is small-angle, and there is no polarisation correction.
- The fitted tilts are reported but not written to the poni (rot = 0). That is fine for tilts of ~0.1° or less.
- The beam centre in `SUMMARY.md` is in the psana lab frame. The poni's poni1/poni2 are in the LCLSGeom pyFAI frame, so compare poni values with poni values.
- The expected accuracy is about ±0.3 mm in distance. The half-split ΔD measures repeatability, not accuracy. In practice the results have agreed with BayFAI to ~0.1 mm on several calibrant runs.
