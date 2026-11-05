# SFX GLINT: cxil1015922 run 136

LCLS-I (psana1). The users' protein 'B2', monoclinic C2 (111.94/172.23/41.23 A, beta 106.2),
Jungfrau-4M at 109.4 mm, 8.85 keV.

The users' CrystFEL 0.10.2 run indexed 1,433 crystals in this run's 37,522 events (about 3.8%), so
3,000 events should give roughly 110 with their peak finding; with this test's it gives far fewer
(see below).

Peak finding: On 30 calibrated events of this run (every 10th) a 200 threshold leaves a median 0
connected 2-30 px components per frame (max 60); 100 leaves a median 58, mostly the water ring.

## Smoke test only

This test passes when the workflow completes; the DAG stops at the concatenated stream (no merge), as
for mfxl1038923 r58. This run cannot support a merge criterion with these settings: at 30,000 events the
peak finder keeps 3,523 frames, and the GLINT handoff merged 22 crystals sharing about 5 reflections
(round 7, 1 Oct 2026). At 3,000 events CrystFEL's own run merged 3 crystals and compare_hkl crashed
(round 4), which failed the workflow. The test still exercises the monoclinic (`mCb`, unique axis b)
path end to end.

## Inputs committed with this test

`cxil1015922_r0136_b2.geom`: The geometry the users indexed with, recovered from the header of their
stream results/b2/indexing/r0136-b2-100-6-350.process/r0136-b2-100-6-350.lst.6.50.stream (the
original file is gone). Photon energy is fixed at 8852 eV there, as in their run. The lysozyme runs
of this experiment (30, 33, 34) have been purged from disk.

`cxil1015922_b2.cell`: the reference cell (111.94 172.23 41.23 90.00 106.20 90.00).

The config reads these files from `/sdf/group/lcls/ds/tools/lute/test_utilities/sfx/`; the copies
here are the source to deploy there (`maybe_lyso.cell` is already in `test_utilities/`).

## GLINT

GLINTIndexer reads the .cxi files FindPeaksSFX writes and reuses their stored peakfinder8 peaks
(`peakfinder: stored`), so it indexes exactly the peaks the CrystFEL test does. It is given the same
cell and writes only orientations (`tofile`, lattice `mCb`). CrystFELIndexer then reads them with
`--indexing=file`, refines them, checks them against the cell and integrates, so both tests share
CrystFEL's integration and differ only in who found the orientation.

Why not GLINT's own integration (`integrate: true`): on the round-4 validation of mfx100848724 r51,
on the 171 frames both indexers indexed (same orientation on 166), CrystFEL's integration merged to
CC1/2 0.27 and GLINT's to 0.08. Handing GLINT's orientations to CrystFEL gave 0.27. CrystFEL's
refinement also rejected all 887 frames that only GLINT had indexed, which the strict gate had passed.

GLINT is not bundled with LUTE. `GLINTIndexer` (lute/managed_tasks.py) runs the release pinned in
`lute/io/models/glint_index.py`, `/sdf/group/lcls/ds/tools/glint/v0.1.0`, in the conda1 torch
environment `ana-4.0.58-py3-minipytorch`; that directory has to exist like the files in
`test_utilities/sfx/` (GLINT v0.1.0 or later has everything these tests use). The indexing step needs
a GPU node (ampere). If run_functional.py is given `--account=...`, that account must also be valid
on ampere.
