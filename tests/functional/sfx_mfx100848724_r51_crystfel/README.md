# SFX CrystFEL: mfx100848724 run 51

Tetragonal lysozyme (P4_32_12, 79.17/79.17/37.96 A), Jungfrau-16M at 356.5 mm, ~11.6 keV. This is
the one run of the experiment in the public data area.

The beamline's cctbx.xfel.process integrated 225 of 17,872 events (1.3%), so 12,000 events should
give roughly 150 indexed frames, enough for the pass criterion below (4,000 gave about 50). Runs
52-55 index at 2.5-4%.

Peak finding: On 16 calibrated frames of this run the 99.9th pixel percentile is 615; a 1000
threshold leaves a median 28 connected 2-30 px components per frame, 500 leaves ~3,500.

## Pass criterion

This test is one of a pair run on the same events (`sfx_mfx100848724_r51_crystfel` and `sfx_mfx100848724_r51_glint`). Both
workflows must complete, and the GLINT test's merged overall CC1/2 must be at least the CrystFEL test's
(CompareHKL `fom: "CC"`, the "Overall CC" line in each log). The crystal counts are reported, not
thresholded. run_functional.py checks only completion; the two CC1/2 are read from the logs by hand.

Why relative: an absolute bar (CC1/2 > 0.3 with ~100 crystals) is a statement about the run, not the
indexer, and these runs cannot reach it. mfx100848724 r51 has 17,872 events in all; the GLINT handoff on
all of them merged 242 crystals at CC1/2 0.175, against 0.278 from 137 crystals on the first 12,000.
What the pair can show is whether GLINT's orientations merge at least as well as CrystFEL's own on the
same peaks. Last validation (round 5, 29 Sep 2026): r51 GLINT 0.278 (137 crystals) vs CrystFEL 0.274
(219); r194 GLINT 0.257 (67) vs CrystFEL 0.132 (70).

## Inputs committed with this test

`mfx100848724_356mm_refined.geom`: Converted from the refined DIALS geometry
results/common/geom/refined_356mm_17jun.expt with a 180-degree rotation about y (x -> -x, z -> -z).
No CrystFEL geometry for this experiment existed.

`lyso_tetragonal_p43212.cell`: the reference cell (79.17 79.17 37.96 90.00 90.00 90.00).

The config reads these files from `/sdf/group/lcls/ds/tools/lute/test_utilities/sfx/`; the copies
here are the source to deploy there (`maybe_lyso.cell` is already in `test_utilities/`).
