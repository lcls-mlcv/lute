# SFX CrystFEL: mfx101343025 run 194

Monoclinic lysozyme (P2_1, 28.0/62.5/60.9 A, beta 90.8) in about 1% of frames, Jungfrau-16M at 124.3
mm, ~15 keV.

The beamline's cctbx.xfel.process indexed 238 of 28,650 events in this run; 22 of those fall in the
first 2,000 events, so 3,000 events should give roughly 30 indexed frames.

Peak finding: On 16 calibrated frames of this run the water ring sits at 400-600 (99.9th pixel
percentile 588); a 1000 threshold leaves a median 49 connected 2-30 px components per frame, 500
leaves ~10,000.

The threshold is 5000, not 1000. At 1000, 64% of the stored peaks fall in a 4-6 A band (median 195
peaks per frame), and both indexers index that band as if it were a lattice. Matched to cctbx by
timestamp, the 31 cctbx-indexed frames among the first 3,000 events have a median brightest peak
of 134,937 ADU against 1,750 for the rest. On the 1000-threshold peak lists, "at least 10 peaks
above 5000 ADU" keeps 28 of the 31 and 310 other frames. Those 310 are not ice (their peaks sit no
closer to the ice rings than chance), so they are probably crystals cctbx did not index, and the
cctbx rate above is a floor.

## Pass criterion

This test is one of a pair run on the same events (`sfx_mfx101343025_r194_crystfel` and `sfx_mfx101343025_r194_glint`). Both
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

`mfx101343025_r0139_refined.geom`: Converted from the BayFAI-refined DIALS geometry
results/common/geom/r0139_psana_imported.expt (distance 124.285 mm) with a 180-degree rotation about
y (x -> -x, z -> -z), which preserves the hand. results/bayfai/lute_output/geom/r0139.geom is the
same geometry mirrored in x and should not be used. The shared test_utilities/jf16mgeom.geom is
nominal (0.2 m) and does not index this run.

`maybe_lyso.cell`: the reference cell (28.00 62.50 60.90 90.00 90.80 90.00).

The config reads these files from `/sdf/group/lcls/ds/tools/lute/test_utilities/sfx/`; the copies
here are the source to deploy there (`maybe_lyso.cell` is already in `test_utilities/`).
