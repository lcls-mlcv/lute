# SFX CrystFEL: mfxl1038923 run 58

LCLS-I (psana1), ePix10k2M at about 50 mm, ~9.8 keV. No unit cell was ever recorded for this sample,
so both indexers run BLIND and the DAG stops at the concatenated stream (no merge).

GLINT's cross-frame consensus on this run finds a cell near 43.6/67.8/89.1 A.

Peak finding: On 30 calibrated events of this run a 200 threshold leaves a median 0 connected 2-30
px components but a 90th percentile of 172; 500 leaves a 90th percentile of 70. 300 is a compromise
and the least certain setting in these tests.

## Smoke test only

This test passes when the workflow completes. It has no science criterion: there is no reference
cell, no merge and no independent indexing of this run to compare with. At the 300 threshold, 57%
of the peaks are faint (median intensity 1000-2000) and sit at 1.8-2.8 A, which looks like detector
noise near the edge, so the number of indexed frames here says little about either indexer.

## Inputs committed with this test

`mfxl1038923_r0058.geom`: Panel X/Y from mfxx49820 results/btx/geom/r0016.geom (same detector
model); psana's deployed geometry for this experiment has a placeholder distance. The 50 mm distance
comes from a consensus-support scan and agrees with a 3.669 A ice ring at 49.4 mm. The interaction
point moved between runs (r0278 sits about 8 mm further), so this file is specific to r0058.

The config reads these files from `/sdf/group/lcls/ds/tools/lute/test_utilities/sfx/`; the copies
here are the source to deploy there (`maybe_lyso.cell` is already in `test_utilities/`).
