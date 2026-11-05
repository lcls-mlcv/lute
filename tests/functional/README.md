# Functional Tests

## Usage
- All tests should be placed here in a sub-directory.
- The `run_functional.py` script (one-level up) will run the tests for every sub-directory.
- Each sub-directory **must contain:**
  - `config.yaml`: The LUTE configuration YAML for this test. **It must have a specific** `experiment` **and** `run` defined in the LUTE header.
  - `dag.yaml`: The workflow definition YAML for this test.
- Each sub-directory may also contain:
  - `SHOULD_FAIL`: An empty file indicating the workflow is intended to return a failure status.
  - `README.md`: An explanatory README for the test.

## List of tests

| Test Name                     | Workflow                                                                                                   | Experiment   | Run | Additional Comments                                                   |
|:-----------------------------:|:----------------------------------------------------------------------------------------------------------:|:------------:|:---:|:---------------------------------------------------------------------:|
| basic_tests                   | Basic LUTE test Tasks                                                                                      | xpptut15     | 670 | Experiment/run are not used by test Tasks. Required for compatibility |
| smd_xpp_default               | SmallDataProducer                                                                                          | xpptut15     | 650 | xpplv9818 run 127. This is to test default production only.           |
| smd_mfx_default               | SmallDataProducer                                                                                          | mfxx49820    | 16  | Default small data production                                         |
| smd2_mfx_prod                 | SmallDataProducer2                                                                                         | mfx101344525 | 70  | LCLS2 non-default SMD (MFX).                                          |
| smd2_multi_node               | SmallDataProducer2                                                                                         | mfx101262725 | 96  | LCLS2 non-default SMD (MFX) submitted across multiple nodes.          |
| param_generation              | Basic LUTE tests + param generation                                                                        | xpptut15     | 670 | Experiment/run not used by the test but required for compatibility.   |
| peakfinder8_lcls2             | Run FindPeaksSFX with the peakfinder8 v1 algorithm on LCLS2 data.                                          | mfx101343025 | 170 | Some Jungfrau16M data.                                                |
| peakfinder8_lcls2_compression | Run FindPeaksSFX with the peakfinder8 v1 algorithm on LCLS2 data but do compress/decompress with RoiBinSZ. | mfx101343025 | 170 | Some Jungfrau16M data.                                                |
| basic_rest                    | Basic tests of `maestro` REST APIs                                                                         | xpptut15     | 670 | Experiment/run are not used by test Tasks. Required for compatibility |
| smd_bayfai                    | SmallDataProducer → BayFAIOptimizer                                                                        | mfx100824024 | 5   | psana1, epix10k2M, LaB6 calibrant. Powder auto-resolved from SMD.    |
| smd2_bayfai                   | SmallDataProducer2 → BayFAIOptimizer2                                                                      | mfx100852324 | 298 | psana2, jungfrau, AgBh calibrant. Powder auto-resolved from SMD2.    |
| smd2_xss                      | SmallDataProducer2 → SmallDataXSSAnalyzer                                                                  | mfx101344525 | 82  | psana2, jungfrau, PyFAI azint (r0082.poni), lxt/lens scan.           |
| sfx_mfx101343025_r194_crystfel | FindPeaksSFX → CrystFEL (xgandalf) → merge | mfx101343025 | 194 | Jungfrau16M, monoclinic lysozyme, BayFAI-refined r0139 geometry. |
| sfx_mfx101343025_r194_glint | FindPeaksSFX → GLINT → merge | mfx101343025 | 194 | Same peaks as the CrystFEL test. GLINT needs an A100 node. |
| sfx_mfx100848724_r51_crystfel | FindPeaksSFX → CrystFEL (xgandalf) → merge | mfx100848724 | 51 | Jungfrau16M, tetragonal lysozyme, refined 356 mm geometry. |
| sfx_mfx100848724_r51_glint | FindPeaksSFX → GLINT → merge | mfx100848724 | 51 | Same peaks as the CrystFEL test. GLINT needs an A100 node. |
| sfx_cxil1015922_r136_crystfel | FindPeaksSFX (psana1) → CrystFEL, no merge (smoke) | cxil1015922 | 136 | LCLS-I, Jungfrau4M, monoclinic C2 'B2' protein. |
| sfx_cxil1015922_r136_glint | FindPeaksSFX (psana1) → GLINT → CrystFEL, no merge (smoke) | cxil1015922 | 136 | LCLS-I. Same peaks as the CrystFEL test. |
| sfx_mfxl1038923_r58_crystfel | FindPeaksSFX (psana1) → CrystFEL, blind | mfxl1038923 | 58 | LCLS-I, epix10k2M. No known cell: indexing only, no merge. |
| sfx_mfxl1038923_r58_glint | FindPeaksSFX (psana1) → GLINT, blind | mfxl1038923 | 58 | LCLS-I, epix10k2M. No known cell: indexing only, no merge. |
|                               |                                                                                                            |              |     |                                                                       |

## SFX tests: what they need deployed

- Geometries and cells are read from `/sdf/group/lcls/ds/tools/lute/test_utilities/sfx/`. The copies
  committed in each test directory are the source to deploy there.
- The GLINT tests run the GLINT release pinned in `lute/io/models/glint_index.py`,
  `/sdf/group/lcls/ds/tools/glint/v0.1.0`, which `GLINTIndexer` puts on PYTHONPATH inside the conda1
  torch environment (`lute/managed_tasks.py`); set `LUTE_GLINT_ROOT` to run another checkout. The
  indexing step runs on ampere, so a run_functional.py `--account` must be valid there.

## SFX tests: what counts as a pass

- The merged SFX tests come in pairs on the same run and events (r194 and r51, CrystFEL and GLINT). A
  pair passes when both workflows complete and the GLINT test's merged overall CC1/2 is at least the
  CrystFEL test's. Crystal counts are reported, not thresholded. run_functional.py checks only
  completion; the CC1/2 are read from the CompareHKL logs by hand. (An absolute bar of CC1/2 > 0.3 with
  ~100 crystals was dropped: the GLINT handoff on all 17,872 events of r51 merged 242 crystals at 0.175.)
- The indexing rates quoted in the test READMEs come from the beamline's cctbx or the users' CrystFEL
  runs. They are floors, not targets: the peak lists here include crystals those runs did not index.
  A test that indexes far MORE than the floor has to be justified by its merge, not by the count.
- mfxl1038923 r58 (no cell) and cxil1015922 r136 (too few crystals per run for a merge) are smoke tests:
  their DAGs stop at the concatenated stream, and they pass when the workflow completes.
