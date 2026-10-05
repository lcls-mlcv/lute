"""LUTE Managed Tasks.

Executor-managed Tasks with specific environment specifications are defined
here.
"""

import os

from lute.execution.executor import Executor, MPIExecutor
from lute.tasks.util.environment import setup_smd2_env
from lute.tasks.tasklets import (
    clone_smalldata,
    compare_hkl_fom_summary,
    indexamajig_summary_indexing_rate,
    setup_dimple_uglymol,
)

# Tests
#######
Tester: Executor = Executor("Test")
"""Runs a basic test of a first-party Task."""
Tester.shell_source("/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh")

BinaryTester: Executor = Executor("TestBinary")
"""Runs a basic test of a multi-threaded third-party Task."""
BinaryTester.shell_source("/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh")

BinaryErrTester = Executor("TestBinaryErr")
"""Runs a test of a third-party task that fails."""
BinaryErrTester.shell_source("/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh")

SocketTester: Executor = Executor("TestSocket")
"""Runs a test of socket-based communication."""
SocketTester.shell_source("/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh")

WriteTester: Executor = Executor("TestWriteOutput")
"""Runs a test to confirm database writing."""
WriteTester.shell_source("/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh")

ReadTester: Executor = Executor("TestReadOutput")
"""Runs a test to confirm database reading."""
ReadTester.shell_source("/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh")

MultiNodeCommunicationTester: MPIExecutor = MPIExecutor("TestMultiNodeCommunication")
"""Runs a test to confirm communication works between multiple nodes."""
MultiNodeCommunicationTester.shell_source(
    "/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh"
)

RequestTester: Executor = Executor("TestRequest")
"""Runs a test whether requests go from Task to Executor to Workflow manager."""
RequestTester.shell_source("/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh")

UnresponsiveTester: Executor = Executor("TestUnresponsive")
"""Runs a test to see if unresponsive jobs are killed."""
RequestTester.shell_source("/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh")

# SmallData-related
###################
SmallDataProducer: Executor = Executor("SubmitSMD")
"""Runs the production of a LCLS1 smalldata HDF5 file."""
SmallDataProducer.shell_source("/sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh")
SmallDataProducer.add_tasklet(
    clone_smalldata,
    ["{{ producer }}", f"{os.getenv('LUTE_PATH')}/config/templates/smd.patch"],
    when="before",
    set_result=False,
    set_summary=False,
)
SmallDataProducer.update_environment(
    {
        "PYTHONPATH": "/sdf/group/lcls/ds/tools/LCLSGeom/src",
    }
)

SmallDataProducer2: Executor = Executor("SubmitSMD")
"""Runs the production of a LCLS2 smalldata HDF5 file."""
SmallDataProducer2.shell_source(
    "/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh"
)
SmallDataProducer2.add_tasklet(
    clone_smalldata,
    ["{{ producer }}", f"{os.getenv('LUTE_PATH')}/config/templates/smd.patch"],
    when="before",
    set_result=False,
    set_summary=False,
)
smd2_env = setup_smd2_env()
smd2_env["PYTHONPATH"] = "/sdf/group/lcls/ds/tools/LCLSGeom/src"
SmallDataProducer2.update_environment(smd2_env)

SmallDataProducer2Test: Executor = Executor("SubmitSMD")
"""Runs the production of a LCLS2 smalldata HDF5 file using the test environment."""
SmallDataProducer2Test.shell_source(
    "/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/pscondatest.sh"
)
SmallDataProducer2Test.add_tasklet(
    clone_smalldata,
    ["{{ producer }}", f"{os.getenv('LUTE_PATH')}/config/templates/smd.patch"],
    when="before",
    set_result=False,
    set_summary=False,
)
SmallDataProducer2Test.update_environment(setup_smd2_env)

SmallDataProducerSpack: Executor = Executor("SubmitSMD")
"""Runs the production of a LCLS2 smalldata HDF5 file using the spack environment."""
SmallDataProducerSpack.shell_source(
    "/sdf/scratch/users/d/dorlhiac/work/lcls2_spack_new/setup_env_spack.sh"
)
SmallDataProducerSpack.add_tasklet(
    clone_smalldata,
    ["{{ producer }}"],
    when="before",
    set_result=False,
    set_summary=False,
)
SmallDataProducerSpack.update_environment(setup_smd2_env)

SmallDataProducerXpp: Executor = Executor("SubmitSMD")
"""Runs the production of a LCLS2 smalldata HDF5 file using the spack environment."""
SmallDataProducerXpp.shell_source(
    "/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/xpp_drp_gpu.sh"
)
SmallDataProducerXpp.add_tasklet(
    clone_smalldata,
    ["{{ producer }}"],
    when="before",
    set_result=False,
    set_summary=False,
)
SmallDataProducerXpp.update_environment(setup_smd2_env)

SmallDataXSSAnalyzer: MPIExecutor = MPIExecutor("AnalyzeSmallDataXSS")
"""Process scattering results from a Small Data HDF5 file."""
SmallDataXSSAnalyzer.shell_source(
    "/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh"
)

SmallDataXASAnalyzer: MPIExecutor = MPIExecutor("AnalyzeSmallDataXAS")
"""Process XAS results from a Small Data HDF5 file."""
SmallDataXASAnalyzer.shell_source(
    "/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh"
)

SmallDataXESAnalyzer: MPIExecutor = MPIExecutor("AnalyzeSmallDataXES")
"""Process XES results from a Small Data HDF5 file."""
SmallDataXESAnalyzer.shell_source(
    "/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh"
)

# Geometry
##########
AgBhGeometryOptimizer: MPIExecutor = MPIExecutor("OptimizeAgBhGeometryExhaustive")
"""Run an exhaustive grid search for center/distance based on Ag Bh run."""
AgBhGeometryOptimizer.shell_source(
    "/sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh"
)

# SFX
#####
CCTBXIndexer: Executor = Executor("IndexCCTBXXFEL")
"""Runs crystallographic indexing using cctbx.xfel."""
CCTBXIndexer.shell_source("/sdf/group/lcls/ds/tools/cctbx/setup.sh")

SMDToNexusConvertor: Executor = Executor("ConvertSMDToNexus")
"""Convert smalldata HDF5 to NEXUS HDF5 for use by CCTBX."""
SMDToNexusConvertor.shell_source(
    "/sdf/group/lcls/ds/tools/cctbx-psana2/build/conda_setpaths.sh"
)
CrystFELIndexer: Executor = Executor("IndexCrystFEL")
"""Runs crystallographic indexing using CrystFEL."""
CrystFELIndexer.update_environment(
    {
        "PATH": (
            "/sdf/group/lcls/ds/tools/XDS-INTEL64_Linux_x86_64:"
            "/sdf/group/lcls/ds/tools:"
            "/sdf/group/lcls/ds/tools/crystfel/0.12.0/bin"
        )
    }
)
CrystFELIndexer.add_tasklet(
    indexamajig_summary_indexing_rate,
    ["{{ out_file }}"],
    when="after",
    set_result=False,
    set_summary=True,
)

StreamFileConcatenator: Executor = Executor("ConcatenateStreamFiles")
"""Concatenate output stream files."""

CCTBXScaler: Executor = Executor("ScaleCCTBXXFEL")
"""Runs crystallographic scaling using cctbx.xfel (scaling-only pipeline).

Produces scaled .expt/.refl files and stores the output directory in the LUTE
database so that a downstream CCTBXMerger task can auto-resolve input_path.
"""
CCTBXScaler.shell_source("/sdf/group/lcls/ds/tools/cctbx/setup.sh")

CCTBXMerger: Executor = Executor("MergeCCTBXXFEL")
"""Runs crystallographic merging using cctbx.xfel."""
CCTBXMerger.shell_source("/sdf/group/lcls/ds/tools/cctbx/setup.sh")

PartialatorMerger: Executor = Executor("MergePartialator")
"""Runs crystallographic merging using CrystFEL's partialator."""

HKLComparer: Executor = Executor("CompareHKL")  # For figures of merit
"""Runs analysis on merge results for statistics/figures of merit.."""
HKLComparer.add_tasklet(
    compare_hkl_fom_summary,
    ["{{ shell_file }}", "r{{ lute_config.run }}/{{ fom }}"],
    when="after",
    set_result=False,
    set_summary=True,
)

HKLManipulator: Executor = Executor("ManipulateHKL")  # For hkl->mtz, but can do more
"""Performs format conversions (among other things) of merge results."""

DimpleSolver: Executor = Executor("DimpleSolve")
"""Solves a crystallographic structure using molecular replacement."""
DimpleSolver.shell_source("/sdf/group/lcls/ds/tools/ccp4-8.0/bin/ccp4.setup-sh")
DimpleSolver.add_tasklet(
    setup_dimple_uglymol,
    ["{{ out_dir }}/final.mtz", "{{ lute_config.experiment }}", "{{ in_file }}"],
    when="after",
    set_result=False,
    set_summary=True,
)

PeakFinderSFX: MPIExecutor = MPIExecutor("FindPeaksSFX")
"""Performs Bragg peak finding using the PyAlgos or Peakfinder8 algorithm."""
PeakFinderSFX.shell_source("/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh")

PeakFinderSFXXpp: MPIExecutor = MPIExecutor("FindPeaksSFX")
"""Performs Bragg peak finding using the PyAlgos or Peakfinder8 algorithm."""
PeakFinderSFXXpp.shell_source(
    "/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/xpp_drp_cpu.sh"
)

PeakFinderSFXXppGpu: MPIExecutor = MPIExecutor("FindPeaksSFX")
"""Performs Bragg peak finding using the PyAlgos or Peakfinder8 algorithm."""
PeakFinderSFXXppGpu.shell_source(
    "/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/xpp_drp_gpu.sh"
)

SHELXCRunner: Executor = Executor("RunSHELXC")
"""Runs CCP4 SHELXC - needed for crystallographic phasing."""
SHELXCRunner.shell_source("/sdf/group/lcls/ds/tools/ccp4-8.0/bin/ccp4.setup-sh")

PeakFinderPsocake: Executor = Executor("FindPeaksPsocake")
"""Performs Bragg peak finding using psocake - *DEPRECATED*."""
PeakFinderPsocake.shell_source("/sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh")

# XTC
#####
Xtc1Reader: MPIExecutor = MPIExecutor("ReadXtc1")
"""Read XTC1 file and send to Xtc2Writer for conversion."""
Xtc1Reader.shell_source("/sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh")

Xtc2Writer: MPIExecutor = MPIExecutor("WriteXtc2")
"""Write XTC2 files from XTC1 data received from to Xtc1Reader."""
Xtc2Writer.shell_source("/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh")

# Cheetah
#########
CheetahRunner: Executor = Executor("RunCheetah")
"""Run Cheetah task."""
CheetahRunner.shell_source("/sdf/group/lcls/ds/tools/om/setup/setup-psana2.sh")

# BayFAI
#######
BayFAIOptimizer: MPIExecutor = MPIExecutor("BayFAI")
"""Optimize LCLS-I detector geometry using BayFAI: PyFAI coupled with Bayesian Optimization."""
BayFAIOptimizer.shell_source("/sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh")
BayFAIOptimizer.update_environment(
    {
        "NUMEXPR_MAX_THREADS": "16",
        "NUMEXPR_NUM_THREADS": "16",
        "PYTHONPATH": "/sdf/group/lcls/ds/tools/LCLSGeom/src",
    }
)

BayFAIOptimizer2: MPIExecutor = MPIExecutor("BayFAI")
"""Optimize LCLS-II detector geometry using BayFAI: PyFAI coupled with Bayesian Optimization."""
BayFAIOptimizer2.shell_source("/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh")
BayFAIOptimizer2.update_environment(
    {
        "NUMEXPR_MAX_THREADS": "16",
        "NUMEXPR_NUM_THREADS": "16",
        "PS_SRV_NODES": "0",
        "PS_EB_NODES": "1",
        "PYTHONPATH": "/sdf/group/lcls/ds/tools/LCLSGeom/src",
    }
)
