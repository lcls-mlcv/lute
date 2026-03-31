"""T-jump Solvent Scattering Analysis Airflow Workflow.

Run T-jump solvent scattering analysis after producing a powder image with SmallData.

Note:
    The task_id MUST match the managed task name when defining DAGs - it is used
    by the operator to properly launch it.

    dag_id names must be unique, and they are not namespaced via folder
    hierarchy. I.e. all DAGs on an Airflow instance must have unique ids. The
    Airflow instance used by LUTE is currently shared by other software - DAG
    IDs should always be prefixed with `lute_`. LUTE scripts should append this
    internally, so a DAG "lute_test" can be triggered by asking for "test"
"""

from typing import Dict, Any
from datetime import datetime
import os
from airflow import DAG
from airflow.decorators import task
from lute.operators.jidoperators import JIDSlurmOperator

dag_id: str = f"lute_{os.path.splitext(os.path.basename(__file__))[0]}"
description: str = "Produces T-jump solvent scattering analysis from SmallData hdf5 files."

dag: DAG = DAG(
    dag_id=dag_id,
    start_date=datetime(2025, 9, 29),
    schedule_interval=None,
    description=description,
    is_paused_upon_creation=False,
)

@task.branch(task_id="Psana1v2Brancher")
def psana1v2_branch_func(**context) -> str:
    if "dag_run" in context:
        conf: Dict[str, Any] = context["dag_run"].conf
        if "is_daq2" in conf:
            if conf["is_daq2"]:
                return "SmallDataProducer2"
            elif conf["is_daq2"] is None:
                raise ValueError("Could not determine psana version: Unknown DAQ state")
    return "SmallDataProducer"

psana1v2_brancher = psana1v2_branch_func()

smd_producer: JIDSlurmOperator = JIDSlurmOperator(task_id="SmallDataProducer", dag=dag)

smd_producer2: JIDSlurmOperator = JIDSlurmOperator(task_id="SmallDataProducer2", dag=dag)

tjump_analyzer: JIDSlurmOperator = JIDSlurmOperator(task_id="TJumpAnalyzer", dag=dag, trigger_rule="one_success")
tjump_classifier: JIDSlurmOperator = JIDSlurmOperator(task_id="TJumpClassifier", dag=dag, trigger_rule="one_success")

# Branch Workflow depending on available psana version
psana1v2_brancher >> [smd_producer, smd_producer2] 

# Either smalldata successfully ran should trigger tjump analysis
smd_producer >> tjump_analyzer
smd_producer >> tjump_classifier

smd_producer2 >> tjump_analyzer
smd_producer2 >> tjump_classifier
