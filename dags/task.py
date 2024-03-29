from datetime import datetime, timedelta
import sys

sys.path.append(r"/home/peerapat/wongnai-sentiment/resources/trb_aware/trbawarepipeline")
from main import job_recieve
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.docker_operator import DockerOperator

default_args = {
    'depends_on_past': False,
    'owner': 'airflow',
    'start_date': datetime(2022, 11, 1),
}
    
with DAG(
        'Trainning_models', 
        default_args=default_args,
        catchup=False,
        schedule='* * 1 * *' 
) as dag1:

    preparedata = PythonOperator(
        task_id='preparedata', 
        python_callable=job_recieve,
        op_args=["/home/peerapat/wongnai-sentiment/resources/trb_aware/jobs/Trainning_models/job_preparedata.json"]     
    )
    datacleansing = PythonOperator(
        task_id='datacleansing',
        python_callable=job_recieve,
        op_args=["/home/peerapat/wongnai-sentiment/resources/trb_aware/jobs/Trainning_models/job_datacleansing.json"]
    )
    datasplit = PythonOperator(
        task_id='datasplit',
        python_callable=job_recieve,
        op_args=["/home/peerapat/wongnai-sentiment/resources/trb_aware/jobs/Trainning_models/job_datasplit.json"]
    )
    datatoindex = PythonOperator(
        task_id='datatoindex',
        python_callable=job_recieve,
        op_args=["/home/peerapat/wongnai-sentiment/resources/trb_aware/jobs/Trainning_models/job_datatoindex.json"]
    )
    buildmodel = PythonOperator(
        task_id='buildmodel',
        python_callable=job_recieve,
        op_args=["/home/peerapat/wongnai-sentiment/resources/trb_aware/jobs/Trainning_models/job_modelbuilder.json"]
    )
    evaluation = PythonOperator(
        task_id='evaluation',
        python_callable=job_recieve,
        op_args=["/home/peerapat/wongnai-sentiment/resources/trb_aware/jobs/Trainning_models/job_evaluatemodels.json"]
    ) 
    deploy = BashOperator(task_id="deploymodel",
        bash_command='cd /home/peerapat/wongnai-sentiment/resources/trb_aware/ && sudo docker compose up -d',)
   
    preparedata>>datacleansing>>datasplit>>datatoindex>>buildmodel>>evaluation>>deploy
    
with DAG(
        'models_predict', 
        default_args=default_args,
        catchup=False,
        schedule='* * 1 * *' 
) as dag2:
    
    preparedata = PythonOperator(
        task_id='preparedata', 
        python_callable=job_recieve,
        op_args=["/home/peerapat/wongnai-sentiment/resources/trb_aware/jobs/models_predict/่job_preparedata.json"]     
    )
    datacleansing = PythonOperator(
        task_id='datacleansing',
        python_callable=job_recieve,
        op_args=["/home/peerapat/wongnai-sentiment/resources/trb_aware/jobs/models_predict/่job_datacleansing.json"]
    )
    datatoindex = PythonOperator(
        task_id='datatoindex',
        python_callable=job_recieve,
        op_args=["/home/peerapat/wongnai-sentiment/resources/trb_aware/jobs/models_predict/่job_datatoindex.json"]
    )
    modelcaller = PythonOperator(
        task_id='modelcaller',
        python_callable=job_recieve,
        op_args=["/home/peerapat/wongnai-sentiment/resources/trb_aware/jobs/models_predict/job_callmodel.json"]
    )
    
    preparedata>>datacleansing>>datatoindex>>modelcaller