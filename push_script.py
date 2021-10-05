import dateutil
from airflow import DAG
from airflow.models import Variable
from datetime import timedelta
from plugins.operators.databricks_operator import DatabricksSubmitRunOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import BranchPythonOperator
from ssc.utils.databricks import get_databricks_cluster_config
from ssc.utils.notify import on_success_notify, on_retry_notify,\
    on_failure_notify
from ssc.utils.spark import databricks_spark_submit_cmd


START_DATE = "2021-04-16"
# Once a week at 6am on Saturday
SCHEDULE_INTERVAL = "0 6 * * 6"

DATALAKE_BUCKET = Variable.get('data_lake_bucket', 's3a://ssc-data-lake-qa')


start_date = dateutil.parser.parse(START_DATE).replace(tzinfo=None)

dag = DAG(
    dag_id='churn_analytic_pipeline',
    catchup=False,
    default_args={
        'depends_on_past': False,
        'owner': 'datasci',
        'pool': 'datasci',
        'retries': 0,
        'retry_delay': timedelta(minutes=5),
        'start_date': start_date,
        'dagrun_timeout': timedelta(hours=6)
    },
    concurrency=5,
    max_active_runs=1,
    schedule_interval=SCHEDULE_INTERVAL
)

cluster_config = get_databricks_cluster_config(
    service='datasci analytic',
    billing='datasci',
    node_type_id='c4.8xlarge',
    driver_node_type_id='r4.8xlarge',
    autoscaling_enabled=False,
    workers=15,
    spark_version='7.0.x-scala2.12'
)


def create_churn_analytic_training_job(dag, cluster_config):
    """ Function to create a task to generate churn analytic training result and model

    Args:
        dag (DAG type): DAG object representing the Airflow DAG definition.

    Return:
        DatabricksSubmitRunOperator: Airflow operator object.
    """
    # Specifications of the Databricks cluster for the task

    app_args = {
        "to_train": "true",
        "output_dir": '{{ var.value.churn_analytic_training_output_bucket }}'
                      '/effective_date={{ ds_nodash }}'
    }

    parameters = databricks_spark_submit_cmd(
        name="churn_analytic_training for {{ ds }}",
        spark_args={"spark.driver.maxResultSize": "200g"},
        py_files=("{{ var.value.customeranalytics_egg_location }}"),
        py_driver_file=("{{ var.value.artifacts_location }}"
                        "/{{ var.value.ssc_python_driver }}"),
        app_cmd_args=[
            "customeranalytics.cli",
            "churn_analytic",
            "contract_churn_classification"],
        app_args=app_args
    )

    return DatabricksSubmitRunOperator(
        task_id="churn_analytic_training",
        dag=dag,
        new_cluster=cluster_config,
        json={
            "spark_submit_task": {
                "parameters": parameters
            }
        },
        on_success_callback=on_success_notify,
        on_retry_callback=on_retry_notify,
        on_failure_callback=on_failure_notify
    )


def create_churn_analytic_classification_job(dag, cluster_config):
    """ Function to create a task to generate churn analytic classification result

    Args:
        dag (DAG type): DAG object representing the Airflow DAG definition.

    Return:
        DatabricksSubmitRunOperator: Airflow operator object.
    """
    # Specifications of the Databricks cluster for the task

    app_args = {
        "to_train": "false",
        "output_dir": '{{ var.value.churn_analytic_classification_output_bucket }}'
                      '/effective_date={{ ds_nodash }}'
    }

    parameters = databricks_spark_submit_cmd(
        name="churn_analytic_classification for {{ ds }}",
        spark_args={"spark.driver.maxResultSize": "200g"},
        py_files=("{{ var.value.customeranalytics_egg_location }}"),
        py_driver_file=("{{ var.value.artifacts_location }}"
                        "/{{ var.value.ssc_python_driver }}"),
        app_cmd_args=[
            "customeranalytics.cli",
            "churn_analytic",
            "contract_churn_classification"],
        app_args=app_args
    )

    return DatabricksSubmitRunOperator(
        task_id="churn_analytic_classification",
        trigger_rule="none_failed_or_skipped",
        dag=dag,
        new_cluster=cluster_config,
        json={
            "spark_submit_task": {
                "parameters": parameters
            }
        },
        on_success_callback=on_success_notify,
        on_retry_callback=on_retry_notify,
        on_failure_callback=on_failure_notify
    )


# Change training frequency to once/month using BranchPythonOperator
DS_NODASH_STR = '{{ ds_nodash }}'


def _check_skip_training():
    """
    return the task_id of the following task
    """
    if ((DS_NODASH_STR[-2:] >= "01") and (DS_NODASH_STR[-2:] <= "07")):
        return "churn_analytic_training"  # Retrain once/month
    else:
        return "skip_training"            # Skip training


# DummyOperator does nothing
skip_training = DummyOperator(
    task_id="skip_training",
    dag=dag
)

check_skip_training = BranchPythonOperator(
    task_id="check_skip_training",
    dag=dag,
    python_callable=_check_skip_training,
    do_xcom_push=False
)


churn_analytic_training = create_churn_analytic_training_job(dag, cluster_config)
churn_analytic_classification = create_churn_analytic_classification_job(dag, cluster_config)

check_skip_training >> [churn_analytic_training, skip_training] >> churn_analytic_classification
