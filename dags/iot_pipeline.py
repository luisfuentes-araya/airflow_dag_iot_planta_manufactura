from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
from sqlalchemy import create_engine

MYSQL_CONN = "mysql+pymysql://iotuser:Duoc2026@172.26.83.206:3306/planta_manufactura"

default_args = {
    "owner": "equipo3",
    "start_date": datetime(2026, 5, 12),
}

dag = DAG(
    dag_id="iot_pipeline",
    default_args=default_args,
    schedule="@hourly",
    catchup=False,
)

def ingesta(**context):
    df = pd.DataFrame({
        "maquina_id": ["M01", "M01", "M02"],
        "fecha_hora": [datetime.now(), datetime.now(), datetime.now()],
        "temperatura": [78, 92, 65],
        "vibracion": [0.3, 1.8, 0.2],
    })
    context["ti"].xcom_push(key="raw_data", value=df.to_json())

def validar(**context):
    df = pd.read_json(context["ti"].xcom_pull(key="raw_data"))
    df = df[(df["temperatura"] > 0) & (df["temperatura"] < 150)]
    df = df[(df["vibracion"] > 0) & (df["vibracion"] < 5)]
    context["ti"].xcom_push(key="clean_data", value=df.to_json())

def cargar(**context):
    df = pd.read_json(context["ti"].xcom_pull(key="clean_data"))
    engine = create_engine(MYSQL_CONN)
    df.to_sql("lecturas_sensores", engine, if_exists="append", index=False)

task1 = PythonOperator(task_id="ingesta", python_callable=ingesta, dag=dag)
task2 = PythonOperator(task_id="validar", python_callable=validar, dag=dag)
task3 = PythonOperator(task_id="cargar", python_callable=cargar, dag=dag)

task1 >> task2 >> task3
