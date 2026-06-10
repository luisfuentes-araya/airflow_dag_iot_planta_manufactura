from datetime import datetime
from io import StringIO

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
import pandas as pd
from sqlalchemy import create_engine

MYSQL_CONN = "mysql+pymysql://iotuser:Iot2026%21%40%23@172.26.83.206:3306/planta_manufactura"

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
    now = datetime.now()
    df = pd.DataFrame({
        "maquina_id": ["M01", "M01", "M02"],
        "fecha_hora": [now, now, now],
        "temperatura": [78, 92, 65],
        "vibracion": [0.3, 1.8, 0.2],
    })
    context["ti"].xcom_push(key="raw_data", value=df.to_json(date_format="iso"))

def validar(**context):
    raw = context["ti"].xcom_pull(task_ids="ingesta", key="raw_data")
    df = pd.read_json(StringIO(raw), convert_dates=["fecha_hora"])
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
    df = df[(df["temperatura"] > 0) & (df["temperatura"] < 150)]
    df = df[(df["vibracion"] > 0) & (df["vibracion"] < 5)]
    context["ti"].xcom_push(key="clean_data", value=df.to_json(date_format="iso"))

def cargar(**context):
    raw = context["ti"].xcom_pull(task_ids="validar", key="clean_data")
    df = pd.read_json(StringIO(raw), convert_dates=["fecha_hora"])
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])

    engine = create_engine(MYSQL_CONN)

    lecturas_sensores = df[["maquina_id", "fecha_hora", "temperatura", "vibracion"]].copy()

    lecturas_procesadas = df[["maquina_id", "fecha_hora", "temperatura", "vibracion"]].copy()
    lecturas_procesadas["lectura_id"] = range(1, len(lecturas_procesadas) + 1)

    predicciones_ia = pd.DataFrame({
        "sensor": df["maquina_id"],
        "prediccion": ["sin_falla" if v < 1 else "posible_falla" for v in df["vibracion"]],
        "probabilidad": [round(min(0.99, 0.40 + (v * 0.30)), 2) for v in df["vibracion"]],
    })

    anomalias = pd.DataFrame({
        "sensor": df["maquina_id"],
        "tipo_anomalia": [
            "vibracion_alta" if v >= 1 else "normal" for v in df["vibracion"]
        ],
        "valor_detectado": df["vibracion"],
        "detalle": [
            "Vibracion sobre el umbral" if v >= 1 else "Sin anomalía" for v in df["vibracion"]
        ],
    })

    alertas = pd.DataFrame({
        "maquina_id": df["maquina_id"],
        "fecha_hora": df["fecha_hora"],
        "posible_falla": ["Si" if v >= 1 else "No" for v in df["vibracion"]],
        "razon": [
            "Vibracion sobre el umbral" if v >= 1 else "Valor normal" for v in df["vibracion"]
        ],
    })

    dashboard_updates = pd.DataFrame({
        "mensaje": [
            f"Maquina {m} con vibracion {v}" for m, v in zip(df["maquina_id"], df["vibracion"])
        ]
    })

    lecturas_sensores.to_sql("lecturas_sensores", engine, if_exists="append", index=False)
    lecturas_procesadas.to_sql("lecturas_procesadas", engine, if_exists="append", index=False)
    predicciones_ia.to_sql("predicciones_ia", engine, if_exists="append", index=False)
    anomalias.to_sql("anomalias", engine, if_exists="append", index=False)
    alertas.to_sql("alertas", engine, if_exists="append", index=False)
    dashboard_updates.to_sql("dashboard_updates", engine, if_exists="append", index=False)

task1 = PythonOperator(task_id="ingesta", python_callable=ingesta, dag=dag)
task2 = PythonOperator(task_id="validar", python_callable=validar, dag=dag)
task3 = PythonOperator(task_id="cargar", python_callable=cargar, dag=dag)

task1 >> task2 >> task3
