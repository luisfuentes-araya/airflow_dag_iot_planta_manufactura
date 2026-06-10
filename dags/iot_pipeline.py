from datetime import datetime
from io import StringIO

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
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


# ----------------------------------------------------
# TASK 1 - INGESTA
# ----------------------------------------------------

def ingesta(**context):

    now = datetime.now()

    df = pd.DataFrame({
        "maquina_id": ["M01", "M02", "M03", "M04", "M05"],
        "fecha_hora": [now] * 5,
        "temperatura": [45, 95, 132, 145, 78],
        "vibracion": [1, 3, 4, 5, 2],
    })

    context["ti"].xcom_push(
        key="raw_data",
        value=df.to_json(date_format="iso")
    )


# ----------------------------------------------------
# TASK 2 - VALIDAR
# ----------------------------------------------------

def validar(**context):

    raw = context["ti"].xcom_pull(
        task_ids="ingesta",
        key="raw_data"
    )

    df = pd.read_json(
        StringIO(raw),
        convert_dates=["fecha_hora"]
    )

    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])

    df = df[
        (df["temperatura"] > 0)
        & (df["temperatura"] <= 150)
    ]

    df = df[
        (df["vibracion"] > 0)
        & (df["vibracion"] <= 5)
    ]

    context["ti"].xcom_push(
        key="clean_data",
        value=df.to_json(date_format="iso")
    )


# ----------------------------------------------------
# TASK 3 - MACHINE LEARNING
# ----------------------------------------------------

def predecir(**context):

    raw = context["ti"].xcom_pull(
        task_ids="validar",
        key="clean_data"
    )

    df = pd.read_json(
        StringIO(raw),
        convert_dates=["fecha_hora"]
    )

    # --------------------------
    # Dataset histórico sintético
    # --------------------------

    historico = []

    for temperatura in range(1, 151):

        for vibracion in range(1, 6):

            if temperatura >= 141 or vibracion == 5:
                estado = "Revisar urgente"

            elif temperatura >= 130 or vibracion == 4:
                estado = "Falla probable"

            elif temperatura >= 80 or vibracion == 3:
                estado = "Riesgo"

            else:
                estado = "Normal"

            historico.append([
                temperatura,
                vibracion,
                estado
            ])

    historico_df = pd.DataFrame(
        historico,
        columns=[
            "temperatura",
            "vibracion",
            "estado"
        ]
    )

    # --------------------------
    # Entrenamiento
    # --------------------------

    X = historico_df[
        ["temperatura", "vibracion"]
    ]

    y = historico_df["estado"]

    modelo = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    modelo.fit(X, y)

    # --------------------------
    # Predicción
    # --------------------------

    X_pred = df[
        ["temperatura", "vibracion"]
    ]

    predicciones = modelo.predict(X_pred)

    probabilidades = modelo.predict_proba(X_pred)

    resultado = df.copy()

    resultado["estado_predicho"] = predicciones

    resultado["probabilidad"] = [
        round(max(p), 2)
        for p in probabilidades
    ]

    context["ti"].xcom_push(
        key="predicciones",
        value=resultado.to_json(date_format="iso")
    )


# ----------------------------------------------------
# TASK 4 - CARGAR
# ----------------------------------------------------

def cargar(**context):

    raw = context["ti"].xcom_pull(
        task_ids="predecir",
        key="predicciones"
    )

    df = pd.read_json(
        StringIO(raw),
        convert_dates=["fecha_hora"]
    )

    engine = create_engine(MYSQL_CONN)

    lecturas_sensores = df[
        [
            "maquina_id",
            "fecha_hora",
            "temperatura",
            "vibracion"
        ]
    ].copy()

    lecturas_procesadas = lecturas_sensores.copy()

    lecturas_procesadas["lectura_id"] = range(
        1,
        len(lecturas_procesadas) + 1
    )

    predicciones_ia = pd.DataFrame({
        "sensor": df["maquina_id"],
        "temperatura": df["temperatura"],
        "vibracion": df["vibracion"],
        "prediccion": df["estado_predicho"],
        "probabilidad": df["probabilidad"],
        "motivo": [
            (
                "Valores dentro del rango normal"
                if estado == "Normal"
                else "Valores en zona de riesgo"
                if estado == "Riesgo"
                else "Temperatura o vibracion elevadas"
                if estado == "Falla probable"
                else "Valores criticos, revisar maquina urgente"
            )
            for estado in df["estado_predicho"]
        ]
    })

    anomalias = pd.DataFrame({
        "sensor": df["maquina_id"],
        "tipo_anomalia": df["estado_predicho"],
        "valor_detectado": df["vibracion"],
        "detalle": [
            f"Temperatura {t} / Vibracion {v}"
            for t, v in zip(
                df["temperatura"],
                df["vibracion"]
            )
        ]
    })

    alertas = pd.DataFrame({
        "maquina_id": df["maquina_id"],
        "fecha_hora": df["fecha_hora"],
        "posible_falla": [
            "Si"
            if estado in [
                "Falla probable",
                "Revisar urgente"
            ]
            else "No"
            for estado in df["estado_predicho"]
        ],
        "razon": df["estado_predicho"]
    })

    dashboard_updates = pd.DataFrame({
        "mensaje": [
            f"Maquina {m}: {estado}"
            for m, estado in zip(
                df["maquina_id"],
                df["estado_predicho"]
            )
        ]
    })

    lecturas_sensores.to_sql(
        "lecturas_sensores",
        engine,
        if_exists="append",
        index=False
    )

    lecturas_procesadas.to_sql(
        "lecturas_procesadas",
        engine,
        if_exists="append",
        index=False
    )

    predicciones_ia.to_sql(
        "predicciones_ia",
        engine,
        if_exists="append",
        index=False,
        columns=["sensor", "prediccion", "temperatura", "vibracion", "probabilidad", "motivo"]
    )

    anomalias.to_sql(
        "anomalias",
        engine,
        if_exists="append",
        index=False
    )

    alertas.to_sql(
        "alertas",
        engine,
        if_exists="append",
        index=False
    )

    dashboard_updates.to_sql(
        "dashboard_updates",
        engine,
        if_exists="append",
        index=False
    )


task1 = PythonOperator(
    task_id="ingesta",
    python_callable=ingesta,
    dag=dag
)

task2 = PythonOperator(
    task_id="validar",
    python_callable=validar,
    dag=dag
)

task3 = PythonOperator(
    task_id="predecir",
    python_callable=predecir,
    dag=dag
)

task4 = PythonOperator(
    task_id="cargar",
    python_callable=cargar,
    dag=dag
)

task1 >> task2 >> task3 >> task4
