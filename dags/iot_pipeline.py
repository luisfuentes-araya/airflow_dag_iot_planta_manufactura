from datetime import datetime
from io import StringIO
import os

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sqlalchemy import create_engine

MYSQL_CONN = "mysql+pymysql://iotuser:Iot2026%21%40%23@172.26.83.206:3306/planta_manufactura"

# ----------------------------------------------------
# Carpeta donde se guardarán los CSV para Power BI
# Cámbiala si quieres guardarlos en otro lugar
# ----------------------------------------------------
CSV_OUTPUT_DIR = "/opt/airflow/exports/powerbi"

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

    X = historico_df[
        ["temperatura", "vibracion"]
    ]

    y = historico_df["estado"]

    modelo = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    modelo.fit(X, y)

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

    predicciones_ia = predicciones_ia[
        ["sensor", "prediccion", "temperatura", "vibracion", "probabilidad", "motivo"]
    ]

    predicciones_ia.to_sql(
        "predicciones_ia",
        engine,
        if_exists="append",
        index=False,
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


# ----------------------------------------------------
# TASK 5 - EXPORTAR CSV PARA POWER BI  ← NUEVA
# ----------------------------------------------------

def exportar_metricas(**context):
    """
    Lee la tabla predicciones_ia desde MySQL, recalcula las métricas
    de clasificación y exporta 3 CSV listos para conectar en Power BI.

    Archivos generados en CSV_OUTPUT_DIR:
      - metricas_clasificacion.csv  → tarjetas KPI
      - predicciones_detalle.csv    → tabla de detalle por máquina
      - matriz_confusion.csv        → matriz de confusión
    """

    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

    engine = create_engine(MYSQL_CONN)

    df = pd.read_sql(
        "SELECT sensor, temperatura, vibracion, prediccion, probabilidad FROM predicciones_ia",
        engine
    )

    if df.empty:
        print("No hay datos en predicciones_ia todavía.")
        return

    df["clase_real"] = df["prediccion"].map({
        "Normal":          0,
        "Riesgo":          0,
        "Falla probable":  1,
        "Revisar urgente": 1,
    })

    historico = []
    for temperatura in range(1, 151):
        for vibracion in range(1, 6):
            if temperatura >= 141 or vibracion == 5:
                clase = 1
            elif temperatura >= 130 or vibracion == 4:
                clase = 1
            elif temperatura >= 80 or vibracion == 3:
                clase = 0
            else:
                clase = 0
            historico.append([temperatura, vibracion, clase])

    historico_df = pd.DataFrame(
        historico,
        columns=["temperatura", "vibracion", "clase_bin"]
    )

    X_hist = historico_df[["temperatura", "vibracion"]]
    y_hist = historico_df["clase_bin"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_hist, y_hist,
        test_size=0.25,
        random_state=42,
        stratify=y_hist
    )

    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_prob)

    fecha_calculo = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # CSV 1 — Tarjetas KPI
    metricas_df = pd.DataFrame({
        "Metrica": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
        "Valor": [
            round(accuracy, 4),
            round(precision, 4),
            round(recall, 4),
            round(f1, 4),
            round(roc_auc, 4),
        ],
        "Porcentaje": [
            f"{v * 100:.1f}%"
            for v in [accuracy, precision, recall, f1, roc_auc]
        ],
        "Descripcion": [
            "Predicciones correctas sobre el total",
            "De las predichas como falla, cuantas eran falla real",
            "De todas las fallas reales, cuantas detecto el modelo",
            "Balance entre Precision y Recall",
            "Capacidad de separar falla vs no_falla",
        ],
        "Fecha_calculo": [fecha_calculo] * 5,
    })

    path_metricas = os.path.join(CSV_OUTPUT_DIR, "metricas_clasificacion.csv")
    metricas_df.to_csv(path_metricas, index=False)
    print(f"Exportado: {path_metricas}")

    # CSV 2 — Detalle de predicciones
    detalle_df = X_test.copy()
    detalle_df["clase_real"]           = y_test.values
    detalle_df["clase_predicha"]       = y_pred
    detalle_df["probabilidad_falla"]   = y_prob.round(4)
    detalle_df["clase_real_texto"]     = detalle_df["clase_real"].map({0: "no_falla", 1: "falla"})
    detalle_df["clase_predicha_texto"] = detalle_df["clase_predicha"].map({0: "no_falla", 1: "falla"})
    detalle_df["correcto"]             = (
        detalle_df["clase_real"] == detalle_df["clase_predicha"]
    ).map({True: "Si", False: "No"})
    detalle_df["tipo_resultado"] = [
        "Verdadero Positivo" if r == 1 and p == 1 else
        "Verdadero Negativo" if r == 0 and p == 0 else
        "Falso Positivo"     if r == 0 and p == 1 else
        "Falso Negativo"
        for r, p in zip(y_test.values, y_pred)
    ]
    detalle_df["fecha_calculo"] = fecha_calculo

    path_detalle = os.path.join(CSV_OUTPUT_DIR, "predicciones_detalle.csv")
    detalle_df.to_csv(path_detalle, index=False)
    print(f"Exportado: {path_detalle}")

    # CSV 3 — Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame({
        "Real":     ["no_falla", "no_falla", "falla",    "falla"],
        "Predicho": ["no_falla", "falla",    "no_falla", "falla"],
        "Cantidad": [cm[0, 0],   cm[0, 1],   cm[1, 0],  cm[1, 1]],
        "Tipo":     [
            "Verdadero Negativo",
            "Falso Positivo",
            "Falso Negativo",
            "Verdadero Positivo",
        ],
        "Fecha_calculo": [fecha_calculo] * 4,
    })

    path_cm = os.path.join(CSV_OUTPUT_DIR, "matriz_confusion.csv")
    cm_df.to_csv(path_cm, index=False)
    print(f"Exportado: {path_cm}")

    print(f"\nMetricas al {fecha_calculo}:")
    print(metricas_df[["Metrica", "Valor", "Porcentaje"]].to_string(index=False))


# ----------------------------------------------------
# DEFINIR TASKS
# ----------------------------------------------------

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

task5 = PythonOperator(
    task_id="exportar_metricas",
    python_callable=exportar_metricas,
    dag=dag
)

# ----------------------------------------------------
# FLUJO: ingesta → validar → predecir → cargar → exportar_metricas
# ----------------------------------------------------

task1 >> task2 >> task3 >> task4 >> task5
