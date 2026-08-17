import streamlit as st
import pandas as pd
import numpy as np
import io
import warnings

warnings.filterwarnings("ignore")


# =========================================================
# IMPORTS OPCIONALES
# =========================================================

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

try:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False


# =========================================================
# CONFIGURACION STREAMLIT
# =========================================================

st.set_page_config(
    page_title="Forecast automatico de ventas",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Forecast automatico de ventas")
st.caption(
    "Aplicacion para seleccionar automaticamente el mejor modelo de prevision "
    "mediante backtesting walk-forward y generar una prevision de 18 meses."
)


# =========================================================
# FUNCIONES DE LIMPIEZA Y LECTURA
# =========================================================

def clean_number(value):
    """
    Convierte numeros pegados desde Excel.
    Soporta:
    1.234,56
    1234,56
    1,234.56
    1234.56
    """

    if pd.isna(value):
        return np.nan

    text = str(value).strip().replace(" ", "")

    if text == "":
        return np.nan

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "")
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text and "." not in text:
        text = text.replace(",", ".")

    try:
        return float(text)
    except Exception:
        return np.nan


def looks_like_date(value):
    text = str(value).strip()

    if text == "":
        return False

    try:
        parsed = pd.to_datetime(text, errors="coerce")
        return not pd.isna(parsed)
    except Exception:
        return False


def parse_pasted_data(raw_text, start_month):
    """
    Permite dos formatos:
    1. Fecha + Ventas
    2. Solo Ventas

    Ejemplo recomendado desde Excel:
    Fecha    Ventas
    2025-01  1200
    2025-02  1350
    """

    if raw_text is None or raw_text.strip() == "":
        raise ValueError("No se han introducido datos.")

    raw_text = raw_text.strip()

    df = None

    # 1. Intentar tabulador, formato habitual al pegar desde Excel
    try:
        temp = pd.read_csv(io.StringIO(raw_text), sep="\t", header=None, engine="python")
        if temp.shape[1] >= 2:
            df = temp
    except Exception:
        pass

    # 2. Intentar punto y coma
    if df is None:
        try:
            temp = pd.read_csv(io.StringIO(raw_text), sep=";", header=None, engine="python")
            if temp.shape[1] >= 2:
                df = temp
        except Exception:
            pass

    # 3. Intentar coma solo si parece Fecha + Ventas
    if df is None:
        try:
            temp = pd.read_csv(io.StringIO(raw_text), sep=",", header=None, engine="python")
            if temp.shape[1] >= 2 and looks_like_date(temp.iloc[0, 0]):
                df = temp
        except Exception:
            pass

    # 4. Si no, tratar como una sola columna
    if df is None:
        lines = raw_text.splitlines()
        df = pd.DataFrame(lines)

    df = df.dropna(how="all").reset_index(drop=True)

    if df.empty:
        raise ValueError("No se han podido interpretar los datos pegados.")

    first_row = " ".join([str(x).lower() for x in df.iloc[0].values])

    if any(w in first_row for w in ["fecha", "mes", "venta", "ventas", "cantidad", "sales"]):
        df = df.iloc[1:].reset_index(drop=True)

    if df.empty:
        raise ValueError("Despues de quitar la cabecera, no quedan datos validos.")

    if df.shape[1] >= 2:
        date_text = df.iloc[:, 0].astype(str).str.strip()
        values = df.iloc[:, 1].apply(clean_number)

        dates = pd.to_datetime(date_text, errors="coerce")

        if dates.isna().all():
            parsed_dates = []

            for x in date_text:
                x = str(x).strip()
                x = x.replace("/", "-").replace(".", "-")

                try:
                    if len(x) == 6 and x.isdigit():
                        parsed_dates.append(pd.to_datetime(x + "01", format="%Y%m%d"))
                    elif len(x) == 7:
                        parsed_dates.append(pd.to_datetime(x + "-01"))
                    else:
                        parsed_dates.append(pd.to_datetime(x))
                except Exception:
                    parsed_dates.append(pd.NaT)

            dates = pd.Series(parsed_dates)

        result = pd.DataFrame({
            "Fecha": dates,
            "Ventas": values
        })

    else:
        values = df.iloc[:, 0].apply(clean_number)
        start_date = pd.to_datetime(start_month + "-01")
        dates = pd.date_range(start=start_date, periods=len(values), freq="MS")

        result = pd.DataFrame({
            "Fecha": dates,
            "Ventas": values
        })

    result = result.dropna(subset=["Fecha", "Ventas"]).copy()
    result["Fecha"] = pd.to_datetime(result["Fecha"]).dt.to_period("M").dt.to_timestamp()
    result["Ventas"] = result["Ventas"].astype(float)

    result = result.sort_values("Fecha")
    result = result.drop_duplicates(subset=["Fecha"], keep="last")
    result = result.reset_index(drop=True)

    if result.empty:
        raise ValueError("No se han podido interpretar los datos pegados.")

    if (result["Ventas"] < 0).any():
        raise ValueError("Existen ventas negativas. Revisa si son devoluciones o errores.")

    return result


def complete_monthly_series(df):
    """
    Completa meses faltantes con cero.
    """

    full_dates = pd.date_range(
        start=df["Fecha"].min(),
        end=df["Fecha"].max(),
        freq="MS"
    )

    full_df = pd.DataFrame({"Fecha": full_dates})
    merged = full_df.merge(df, on="Fecha", how="left")
    merged["Ventas"] = merged["Ventas"].fillna(0)

    return merged


# =========================================================
# METRICAS
# =========================================================

def safe_mape(actual, forecast):
    actual = np.array(actual, dtype=float)
    forecast = np.array(forecast, dtype=float)

    mask = actual != 0

    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100


def smape(actual, forecast):
    actual = np.array(actual, dtype=float)
    forecast = np.array(forecast, dtype=float)

    denominator = (np.abs(actual) + np.abs(forecast)) / 2
    mask = denominator != 0

    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs(actual[mask] - forecast[mask]) / denominator[mask]) * 100


def wmape(actual, forecast):
    actual = np.array(actual, dtype=float)
    forecast = np.array(forecast, dtype=float)

    denominator = np.sum(np.abs(actual))

    if denominator == 0:
        return np.nan

    return np.sum(np.abs(actual - forecast)) / denominator * 100


def postprocess_forecast(values):
    """
    Evita negativos y redondea a unidades.
    """

    arr = np.array(values, dtype=float)
    arr = np.where(np.isnan(arr), 0, arr)
    arr = np.maximum(arr, 0)

    return np.round(arr, 0).astype(int)


def reliability_label(metric_value):
    if pd.isna(metric_value):
        return "No calculable"

    if metric_value <= 15:
        return "Alta"
    elif metric_value <= 30:
        return "Media"
    elif metric_value <= 50:
        return "Baja"
    else:
        return "Muy baja"


# =========================================================
# MODELOS DE FORECASTING
# =========================================================

def forecast_naive(y, horizon):
    return np.repeat(y.iloc[-1], horizon)


def forecast_moving_average(y, horizon, window):
    if len(y) < window:
        raise ValueError(f"No hay suficientes datos para media movil {window}.")

    return np.repeat(y.iloc[-window:].mean(), horizon)


def forecast_seasonal_naive_12(y, horizon):
    if len(y) < 12:
        raise ValueError("Naive estacional necesita al menos 12 meses.")

    last_12 = y.iloc[-12:].values

    return np.array([last_12[i % 12] for i in range(horizon)])


def forecast_ses(y, horizon):
    if not STATSMODELS_AVAILABLE:
        raise ValueError("statsmodels no esta instalado.")

    model = SimpleExpSmoothing(
        y,
        initialization_method="estimated"
    ).fit(optimized=True)

    return model.forecast(horizon).values


def forecast_holt(y, horizon, damped=False):
    if not STATSMODELS_AVAILABLE:
        raise ValueError("statsmodels no esta instalado.")

    model = Holt(
        y,
        damped_trend=damped,
        initialization_method="estimated"
    ).fit(optimized=True)

    return model.forecast(horizon).values


def forecast_holt_winters(y, horizon):
    if not STATSMODELS_AVAILABLE:
        raise ValueError("statsmodels no esta instalado.")

    if len(y) < 24:
        raise ValueError("Holt-Winters estacional necesita al menos 24 meses.")

    model = ExponentialSmoothing(
        y,
        trend="add",
        seasonal="add",
        seasonal_periods=12,
        initialization_method="estimated"
    ).fit(optimized=True)

    return model.forecast(horizon).values


def forecast_arima_simple(y, horizon):
    if not STATSMODELS_AVAILABLE:
        raise ValueError("statsmodels no esta instalado.")

    candidate_orders = [
        (0, 1, 0),
        (1, 1, 0),
        (0, 1, 1),
        (1, 1, 1),
        (2, 1, 1)
    ]

    best_model = None
    best_aic = np.inf

    for order in candidate_orders:
        try:
            model = SARIMAX(
                y,
                order=order,
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)

            if model.aic < best_aic:
                best_aic = model.aic
                best_model = model

        except Exception:
            continue

    if best_model is None:
        raise ValueError("ARIMA no ha podido ajustarse.")

    return best_model.forecast(horizon).values


def forecast_prophet(y, dates, horizon):
    if not PROPHET_AVAILABLE:
        raise ValueError("Prophet no esta instalado.")

    df_prophet = pd.DataFrame({
        "ds": pd.to_datetime(dates),
        "y": y.values
    })

    yearly = True if len(y) >= 24 else False

    configs = [
        {
            "seasonality_mode": "additive",
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 5.0
        },
        {
            "seasonality_mode": "additive",
            "changepoint_prior_scale": 0.10,
            "seasonality_prior_scale": 10.0
        },
        {
            "seasonality_mode": "multiplicative",
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 5.0
        }
    ]

    last_error = None

    for cfg in configs:
        try:
            model = Prophet(
                yearly_seasonality=yearly,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode=cfg["seasonality_mode"],
                changepoint_prior_scale=cfg["changepoint_prior_scale"],
                seasonality_prior_scale=cfg["seasonality_prior_scale"],
                interval_width=0.80
            )

            model.fit(df_prophet)

            future = model.make_future_dataframe(
                periods=horizon,
                freq="MS",
                include_history=True
            )

            forecast = model.predict(future)

            return forecast.tail(horizon)["yhat"].values

        except Exception as e:
            last_error = e

    raise ValueError(f"Prophet no ha podido ajustarse: {last_error}")


def make_lgbm_training_table(values, dates):
    df = pd.DataFrame({
        "Fecha": pd.Series(pd.to_datetime(dates)).reset_index(drop=True),
        "y": pd.Series(values).reset_index(drop=True)
    })

    df["lag_1"] = df["y"].shift(1)
    df["lag_2"] = df["y"].shift(2)
    df["lag_3"] = df["y"].shift(3)
    df["rolling_mean_3"] = df["y"].shift(1).rolling(3).mean()
    df["rolling_mean_6"] = df["y"].shift(1).rolling(6).mean()
    df["month"] = df["Fecha"].dt.month
    df["trend"] = np.arange(len(df))

    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "lag_1",
        "lag_2",
        "lag_3",
        "rolling_mean_3",
        "rolling_mean_6",
        "month",
        "trend"
    ]

    return df, feature_cols


def forecast_lgbm(y, dates, horizon):
    if not LGBM_AVAILABLE:
        raise ValueError("LightGBM no esta instalado.")

    if len(y) < 12:
        raise ValueError("LightGBM necesita al menos 12 meses.")

    hist_dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    hist_values = pd.Series(y.values).reset_index(drop=True)

    train_df, feature_cols = make_lgbm_training_table(hist_values, hist_dates)

    if len(train_df) < 5:
        raise ValueError("No hay suficientes observaciones entrenables para LightGBM.")

    X_train = train_df[feature_cols]
    y_train = train_df["y"]

    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=7,
        max_depth=3,
        min_child_samples=1,
        random_state=42,
        verbose=-1
    )

    model.fit(X_train, y_train)

    current_values = list(hist_values.values)
    last_date = hist_dates.iloc[-1]

    predictions = []

    for step in range(1, horizon + 1):
        next_date = last_date + pd.DateOffset(months=step)

        lag_1 = current_values[-1]
        lag_2 = current_values[-2] if len(current_values) >= 2 else current_values[-1]
        lag_3 = current_values[-3] if len(current_values) >= 3 else current_values[-1]

        rolling_mean_3 = np.mean(current_values[-3:]) if len(current_values) >= 3 else np.mean(current_values)
        rolling_mean_6 = np.mean(current_values[-6:]) if len(current_values) >= 6 else np.mean(current_values)

        row = pd.DataFrame([{
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "rolling_mean_3": rolling_mean_3,
            "rolling_mean_6": rolling_mean_6,
            "month": next_date.month,
            "trend": len(current_values)
        }])

        pred = float(model.predict(row[feature_cols])[0])
        pred = max(pred, 0)

        predictions.append(pred)
        current_values.append(pred)

    return np.array(predictions)


# =========================================================
# CATALOGO OFICIAL DE MODELOS
# =========================================================

def available_models(n_months):
    """
    Modelos incluidos en la herramienta:
    1. Naive
    2. Media movil 3/6
    3. Suavizado exponencial
    4. Holt
    5. ARIMA simple
    6. Prophet
    7. LightGBM
    8. Estacionales si hay >=24 meses
    """

    models = [
        "Naive",
        "Media movil 3 meses",
        "Media movil 6 meses",
        "Suavizado exponencial",
        "Holt",
        "Holt amortiguado",
        "ARIMA simple",
        "Prophet",
        "LightGBM"
    ]

    if n_months >= 24:
        models.extend([
            "Naive estacional 12 meses",
            "Holt-Winters estacional"
        ])

    return models


def run_model(model_name, y, dates, horizon):
    if model_name == "Naive":
        return forecast_naive(y, horizon)

    if model_name == "Media movil 3 meses":
        return forecast_moving_average(y, horizon, 3)

    if model_name == "Media movil 6 meses":
        return forecast_moving_average(y, horizon, 6)

    if model_name == "Suavizado exponencial":
        return forecast_ses(y, horizon)

    if model_name == "Holt":
        return forecast_holt(y, horizon, damped=False)

    if model_name == "Holt amortiguado":
        return forecast_holt(y, horizon, damped=True)

    if model_name == "ARIMA simple":
        return forecast_arima_simple(y, horizon)

    if model_name == "Prophet":
        return forecast_prophet(y, dates, horizon)

    if model_name == "LightGBM":
        return forecast_lgbm(y, dates, horizon)

    if model_name == "Naive estacional 12 meses":
        return forecast_seasonal_naive_12(y, horizon)

    if model_name == "Holt-Winters estacional":
        return forecast_holt_winters(y, horizon)

    raise ValueError(f"Modelo no reconocido: {model_name}")


def complexity_rank(model_name):
    """
    Para desempates.
    Si dos modelos tienen metrica similar, se prioriza el modelo mas simple.
    """

    ranks = {
        "Naive": 1,
        "Media movil 3 meses": 2,
        "Media movil 6 meses": 3,
        "Naive estacional 12 meses": 4,
        "Suavizado exponencial": 5,
        "Holt": 6,
        "Holt amortiguado": 7,
        "Holt-Winters estacional": 8,
        "ARIMA simple": 9,
        "Prophet": 10,
        "LightGBM": 11
    }

    return ranks.get(model_name, 99)


def minimum_training_months_for_model(model_name):
    """
    Meses minimos de entrenamiento necesarios por modelo.
    Se usa para decidir si un modelo puede participar en el backtesting walk-forward.
    """

    minimums = {
        "Naive": 1,
        "Media movil 3 meses": 3,
        "Media movil 6 meses": 6,
        "Suavizado exponencial": 6,
        "Holt": 8,
        "Holt amortiguado": 8,
        "ARIMA simple": 10,
        "Prophet": 12,
        "LightGBM": 12,
        "Naive estacional 12 meses": 12,
        "Holt-Winters estacional": 24
    }

    return minimums.get(model_name, 12)


def backtest_candidate_models(n_months):
    """
    Selecciona los modelos que pueden participar en el backtesting.

    El primer entrenamiento usa n_months - 4 meses.
    Por tanto, solo participan modelos que tengan suficiente historico
    en ese primer corte.
    """

    first_train_size = n_months - 4
    candidates = []

    for model_name in available_models(n_months):
        min_months = minimum_training_months_for_model(model_name)

        if first_train_size >= min_months:
            candidates.append(model_name)

    return candidates


# =========================================================
# BACKTESTING WALK-FORWARD Y FORECAST FINAL
# =========================================================

def run_backtesting(df):
    """
    Backtesting correcto tipo walk-forward.

    Procedimiento:
    1. Se reservan los ultimos 4 meses como test.
    2. Para cada uno de esos 4 meses:
       - se entrena con todos los datos reales disponibles hasta el mes anterior;
       - se predice solo 1 mes hacia adelante;
       - se compara contra el valor real.
    3. Se calculan MAPE, sMAPE y WMAPE sobre las 4 predicciones.
    4. Se selecciona el modelo con menor error.
    """

    n_months = len(df)

    if n_months < 16:
        raise ValueError("Para hacer backtesting 12+4 se necesitan al menos 16 meses.")

    test_start_index = n_months - 4
    test = df.iloc[test_start_index:].copy()
    y_test = test["Ventas"].values

    rows = []

    models_to_test = backtest_candidate_models(n_months)

    for model_name in models_to_test:

        backtest_predictions = []
        model_failed = False
        error_message = ""

        for i in range(test_start_index, n_months):

            try:
                # Entrenamiento expandido hasta el mes anterior al que se quiere predecir
                train_i = df.iloc[:i].copy()

                y_train_i = train_i["Ventas"]
                dates_train_i = train_i["Fecha"]

                # Prediccion de solo 1 mes hacia adelante
                pred_i = run_model(
                    model_name=model_name,
                    y=y_train_i,
                    dates=dates_train_i,
                    horizon=1
                )

                pred_i = postprocess_forecast(pred_i)[0]
                backtest_predictions.append(pred_i)

            except Exception as e:
                model_failed = True
                error_message = str(e)
                break

        if model_failed:
            rows.append({
                "Modelo": model_name,
                "MAPE_%": np.nan,
                "sMAPE_%": np.nan,
                "WMAPE_%": np.nan,
                "Metrica seleccion": "",
                "Valor metrica seleccion": np.nan,
                "Backtest M+1": np.nan,
                "Backtest M+2": np.nan,
                "Backtest M+3": np.nan,
                "Backtest M+4": np.nan,
                "Estado": "Error",
                "Error": error_message
            })

            continue

        pred = np.array(backtest_predictions, dtype=float)

        model_mape = safe_mape(y_test, pred)
        model_smape = smape(y_test, pred)
        model_wmape = wmape(y_test, pred)

        if np.any(y_test == 0):
            selection_metric_name = "sMAPE"
            selection_metric_value = model_smape
        else:
            selection_metric_name = "MAPE"
            selection_metric_value = model_mape

        rows.append({
            "Modelo": model_name,
            "MAPE_%": model_mape,
            "sMAPE_%": model_smape,
            "WMAPE_%": model_wmape,
            "Metrica seleccion": selection_metric_name,
            "Valor metrica seleccion": selection_metric_value,
            "Backtest M+1": pred[0],
            "Backtest M+2": pred[1],
            "Backtest M+3": pred[2],
            "Backtest M+4": pred[3],
            "Estado": "OK",
            "Error": ""
        })

    results = pd.DataFrame(rows)

    valid = results[
        (results["Estado"] == "OK") &
        (~results["Valor metrica seleccion"].isna())
    ].copy()

    if valid.empty:
        raise ValueError("Ningun modelo ha podido calcularse correctamente en backtesting walk-forward.")

    valid["Complejidad"] = valid["Modelo"].apply(complexity_rank)

    valid = valid.sort_values(
        by=["Valor metrica seleccion", "Complejidad"],
        ascending=[True, True]
    )

    best_model = valid.iloc[0]["Modelo"]
    best_metric = valid.iloc[0]["Valor metrica seleccion"]
    best_metric_name = valid.iloc[0]["Metrica seleccion"]

    train = df.iloc[:test_start_index].copy()

    return best_model, best_metric, best_metric_name, results, train, test


def forecast_future(df, model_name, horizon):
    """
    Forecast futuro definitivo.
    Despues de seleccionar el mejor modelo, se reentrena con todo el historico disponible.
    """

    pred = run_model(
        model_name=model_name,
        y=df["Ventas"],
        dates=df["Fecha"],
        horizon=horizon
    )

    pred = postprocess_forecast(pred)

    future_dates = pd.date_range(
        start=df["Fecha"].max() + pd.DateOffset(months=1),
        periods=horizon,
        freq="MS"
    )

    return pd.DataFrame({
        "Fecha": future_dates,
        "Mes": [f"M+{i}" for i in range(1, horizon + 1)],
        "Forecast": pred
    })


def make_horizontal_forecast(forecast_df, forecast_type, model_name):
    row = {
        "Tipo forecast": forecast_type,
        "Modelo": model_name
    }

    for i, value in enumerate(forecast_df["Forecast"].values, start=1):
        row[f"M+{i}"] = int(value)

    return row


def build_chart(df, auto_forecast=None, manual_forecast=None):
    if not PLOTLY_AVAILABLE:
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Fecha"],
        y=df["Ventas"],
        mode="lines+markers",
        name="Historico real",
        line=dict(width=3)
    ))

    if auto_forecast is not None:
        fig.add_trace(go.Scatter(
            x=auto_forecast["Fecha"],
            y=auto_forecast["Forecast"],
            mode="lines+markers",
            name="Forecast automatico",
            line=dict(width=3)
        ))

    if manual_forecast is not None:
        fig.add_trace(go.Scatter(
            x=manual_forecast["Fecha"],
            y=manual_forecast["Forecast"],
            mode="lines+markers",
            name="Forecast modelo seleccionado",
            line=dict(width=3, dash="dash")
        ))

    fig.update_layout(
        title="Historico y prevision a futuro",
        xaxis_title="Fecha",
        yaxis_title="Ventas",
        template="plotly_white",
        height=520,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    return fig


# =========================================================
# ESTADO INICIAL
# =========================================================

example_data = """Fecha\tVentas
2025-01\t1200
2025-02\t1350
2025-03\t980
2025-04\t1450
2025-05\t1600
2025-06\t1580
2025-07\t1700
2025-08\t900
2025-09\t1250
2025-10\t1400
2025-11\t1550
2025-12\t1900
2026-01\t1300
2026-02\t1450
2026-03\t1500
2026-04\t1650"""

if "raw_data" not in st.session_state:
    st.session_state["raw_data"] = example_data

if "clear_counter" not in st.session_state:
    st.session_state["clear_counter"] = 0

if "result_ready" not in st.session_state:
    st.session_state["result_ready"] = False

if "df" not in st.session_state:
    st.session_state["df"] = None

if "results_df" not in st.session_state:
    st.session_state["results_df"] = None

if "best_model" not in st.session_state:
    st.session_state["best_model"] = None

if "best_metric" not in st.session_state:
    st.session_state["best_metric"] = None

if "best_metric_name" not in st.session_state:
    st.session_state["best_metric_name"] = None

if "train_df" not in st.session_state:
    st.session_state["train_df"] = None

if "test_df" not in st.session_state:
    st.session_state["test_df"] = None

if "mode" not in st.session_state:
    st.session_state["mode"] = None


def clear_history():
    st.session_state["raw_data"] = ""
    st.session_state["clear_counter"] += 1
    st.session_state["result_ready"] = False
    st.session_state["df"] = None
    st.session_state["results_df"] = None
    st.session_state["best_model"] = None
    st.session_state["best_metric"] = None
    st.session_state["best_metric_name"] = None
    st.session_state["train_df"] = None
    st.session_state["test_df"] = None
    st.session_state["mode"] = None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("⚙️ Configuracion")

    start_month = st.text_input(
        "Mes inicial si pegas solo ventas",
        value="2025-01",
        help="Formato recomendado: YYYY-MM"
    )

    horizon = st.number_input(
        "Horizonte de forecast",
        min_value=1,
        max_value=36,
        value=18,
        step=1
    )

    st.markdown("---")

    st.subheader("Modelos incluidos")

    st.markdown(
        """
        - Naive
        - Media movil 3/6
        - Suavizado exponencial
        - Holt
        - ARIMA simple
        - Prophet
        - LightGBM
        - Estacionales si hay 24 meses o mas
        """
    )

    st.markdown("---")

    st.subheader("Estado de librerias")
    st.write(f"statsmodels: {'✅' if STATSMODELS_AVAILABLE else '❌'}")
    st.write(f"Prophet: {'✅' if PROPHET_AVAILABLE else '❌'}")
    st.write(f"LightGBM: {'✅' if LGBM_AVAILABLE else '❌'}")
    st.write(f"Plotly: {'✅' if PLOTLY_AVAILABLE else '❌'}")


# =========================================================
# ENTRADA PRINCIPAL
# =========================================================

st.subheader("1. Pega el historico de ventas desde Excel")

st.info(
    "Formato recomendado: dos columnas `Fecha` y `Ventas`. "
    "Tambien puedes pegar solo una columna de ventas. "
    "Con 16 meses o mas se hara backtesting walk-forward de los ultimos 4 meses."
)

raw_data = st.text_area(
    "Historico de ventas",
    value=st.session_state["raw_data"],
    height=280,
    key=f"input_area_{st.session_state['clear_counter']}"
)

col1, col2 = st.columns([1, 1])

with col1:
    calculate = st.button(
        "🚀 Calcular mejor forecast",
        type="primary",
        use_container_width=True
    )

with col2:
    st.button(
        "🧹 Limpiar historico",
        on_click=clear_history,
        use_container_width=True
    )


# =========================================================
# EJECUCION DEL CALCULO
# =========================================================

if calculate:
    try:
        st.session_state["raw_data"] = raw_data

        df = parse_pasted_data(raw_data, start_month)
        df = complete_monthly_series(df)

        n_months = len(df)

        if n_months < 12:
            st.error("Se necesitan al menos 12 meses de historico.")
            st.stop()

        st.session_state["df"] = df

        if n_months >= 16:
            best_model, best_metric, best_metric_name, results_df, train_df, test_df = run_backtesting(df)

            st.session_state["best_model"] = best_model
            st.session_state["best_metric"] = best_metric
            st.session_state["best_metric_name"] = best_metric_name
            st.session_state["results_df"] = results_df
            st.session_state["train_df"] = train_df
            st.session_state["test_df"] = test_df
            st.session_state["mode"] = "backtesting"
            st.session_state["result_ready"] = True

        else:
            st.session_state["best_model"] = None
            st.session_state["best_metric"] = None
            st.session_state["best_metric_name"] = None
            st.session_state["results_df"] = None
            st.session_state["train_df"] = None
            st.session_state["test_df"] = None
            st.session_state["mode"] = "no_backtesting"
            st.session_state["result_ready"] = True

    except Exception as e:
        st.session_state["result_ready"] = False
        st.error("No se ha podido calcular el forecast.")
        st.exception(e)


# =========================================================
# RENDERIZADO DE RESULTADOS
# =========================================================

if st.session_state["result_ready"] and st.session_state["df"] is not None:

    df = st.session_state["df"]
    n_months = len(df)

    st.subheader("2. Datos interpretados")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Meses historicos", n_months)

    with c2:
        st.metric("Desde", df["Fecha"].min().strftime("%Y-%m"))

    with c3:
        st.metric("Hasta", df["Fecha"].max().strftime("%Y-%m"))

    st.dataframe(df, use_container_width=True)

    model_list = available_models(n_months)

    if st.session_state["mode"] == "no_backtesting":

        st.warning(
            "Con menos de 16 meses no se puede hacer el backtesting 12+4. "
            "La app puede calcular forecast, pero sin seleccion automatica validada."
        )

        st.subheader("3. Seleccion de modelo manual")

        default_model = "Holt amortiguado"

        manual_model = st.selectbox(
            "Selecciona un modelo especifico",
            model_list,
            index=model_list.index(default_model) if default_model in model_list else 0
        )

        try:
            manual_forecast = forecast_future(df, manual_model, int(horizon))

            horizontal_df = pd.DataFrame([
                make_horizontal_forecast(
                    manual_forecast,
                    "Manual sin backtesting completo",
                    manual_model
                )
            ])

            st.subheader("4. Forecast horizontal")

            st.dataframe(horizontal_df, use_container_width=True)

            fig = build_chart(df, auto_forecast=None, manual_forecast=manual_forecast)

            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                chart_df = pd.concat([
                    df.set_index("Fecha")["Ventas"].rename("Historico"),
                    manual_forecast.set_index("Fecha")["Forecast"].rename("Forecast manual")
                ], axis=1)

                st.line_chart(chart_df)

            st.subheader("5. Descarga")

            csv_horizontal = horizontal_df.to_csv(index=False, sep=";").encode("utf-8-sig")

            st.download_button(
                "⬇️ Descargar forecast horizontal CSV",
                data=csv_horizontal,
                file_name="forecast_horizontal.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.error("No se ha podido calcular el modelo manual seleccionado.")
            st.exception(e)

    if st.session_state["mode"] == "backtesting":

        best_model = st.session_state["best_model"]
        best_metric = st.session_state["best_metric"]
        best_metric_name = st.session_state["best_metric_name"]
        results_df = st.session_state["results_df"]
        test_df = st.session_state["test_df"]

        st.subheader("3. Modelo automatico seleccionado")

        cc1, cc2, cc3 = st.columns(3)

        with cc1:
            st.metric("Modelo ganador", best_model)

        with cc2:
            st.metric(best_metric_name, f"{best_metric:.2f}%")

        with cc3:
            st.metric("Fiabilidad", reliability_label(best_metric))

        st.caption(
            "La seleccion se basa en backtesting walk-forward. Cada uno de los ultimos 4 meses "
            "se predice a 1 mes vista, reentrenando con los datos reales disponibles hasta el mes anterior."
        )

        st.subheader("4. Seleccion de modelo alternativo")

        manual_model = st.selectbox(
            "Selecciona un modelo especifico para comparar",
            model_list,
            index=model_list.index(best_model) if best_model in model_list else 0
        )

        try:
            auto_forecast = forecast_future(df, best_model, int(horizon))
            manual_forecast = forecast_future(df, manual_model, int(horizon))

            st.subheader("5. Forecast horizontal automatico y manual")

            horizontal_rows = [
                make_horizontal_forecast(
                    auto_forecast,
                    "Automatico",
                    best_model
                ),
                make_horizontal_forecast(
                    manual_forecast,
                    "Manual seleccionado",
                    manual_model
                )
            ]

            horizontal_df = pd.DataFrame(horizontal_rows)

            st.dataframe(horizontal_df, use_container_width=True)

            st.subheader("6. Comparativa de modelos en backtesting")

            display_results = results_df.copy()

            for col in ["MAPE_%", "sMAPE_%", "WMAPE_%", "Valor metrica seleccion"]:
                if col in display_results.columns:
                    display_results[col] = display_results[col].round(2)

            display_results = display_results.sort_values(
                by=["Estado", "Valor metrica seleccion"],
                ascending=[False, True]
            )

            st.dataframe(display_results, use_container_width=True)

            st.subheader("7. Backtesting walk-forward de los ultimos 4 meses")

            st.caption(
                "Ejemplo para media movil 3 meses: el primer mes de test usa los 3 meses reales anteriores; "
                "el segundo mes de test ya incorpora el primer mes real de test; y asi sucesivamente."
            )

            best_row = results_df[results_df["Modelo"] == best_model].iloc[0]

            backtest_pred = np.array([
                best_row["Backtest M+1"],
                best_row["Backtest M+2"],
                best_row["Backtest M+3"],
                best_row["Backtest M+4"]
            ])

            backtest_view = test_df.copy()
            backtest_view["Forecast modelo ganador"] = backtest_pred
            backtest_view["Error absoluto"] = np.abs(
                backtest_view["Ventas"] - backtest_view["Forecast modelo ganador"]
            )
            backtest_view["Error %"] = np.where(
                backtest_view["Ventas"] != 0,
                backtest_view["Error absoluto"] / backtest_view["Ventas"] * 100,
                np.nan
            )

            st.dataframe(backtest_view, use_container_width=True)

            st.subheader("8. Grafico comparativo")

            fig = build_chart(df, auto_forecast, manual_forecast)

            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                chart_df = pd.concat([
                    df.set_index("Fecha")["Ventas"].rename("Historico"),
                    auto_forecast.set_index("Fecha")["Forecast"].rename("Forecast automatico"),
                    manual_forecast.set_index("Fecha")["Forecast"].rename("Forecast manual")
                ], axis=1)

                st.line_chart(chart_df)

            st.subheader("9. Descargas")

            d1, d2, d3 = st.columns(3)

            with d1:
                csv_horizontal = horizontal_df.to_csv(index=False, sep=";").encode("utf-8-sig")

                st.download_button(
                    "⬇️ Forecast horizontal CSV",
                    data=csv_horizontal,
                    file_name="forecast_horizontal.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with d2:
                csv_models = results_df.to_csv(index=False, sep=";").encode("utf-8-sig")

                st.download_button(
                    "⬇️ Comparativa modelos CSV",
                    data=csv_models,
                    file_name="comparativa_modelos.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with d3:
                csv_backtest = backtest_view.to_csv(index=False, sep=";").encode("utf-8-sig")

                st.download_button(
                    "⬇️ Backtesting ultimos 4 meses CSV",
                    data=csv_backtest,
                    file_name="backtesting_walk_forward_ultimos_4_meses.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            st.markdown("---")

            st.warning(
                "Revision recomendada de Demand Planning: valida manualmente el forecast si existen "
                "lanzamientos, promociones, roturas de stock, ventas extraordinarias, phase-out, "
                "cambios de cliente, cambios de distribucion o restricciones de suministro."
            )

        except Exception as e:
            st.error("No se ha podido calcular el forecast con el modelo seleccionado.")
            st.exception(e)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "MVP de forecasting automatico. Compara modelos mediante backtesting walk-forward, "
    "selecciona el mejor y permite calcular una alternativa manual seleccionada por el usuario."
)
