import streamlit as st
import pandas as pd
import numpy as np
import io
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# =========================
# Optional imports
# =========================

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


# =========================
# Streamlit config
# =========================

st.set_page_config(
    page_title="Forecast automático de ventas",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Forecast automático de ventas")
st.caption(
    "App para seleccionar automáticamente el mejor modelo de previsión mediante backtesting "
    "y generar una previsión de 18 meses."
)


# =========================
# Utility functions
# =========================

def clean_number(value):
    """
    Convierte textos numéricos desde Excel en número.
    Soporta formatos:
    - 1.234,56
    - 1234,56
    - 1,234.56
    - 1234.56
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if text == "":
        return np.nan

    text = text.replace(" ", "")

    # Caso europeo: 1.234,56
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


def parse_pasted_data(raw_text, start_month):
    """
    Lee datos pegados desde Excel.
    Permite:
    1) Dos columnas: Fecha + Ventas
    2) Una columna: Ventas, generando fechas desde start_month
    """
    if raw_text is None or raw_text.strip() == "":
        raise ValueError("No se han introducido datos.")

    raw_text = raw_text.strip()

    # Intentar detectar separador
    possible_separators = ["\t", ";", ","]

    df = None

    for sep in possible_separators:
        try:
            temp = pd.read_csv(io.StringIO(raw_text), sep=sep, header=None)
            if temp.shape[1] >= 1:
                df = temp
                break
        except Exception:
            continue

    if df is None:
        # Fallback por líneas
        lines = raw_text.splitlines()
        df = pd.DataFrame(lines)

    # Eliminar filas completamente vacías
    df = df.dropna(how="all")

    # Si hay una cabecera textual, intentar quitarla
    first_row = " ".join([str(x).lower() for x in df.iloc[0].values])
    if any(word in first_row for word in ["fecha", "mes", "venta", "ventas", "cantidad"]):
        df = df.iloc[1:].reset_index(drop=True)

    if df.shape[1] >= 2:
        # Dos columnas: fecha + ventas
        dates = pd.to_datetime(df.iloc[:, 0].astype(str).str.strip(), errors="coerce")
        values = df.iloc[:, 1].apply(clean_number)

        result = pd.DataFrame({
            "Fecha": dates,
            "Ventas": values
        })

        # Si la fecha viene como 202501 o 2025-01 y no se ha reconocido bien
        if result["Fecha"].isna().all():
            date_text = df.iloc[:, 0].astype(str).str.strip()
            parsed_dates = []
            for x in date_text:
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

            result["Fecha"] = parsed_dates

    else:
        # Una sola columna: ventas
        values = df.iloc[:, 0].apply(clean_number)
        start_date = pd.to_datetime(start_month + "-01")
        dates = pd.date_range(start=start_date, periods=len(values), freq="MS")

        result = pd.DataFrame({
            "Fecha": dates,
            "Ventas": values
        })

    result = result.dropna(subset=["Fecha", "Ventas"]).copy()
    result["Fecha"] = pd.to_datetime(result["Fecha"])
    result["Fecha"] = result["Fecha"].dt.to_period("M").dt.to_timestamp()
    result["Ventas"] = result["Ventas"].astype(float)

    result = result.sort_values("Fecha").drop_duplicates(subset=["Fecha"], keep="last")
    result = result.reset_index(drop=True)

    if result.empty:
        raise ValueError("No se han podido interpretar los datos pegados.")

    if (result["Ventas"] < 0).any():
        raise ValueError("Existen ventas negativas. Revisa si son devoluciones o errores de dato.")

    return result


def complete_monthly_series(df):
    """
    Completa meses faltantes.
    Si falta algún mes, se rellena con 0.
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


# =========================
# Forecast model functions
# =========================

def forecast_naive(y, horizon):
    return np.repeat(y.iloc[-1], horizon)


def forecast_moving_average(y, horizon, window):
    if len(y) < window:
        raise ValueError(f"No hay datos suficientes para media móvil {window}.")
    return np.repeat(y.iloc[-window:].mean(), horizon)


def forecast_seasonal_naive_12(y, horizon):
    if len(y) < 12:
        raise ValueError("No hay datos suficientes para seasonal naive 12.")
    last_12 = y.iloc[-12:].values
    values = [last_12[i % 12] for i in range(horizon)]
    return np.array(values)


def forecast_ses(y, horizon):
    if not STATSMODELS_AVAILABLE:
        raise ValueError("statsmodels no está instalado.")
    model = SimpleExpSmoothing(
        y,
        initialization_method="estimated"
    ).fit(optimized=True)
    return model.forecast(horizon).values


def forecast_holt(y, horizon, damped=False):
    if not STATSMODELS_AVAILABLE:
        raise ValueError("statsmodels no está instalado.")
    model = Holt(
        y,
        damped_trend=damped,
        initialization_method="estimated"
    ).fit(optimized=True)
    return model.forecast(horizon).values


def forecast_hw_additive(y, horizon):
    if not STATSMODELS_AVAILABLE:
        raise ValueError("statsmodels no está instalado.")
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
        raise ValueError("statsmodels no está instalado.")

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
        raise ValueError("Prophet no está instalado.")

    df_prophet = pd.DataFrame({
        "ds": dates,
        "y": y.values
    })

    # Configuraciones prudentes para histórico corto
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

    # Para series muy cortas, mejor no forzar estacionalidad anual
    yearly = True if len(y) >= 24 else False

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
            continue

    raise ValueError(f"Prophet no ha podido ajustarse: {last_error}")


def make_lgbm_features(values, dates):
    """
    Convierte una serie en dataset tabular para LightGBM.
    Features:
    - lag_1, lag_2, lag_3
    - rolling_mean_3
    - rolling_mean_6
    - month
    - trend
    """
    df = pd.DataFrame({
        "Fecha": dates,
        "y": values
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


def forecast_lgbm_recursive(y, dates, horizon):
    if not LGBM_AVAILABLE:
        raise ValueError("LightGBM no está instalado.")

    if len(y) < 12:
        raise ValueError("LightGBM necesita al menos 12 meses.")

    hist_dates = pd.Series(dates).reset_index(drop=True)
    hist_values = pd.Series(y.values).reset_index(drop=True)

    train_df, feature_cols = make_lgbm_features(hist_values, hist_dates)

    if len(train_df) < 5:
        raise ValueError("No hay suficientes filas entrenables para LightGBM.")

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

    future_predictions = []

    current_values = list(hist_values.values)
    last_date = hist_dates.iloc[-1]

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

        future_predictions.append(pred)
        current_values.append(pred)

    return np.array(future_predictions)


def run_model(model_name, y, dates, horizon):
    """
    Ejecuta un modelo concreto.
    """
    if model_name == "Naive - último valor":
        return forecast_naive(y, horizon)

    if model_name == "Media móvil 3 meses":
        return forecast_moving_average(y, horizon, 3)

    if model_name == "Media móvil 6 meses":
        return forecast_moving_average(y, horizon, 6)

    if model_name == "Seasonal naive 12 meses":
        return forecast_seasonal_naive_12(y, horizon)

    if model_name == "Suavizado exponencial simple":
        return forecast_ses(y, horizon)

    if model_name == "Holt tendencia":
        return forecast_holt(y, horizon, damped=False)

    if model_name == "Holt tendencia amortiguada":
        return forecast_holt(y, horizon, damped=True)

    if model_name == "Holt-Winters estacional":
        return forecast_hw_additive(y, horizon)

    if model_name == "ARIMA simple":
        return forecast_arima_simple(y, horizon)

    if model_name == "Prophet":
        return forecast_prophet(y, dates, horizon)

    if model_name == "LightGBM":
        return forecast_lgbm_recursive(y, dates, horizon)

    raise ValueError(f"Modelo no reconocido: {model_name}")


def candidate_models(n_months):
    """
    Define modelos candidatos en función del histórico disponible.
    """
    models = [
        "Naive - último valor",
        "Media móvil 3 meses",
        "Media móvil 6 meses",
        "Suavizado exponencial simple",
        "Holt tendencia",
        "Holt tendencia amortiguada",
        "ARIMA simple",
        "Prophet",
        "LightGBM"
    ]

    if n_months >= 24:
        models.append("Seasonal naive 12 meses")
        models.append("Holt-Winters estacional")

    return models


def model_complexity_rank(model_name):
    """
    Para desempates: si dos modelos tienen métrica muy parecida,
    se prefiere el modelo más simple.
    """
    ranking = {
        "Naive - último valor": 1,
        "Media móvil 3 meses": 2,
        "Media móvil 6 meses": 3,
        "Seasonal naive 12 meses": 4,
        "Suavizado exponencial simple": 5,
        "Holt tendencia": 6,
        "Holt tendencia amortiguada": 7,
        "Holt-Winters estacional": 8,
        "ARIMA simple": 9,
        "Prophet": 10,
        "LightGBM": 11
    }
    return ranking.get(model_name, 99)


def run_backtest(df):
    """
    Backtesting:
    - Train: todo menos últimos 4 meses
    - Test: últimos 4 meses reales
    """
    train = df.iloc[:-4].copy()
    test = df.iloc[-4:].copy()

    y_train = train["Ventas"]
    dates_train = train["Fecha"]

    y_test = test["Ventas"].values

    results = []

    models = candidate_models(len(df))

    for model_name in models:
        try:
            pred = run_model(
                model_name=model_name,
                y=y_train,
                dates=dates_train,
                horizon=4
            )

            pred = postprocess_forecast(pred)

            model_mape = safe_mape(y_test, pred)
            model_smape = smape(y_test, pred)
            model_wmape = wmape(y_test, pred)

            # Si hay ceros en el test, el MAPE puede ser engañoso.
            if np.any(y_test == 0):
                selection_metric_name = "sMAPE"
                selection_metric_value = model_smape
            else:
                selection_metric_name = "MAPE"
                selection_metric_value = model_mape

            results.append({
                "Modelo": model_name,
                "MAPE_%": model_mape,
                "sMAPE_%": model_smape,
                "WMAPE_%": model_wmape,
                "Métrica selección": selection_metric_name,
                "Valor métrica selección": selection_metric_value,
                "Backtest M+1": pred[0],
                "Backtest M+2": pred[1],
                "Backtest M+3": pred[2],
                "Backtest M+4": pred[3],
                "Estado": "OK",
                "Error": ""
            })

        except Exception as e:
            results.append({
                "Modelo": model_name,
                "MAPE_%": np.nan,
                "sMAPE_%": np.nan,
                "WMAPE_%": np.nan,
                "Métrica selección": "",
                "Valor métrica selección": np.nan,
                "Backtest M+1": np.nan,
                "Backtest M+2": np.nan,
                "Backtest M+3": np.nan,
                "Backtest M+4": np.nan,
                "Estado": "Error",
                "Error": str(e)
            })

    results_df = pd.DataFrame(results)

    valid = results_df[
        (results_df["Estado"] == "OK") &
        (~results_df["Valor métrica selección"].isna())
    ].copy()

    if valid.empty:
        raise ValueError("Ningún modelo ha podido calcularse correctamente.")

    valid["Complejidad"] = valid["Modelo"].apply(model_complexity_rank)

    # Orden por métrica y luego por simplicidad
    valid = valid.sort_values(
        by=["Valor métrica selección", "Complejidad"],
        ascending=[True, True]
    )

    best_model = valid.iloc[0]["Modelo"]
    best_metric = valid.iloc[0]["Valor métrica selección"]
    metric_name = valid.iloc[0]["Métrica selección"]

    return best_model, best_metric, metric_name, results_df, train, test


def forecast_final(df, best_model, horizon=18):
    y = df["Ventas"]
    dates = df["Fecha"]

    pred = run_model(
        model_name=best_model,
        y=y,
        dates=dates,
        horizon=horizon
    )

    pred = postprocess_forecast(pred)

    last_date = df["Fecha"].max()
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=horizon,
        freq="MS"
    )

    forecast_df = pd.DataFrame({
        "Fecha": future_dates,
        "Mes": [f"M+{i}" for i in range(1, horizon + 1)],
        "Forecast": pred
    })

    return forecast_df


def fallback_forecast_without_backtest(df, horizon=18):
    """
    Para histórico entre 12 y 15 meses:
    no hay backtesting 12+4 suficiente.
    Se usa una regla conservadora.
    """
    y = df["Ventas"]
    dates = df["Fecha"]

    fallback_models = [
        "Holt tendencia amortiguada",
        "Holt tendencia",
        "Suavizado exponencial simple",
        "Media móvil 3 meses",
        "Naive - último valor"
    ]

    last_error = None

    for model_name in fallback_models:
        try:
            pred = run_model(model_name, y, dates, horizon)
            pred = postprocess_forecast(pred)

            last_date = df["Fecha"].max()
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=horizon,
                freq="MS"
            )

            forecast_df = pd.DataFrame({
                "Fecha": future_dates,
                "Mes": [f"M+{i}" for i in range(1, horizon + 1)],
                "Forecast": pred
            })

            return model_name, forecast_df

        except Exception as e:
            last_error = e
            continue

    raise ValueError(f"No se ha podido generar forecast fallback: {last_error}")


def build_forecast_chart(df, forecast_df, test_df=None, backtest_pred=None):
    if not PLOTLY_AVAILABLE:
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Fecha"],
        y=df["Ventas"],
        mode="lines+markers",
        name="Histórico real",
        line=dict(color="#1f77b4", width=3)
    ))

    if test_df is not None and backtest_pred is not None:
        fig.add_trace(go.Scatter(
            x=test_df["Fecha"],
            y=backtest_pred,
            mode="lines+markers",
            name="Backtest modelo ganador",
            line=dict(color="#ff7f0e", width=3, dash="dash")
        ))

    fig.add_trace(go.Scatter(
        x=forecast_df["Fecha"],
        y=forecast_df["Forecast"],
        mode="lines+markers",
        name="Forecast 18 meses",
        line=dict(color="#2ca02c", width=3)
    ))

    fig.update_layout(
        title="Histórico, backtesting y forecast",
        xaxis_title="Fecha",
        yaxis_title="Ventas",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=520
    )

    return fig


# =========================
# Sidebar
# =========================

with st.sidebar:
    st.header("⚙️ Configuración")

    start_month = st.text_input(
        "Mes inicial si pegas solo una columna",
        value="2025-01",
        help="Formato recomendado: YYYY-MM. Solo se usa si pegas una única columna de ventas."
    )

    horizon = st.number_input(
        "Horizonte forecast",
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
        - Media móvil 3/6
        - Suavizado exponencial
        - Holt
        - ARIMA simple
        - Prophet
        - LightGBM
        - Estacionales si hay ≥24 meses
        """
    )

    st.markdown("---")

    st.subheader("Librerías detectadas")

    st.write(f"statsmodels: {'✅' if STATSMODELS_AVAILABLE else '❌'}")
    st.write(f"Prophet: {'✅' if PROPHET_AVAILABLE else '❌'}")
    st.write(f"LightGBM: {'✅' if LGBM_AVAILABLE else '❌'}")
    st.write(f"Plotly: {'✅' if PLOTLY_AVAILABLE else '❌'}")


# =========================
# Input area
# =========================

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

st.subheader("1. Pega el histórico de ventas desde Excel")

st.info(
    "Formato recomendado: dos columnas `Fecha` y `Ventas`. "
    "También puedes pegar solo una columna de ventas; en ese caso se generarán fechas desde el mes inicial indicado."
)

raw_data = st.text_area(
    "Datos históricos",
    value=st.session_state["raw_data"],
    height=280,
    key="input_area"
)

col_btn_1, col_btn_2, col_btn_3 = st.columns([1, 1, 4])

with col_btn_1:
    calculate = st.button("🚀 Calcular mejor forecast", type="primary")

with col_btn_2:
    clear = st.button("🧹 Limpiar")

if clear:
    st.session_state["raw_data"] = ""
    st.rerun()


# =========================
# Main calculation
# =========================

if calculate:
    try:
        st.session_state["raw_data"] = raw_data

        df = parse_pasted_data(raw_data, start_month)
        df = complete_monthly_series(df)

        n_months = len(df)

        st.subheader("2. Datos interpretados")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric("Meses históricos", n_months)

        with col_b:
            st.metric("Desde", df["Fecha"].min().strftime("%Y-%m"))

        with col_c:
            st.metric("Hasta", df["Fecha"].max().strftime("%Y-%m"))

        st.dataframe(df, use_container_width=True)

        if n_months < 12:
            st.error(
                "Histórico insuficiente. Se necesitan al menos 12 meses para generar una previsión mínima."
            )
            st.stop()

        if n_months < 16:
            st.warning(
                "Hay entre 12 y 15 meses de histórico. "
                "La app generará forecast, pero no puede hacer el backtesting completo 12+4."
            )

            selected_model, forecast_df = fallback_forecast_without_backtest(
                df=df,
                horizon=int(horizon)
            )

            st.subheader("3. Resultado forecast sin backtesting completo")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Modelo usado", selected_model)

            with c2:
                st.metric("Backtesting", "No disponible")

            with c3:
                st.metric("Fiabilidad", "No evaluada")

            fig = build_forecast_chart(df, forecast_df)

            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("4. Forecast generado")
            st.dataframe(forecast_df, use_container_width=True)

            csv = forecast_df.to_csv(index=False, sep=";").encode("utf-8-sig")

            st.download_button(
                label="⬇️ Descargar forecast CSV",
                data=csv,
                file_name="forecast_18_meses.csv",
                mime="text/csv"
            )

            st.stop()

        # Backtesting completo
        best_model, best_metric, metric_name, results_df, train_df, test_df = run_backtest(df)

        forecast_df = forecast_final(
            df=df,
            best_model=best_model,
            horizon=int(horizon)
        )

        best_row = results_df[results_df["Modelo"] == best_model].iloc[0]

        backtest_pred = np.array([
            best_row["Backtest M+1"],
            best_row["Backtest M+2"],
            best_row["Backtest M+3"],
            best_row["Backtest M+4"]
        ])

        reliability = reliability_label(best_metric)

        st.subheader("3. Mejor modelo seleccionado")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Modelo ganador", best_model)

        with col2:
            st.metric(metric_name, f"{best_metric:.2f}%")

        with col3:
            st.metric("Fiabilidad", reliability)

        with col4:
            st.metric("Horizonte forecast", f"{int(horizon)} meses")

        st.caption(
            "La fiabilidad se calcula según la métrica de selección del backtesting. "
            "Si existen meses con venta real igual a cero, se usa sMAPE en lugar de MAPE."
        )

        st.subheader("4. Comparativa de modelos")

        display_results = results_df.copy()

        numeric_cols = [
            "MAPE_%",
            "sMAPE_%",
            "WMAPE_%",
            "Valor métrica selección",
            "Backtest M+1",
            "Backtest M+2",
            "Backtest M+3",
            "Backtest M+4"
        ]

        for col in numeric_cols:
            if col in display_results.columns:
                display_results[col] = display_results[col].round(2)

        display_results = display_results.sort_values(
            by=["Estado", "Valor métrica selección"],
            ascending=[False, True]
        )

        st.dataframe(display_results, use_container_width=True)

        st.subheader("5. Backtesting últimos 4 meses")

        backtest_view = test_df.copy()
        backtest_view["Forecast modelo ganador"] = backtest_pred
        backtest_view["Error abs."] = np.abs(
            backtest_view["Ventas"] - backtest_view["Forecast modelo ganador"]
        )
        backtest_view["Error %"] = np.where(
            backtest_view["Ventas"] != 0,
            backtest_view["Error abs."] / backtest_view["Ventas"] * 100,
            np.nan
        )

        st.dataframe(backtest_view, use_container_width=True)

        st.subheader("6. Gráfico histórico + backtesting + forecast")

        fig = build_forecast_chart(
            df=df,
            forecast_df=forecast_df,
            test_df=test_df,
            backtest_pred=backtest_pred
        )

        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(
                pd.concat([
                    df.set_index("Fecha")["Ventas"].rename("Histórico"),
                    forecast_df.set_index("Fecha")["Forecast"].rename("Forecast")
                ], axis=1)
            )

        st.subheader("7. Forecast final")

        st.dataframe(forecast_df, use_container_width=True)

        # =========================
        # Downloads
        # =========================

        st.subheader("8. Descargas")

        col_down_1, col_down_2, col_down_3 = st.columns(3)

        with col_down_1:
            forecast_csv = forecast_df.to_csv(index=False, sep=";").encode("utf-8-sig")

            st.download_button(
                label="⬇️ Descargar forecast CSV",
                data=forecast_csv,
                file_name="forecast_18_meses.csv",
                mime="text/csv"
            )

        with col_down_2:
            models_csv = results_df.to_csv(index=False, sep=";").encode("utf-8-sig")

            st.download_button(
                label="⬇️ Descargar comparativa modelos CSV",
                data=models_csv,
                file_name="comparativa_modelos.csv",
                mime="text/csv"
            )

        with col_down_3:
            backtest_csv = backtest_view.to_csv(index=False, sep=";").encode("utf-8-sig")

            st.download_button(
                label="⬇️ Descargar backtesting CSV",
                data=backtest_csv,
                file_name="backtesting_4_meses.csv",
                mime="text/csv"
            )

        # =========================
        # Business warning
        # =========================

        st.markdown("---")
        st.warning(
            "Recomendación de negocio: revisar manualmente el forecast si existen lanzamientos, "
            "promociones, roturas de stock, cambios de cliente, phase-out, cambios de fórmula, "
            "ventas extraordinarias o restricciones de suministro."
        )

    except Exception as e:
        st.error("No se ha podido calcular el forecast.")
        st.exception(e)


# =========================
# Footer
# =========================

st.markdown("---")

st.caption(
    "MVP de forecasting automático. Selecciona el modelo con mejor comportamiento en backtesting "
    "y genera una previsión futura usando todo el histórico disponible."
)
