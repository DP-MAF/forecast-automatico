import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
warnings.filterwarnings("ignore")

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False
try:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_OK = True
except Exception:
    STATSMODELS_OK = False
try:
    from prophet import Prophet
    PROPHET_OK = True
except Exception:
    PROPHET_OK = False
try:
    from lightgbm import LGBMRegressor
    LGBM_OK = True
except Exception:
    LGBM_OK = False
try:
    from xgboost import XGBRegressor
    XGBOOST_OK = True
except Exception:
    XGBOOST_OK = False
try:
    from sklearn.ensemble import RandomForestRegressor
    RFOREST_OK = True
except Exception:
    RFOREST_OK = False

st.set_page_config(page_title="Forecast automático V2", page_icon="📈", layout="wide")
st.title("📈 Forecast automático de ventas V2")
st.caption("Backtesting walk-forward, demanda intermitente, confiabilidad, selección automática y ensembles Top 4.")

# ============================== DATOS ==============================
def clean_number(value):
    if pd.isna(value): return np.nan
    text = str(value).strip().replace(" ", "")
    if not text: return np.nan
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".") if text.rfind(",") > text.rfind(".") else text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    try: return float(text)
    except Exception: return np.nan

def parse_pasted_data(raw_text, start_month):
    if not raw_text or not raw_text.strip(): raise ValueError("Pega al menos 16 meses de datos.")
    df = None
    for sep in ["\t", ";"]:
        try:
            temp = pd.read_csv(io.StringIO(raw_text.strip()), sep=sep, header=None, engine="python")
            if temp.shape[1] >= 2: df = temp; break
        except Exception: pass
    if df is None: df = pd.DataFrame(raw_text.strip().splitlines())
    df = df.dropna(how="all").reset_index(drop=True)
    if df.empty: raise ValueError("No se han podido interpretar los datos.")
    first = " ".join(str(x).lower() for x in df.iloc[0].values)
    if any(word in first for word in ["fecha", "mes", "venta", "cantidad", "sales"]):
        df = df.iloc[1:].reset_index(drop=True)
    if df.shape[1] >= 2:
        raw_dates = df.iloc[:, 0].astype(str).str.strip()
        dates = pd.to_datetime(raw_dates, errors="coerce")
        if dates.isna().all(): dates = pd.to_datetime(raw_dates + "-01", errors="coerce")
        result = pd.DataFrame({"Fecha": dates, "Ventas": df.iloc[:, 1].map(clean_number)})
    else:
        values = df.iloc[:, 0].map(clean_number)
        result = pd.DataFrame({"Fecha": pd.date_range(pd.to_datetime(start_month + "-01"), periods=len(values), freq="MS"), "Ventas": values})
    result = result.dropna(subset=["Fecha", "Ventas"]).copy()
    result["Fecha"] = pd.to_datetime(result["Fecha"]).dt.to_period("M").dt.to_timestamp()
    result = result.sort_values("Fecha").drop_duplicates("Fecha", keep="last").reset_index(drop=True)
    if result.empty: raise ValueError("No hay filas válidas.")
    if (result["Ventas"] < 0).any(): raise ValueError("Hay ventas negativas.")
    all_dates = pd.date_range(result["Fecha"].min(), result["Fecha"].max(), freq="MS")
    return pd.DataFrame({"Fecha": all_dates}).merge(result, on="Fecha", how="left").fillna({"Ventas": 0})

# ============================== MÉTRICAS ==============================
def mape(actual, pred):
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100 if mask.any() else np.nan

def smape(actual, pred):
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    den = (np.abs(actual) + np.abs(pred)) / 2
    mask = den != 0
    return np.mean(np.abs(actual[mask] - pred[mask]) / den[mask]) * 100 if mask.any() else np.nan

def wmape(actual, pred):
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    den = np.abs(actual).sum()
    return np.abs(actual - pred).sum() / den * 100 if den else np.nan

def round_zero(values):
    return np.rint(np.maximum(np.nan_to_num(np.asarray(values, float)), 0)).astype(int)

# ============================== DIAGNÓSTICO Y CONFIABILIDAD ==============================
def demand_diagnosis(df):
    y = df["Ventas"].astype(float)
    mean = y.mean()
    cv = y.std(ddof=0) / mean if mean else np.nan
    zero_ratio = (y == 0).mean()
    slope = np.polyfit(np.arange(len(y)), y, 1)[0] / mean if len(y) > 1 and mean else 0
    seasonal, strength = False, np.nan
    if len(y) >= 24 and mean:
        strength = df.assign(Mes=df["Fecha"].dt.month).groupby("Mes")["Ventas"].mean().std(ddof=0) / mean
        seasonal = strength >= 0.20
    first4, last4 = y.head(4).mean(), y.tail(4).mean()
    last3 = y.tail(3).mean()
    previous6 = y.iloc[-9:-3].mean() if len(y) >= 9 else np.nan
    if mean == 0:
        kind, explanation, recommendation = "Sin demanda histórica", "La media histórica es cero.", "Utilizar información comercial o de lanzamiento."
    elif zero_ratio >= 0.40:
        kind, explanation, recommendation = "Intermitente", "Existe una proporción elevada de meses con venta cero.", "Revisar especialmente Croston, SBA y TSB."
    elif len(y) >= 8 and first4 > 0 and last4 >= 1.8 * first4 and slope > 0.03:
        kind, explanation, recommendation = "Posible lanzamiento", "La demanda reciente crece claramente frente al inicio.", "Validar pipeline, distribución y ramp-up."
    elif len(y) >= 9 and previous6 > 0 and last3 <= 0.5 * previous6 and slope < -0.03:
        kind, explanation, recommendation = "Posible phase-out", "La demanda reciente cae claramente frente al periodo anterior.", "Validar descatalogación o sustitución."
    elif seasonal:
        kind, explanation, recommendation = "Estacional", "Se observan diferencias relevantes entre meses calendario.", "Revisar modelos estacionales."
    elif not pd.isna(cv) and cv <= 0.20:
        kind, explanation, recommendation = "Estable", "La variabilidad relativa es baja.", "El forecast estadístico suele ser defendible."
    elif not pd.isna(cv) and cv >= 0.70:
        kind, explanation, recommendation = "Volátil", "La variabilidad relativa es elevada.", "Revisar promociones, roturas y pedidos extraordinarios."
    elif slope >= 0.03:
        kind, explanation, recommendation = "Tendencia creciente", "La pendiente mensual es positiva.", "Validar la continuidad del crecimiento."
    elif slope <= -0.03:
        kind, explanation, recommendation = "Tendencia decreciente", "La pendiente mensual es negativa.", "Validar pérdida de distribución o declive."
    else:
        kind, explanation, recommendation = "Sin patrón dominante", "No se detecta un patrón claramente dominante.", "Usar el resultado automático con revisión de negocio."
    return {"Tipo de demanda": kind, "Explicación": explanation, "Recomendación": recommendation, "CV": cv, "Meses con venta cero %": zero_ratio * 100, "Tendencia mensual %": slope * 100, "Estacionalidad": "Sí" if seasonal else "No", "Fuerza estacional": strength}

def confidence_level(score):
    if score >= 90: return "Muy Alta", "🟢", "Forecast muy robusto. Revisión mínima salvo eventos relevantes."
    if score >= 75: return "Alta", "🟢", "Forecast fiable. Revisar promociones, lanzamientos y cambios estructurales."
    if score >= 60: return "Media", "🟡", "Forecast aceptable. Validar con información de negocio."
    if score >= 40: return "Baja", "🟠", "Incertidumbre significativa. Revisión de Demand Planning necesaria."
    return "Muy Baja", "🔴", "Forecast poco fiable. No utilizar sin validación manual y fuentes adicionales."

def forecast_confidence(df, error_value):
    y, n = df["Ventas"].astype(float), len(df)
    mean = y.mean()
    cv = y.std(ddof=0) / mean if mean else np.nan
    error_score = max(0, 100 - 2 * error_value) if not pd.isna(error_value) else 0
    stability_score = 0 if pd.isna(cv) else 100 if cv <= 0.20 else 0 if cv >= 1 else 100 * (1 - (cv - 0.20) / 0.80)
    history_score = 100 if n >= 48 else 85 + (n - 36) * 1.25 if n >= 36 else 70 + (n - 24) * 1.25 if n >= 24 else 50 + (n - 16) * 2.5
    score = round(0.50 * error_score + 0.30 * stability_score + 0.20 * history_score, 1)
    level, icon, use = confidence_level(score)
    return {"Confiabilidad": score, "Nivel": level, "Indicador": icon, "Interpretación": use, "Score precisión": round(error_score, 1), "Score estabilidad": round(stability_score, 1), "Score histórico": round(history_score, 1)}

# ============================== CROSTON, SBA Y TSB ==============================
def croston_rate(y, alpha=0.1, variant="Croston", beta=0.05):
    y = np.asarray(y, float)
    nz = np.flatnonzero(y > 0)
    if len(nz) == 0: return 0.0
    if variant == "TSB":
        size, probability = y[nz[0]], 1.0
        for value in y[nz[0] + 1:]:
            occurrence = 1.0 if value > 0 else 0.0
            probability = beta * occurrence + (1 - beta) * probability
            if value > 0: size = alpha * value + (1 - alpha) * size
        return probability * size
    size, interval, gap = y[nz[0]], 1.0, 0
    for value in y[nz[0] + 1:]:
        gap += 1
        if value > 0:
            size = alpha * value + (1 - alpha) * size
            interval = alpha * gap + (1 - alpha) * interval
            gap = 0
    rate = size / max(interval, 1e-9)
    return rate * (1 - alpha / 2) if variant == "SBA" else rate

def intermittent_forecast(y, horizon, variant):
    best = (float("inf"), 0.1, 0.05)
    y = np.asarray(y, float)
    for alpha in [0.05, 0.10, 0.15, 0.20, 0.30]:
        for beta in ([0.02, 0.05, 0.10, 0.20] if variant == "TSB" else [0.05]):
            errors = [abs(y[i] - croston_rate(y[:i], alpha, variant, beta)) for i in range(max(6, len(y) - 4), len(y))]
            score = np.mean(errors) if errors else float("inf")
            if score < best[0]: best = (score, alpha, beta)
    return np.repeat(croston_rate(y, best[1], variant, best[2]), horizon)

# ============================== MACHINE LEARNING ==============================
BASE_FEATURES = ["lag1", "lag2", "lag3", "lag6", "mean3", "mean6", "std3", "month_sin", "month_cos", "trend"]
def ml_training_table(y, dates):
    d = pd.DataFrame({"Fecha": pd.to_datetime(dates).reset_index(drop=True), "y": pd.Series(np.asarray(y, float))})
    for lag in [1, 2, 3, 6]: d[f"lag{lag}"] = d["y"].shift(lag)
    features = BASE_FEATURES.copy()
    if len(d) >= 24: d["lag12"] = d["y"].shift(12); features.insert(4, "lag12")
    d["mean3"] = d["y"].shift(1).rolling(3).mean(); d["mean6"] = d["y"].shift(1).rolling(6).mean(); d["std3"] = d["y"].shift(1).rolling(3).std(ddof=0)
    d["month_sin"] = np.sin(2 * np.pi * d["Fecha"].dt.month / 12); d["month_cos"] = np.cos(2 * np.pi * d["Fecha"].dt.month / 12); d["trend"] = np.arange(len(d))
    return d.dropna().reset_index(drop=True), features

def ml_forecast(y, dates, horizon, model_name):
    history, date_history = list(np.asarray(y, float)), list(pd.to_datetime(dates))
    train, features = ml_training_table(pd.Series(history), pd.Series(date_history))
    if len(train) < 6: raise ValueError(f"{model_name}: pocas filas entrenables.")
    if model_name == "LightGBM":
        if not LGBM_OK: raise ValueError("LightGBM no está instalado.")
        model = LGBMRegressor(n_estimators=200, learning_rate=0.04, num_leaves=7, max_depth=3, min_child_samples=1, verbosity=-1, random_state=42)
    elif model_name == "XGBoost":
        if not XGBOOST_OK: raise ValueError("XGBoost no está instalado.")
        model = XGBRegressor(n_estimators=200, learning_rate=0.04, max_depth=3, objective="reg:squarederror", n_jobs=1, verbosity=0, random_state=42)
    else:
        if not RFOREST_OK: raise ValueError("Random Forest no está instalado.")
        model = RandomForestRegressor(n_estimators=250, max_depth=5, min_samples_leaf=1, n_jobs=-1, random_state=42)
    model.fit(train[features], train["y"])
    output, last_date = [], date_history[-1]
    for step in range(1, horizon + 1):
        next_date = last_date + pd.DateOffset(months=step)
        lag = lambda k: history[-k] if len(history) >= k else history[0]
        row = {"lag1": lag(1), "lag2": lag(2), "lag3": lag(3), "lag6": lag(6), "mean3": np.mean(history[-3:]), "mean6": np.mean(history[-6:]), "std3": np.std(history[-3:]), "month_sin": np.sin(2*np.pi*next_date.month/12), "month_cos": np.cos(2*np.pi*next_date.month/12), "trend": len(history)}
        if "lag12" in features: row["lag12"] = lag(12)
        pred = max(0, float(model.predict(pd.DataFrame([row])[features])[0]))
        output.append(pred); history.append(pred)
    return np.asarray(output)

# ============================== MODELOS ==============================
def available_models(n_months):
    names = ["Naive", "Media móvil 3", "Media móvil 6", "Suavizado exponencial", "Holt", "Holt amortiguado", "ARIMA simple", "Prophet", "Croston", "SBA", "TSB"]
    if n_months >= 18: names += ["LightGBM", "XGBoost", "Random Forest"]
    if n_months >= 24: names += ["Naive estacional 12", "Holt-Winters estacional"]
    return names

def run_model(name, y, dates, horizon):
    y = pd.Series(np.asarray(y, float)).reset_index(drop=True)
    if name == "Naive": return np.repeat(y.iloc[-1], horizon)
    if name.startswith("Media móvil"):
        window = 3 if name.endswith("3") else 6
        return np.repeat(y.tail(window).mean(), horizon)
    if name == "Naive estacional 12":
        base = y.tail(12).values
        return np.array([base[i % 12] for i in range(horizon)])
    if name in ["Croston", "SBA", "TSB"]: return intermittent_forecast(y, horizon, name)
    if name == "Suavizado exponencial":
        if not STATSMODELS_OK: raise ValueError("statsmodels no está instalado.")
        return SimpleExpSmoothing(y, initialization_method="estimated").fit().forecast(horizon).values
    if name in ["Holt", "Holt amortiguado"]:
        if not STATSMODELS_OK: raise ValueError("statsmodels no está instalado.")
        return Holt(y, damped_trend=name.endswith("amortiguado"), initialization_method="estimated").fit().forecast(horizon).values
    if name == "Holt-Winters estacional":
        if not STATSMODELS_OK: raise ValueError("statsmodels no está instalado.")
        return ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=12, initialization_method="estimated").fit().forecast(horizon).values
    if name == "ARIMA simple":
        if not STATSMODELS_OK: raise ValueError("statsmodels no está instalado.")
        best, best_aic = None, float("inf")
        for order in [(0,1,0), (1,1,0), (0,1,1), (1,1,1), (2,1,1)]:
            try:
                fitted = SARIMAX(y, order=order, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                if fitted.aic < best_aic: best, best_aic = fitted, fitted.aic
            except Exception: pass
        if best is None: raise ValueError("ARIMA no ha podido ajustarse.")
        return best.forecast(horizon).values
    if name == "Prophet":
        if not PROPHET_OK: raise ValueError("Prophet no está instalado.")
        model = Prophet(yearly_seasonality=len(y) >= 24, weekly_seasonality=False, daily_seasonality=False)
        model.fit(pd.DataFrame({"ds": pd.to_datetime(dates), "y": y.values}))
        return model.predict(model.make_future_dataframe(periods=horizon, freq="MS")).tail(horizon)["yhat"].values
    if name in ["LightGBM", "XGBoost", "Random Forest"]: return ml_forecast(y, dates, horizon, name)
    raise ValueError("Modelo no reconocido.")

# ============================== BACKTESTING Y BLENDS ==============================
def run_backtesting(df):
    cut = len(df) - 4
    actual = df.iloc[cut:]["Ventas"].values
    rows = []
    for name in available_models(len(df)):
        try:
            pred = np.array([round_zero(run_model(name, df.iloc[:i]["Ventas"], df.iloc[:i]["Fecha"], 1))[0] for i in range(cut, len(df))], float)
            ma, sm, wm = mape(actual, pred), smape(actual, pred), wmape(actual, pred)
            metric = "sMAPE" if np.any(actual == 0) else "MAPE"
            selected_error = sm if metric == "sMAPE" else ma
            rows.append({"Modelo": name, "MAPE_%": ma, "sMAPE_%": sm, "WMAPE_%": wm, "Métrica": metric, "Error selección_%": selected_error, **{f"BT{i+1}": pred[i] for i in range(4)}, "Estado": "OK", "Detalle": ""})
        except Exception as exc:
            rows.append({"Modelo": name, "MAPE_%": np.nan, "sMAPE_%": np.nan, "WMAPE_%": np.nan, "Métrica": "", "Error selección_%": np.nan, **{f"BT{i+1}": np.nan for i in range(4)}, "Estado": "Error", "Detalle": str(exc)})
    results = pd.DataFrame(rows)
    valid = results[(results["Estado"] == "OK") & results["Error selección_%"].notna()].sort_values("Error selección_%")
    if len(valid) < 4: raise ValueError("Se necesitan al menos cuatro modelos válidos.")
    return results, valid, df.iloc[cut:].copy()

def future_forecast(df, name):
    values = round_zero(run_model(name, df["Ventas"], df["Fecha"], 18))
    return pd.DataFrame({"Fecha": pd.date_range(df["Fecha"].max() + pd.DateOffset(months=1), periods=18, freq="MS"), "Forecast": values})

def ensemble_forecasts(df, valid):
    items = []
    for name in valid["Modelo"]:
        try: items.append((name, future_forecast(df, name), float(valid.loc[valid["Modelo"] == name, "Error selección_%"].iloc[0])))
        except Exception: pass
        if len(items) == 4: break
    if len(items) < 4: raise ValueError("No hay cuatro forecasts finales válidos.")
    fixed_weights = np.array([0.4, 0.3, 0.2, 0.1])
    errors = np.array([max(item[2], 0.01) for item in items])
    dynamic_weights = (1 / errors) / (1 / errors).sum()
    matrix = np.vstack([item[1]["Forecast"].values for item in items])
    base = items[0][1][["Fecha"]].copy()
    fixed, dynamic = base.copy(), base.copy()
    fixed["Forecast"] = round_zero(fixed_weights @ matrix)
    dynamic["Forecast"] = round_zero(dynamic_weights @ matrix)
    detail = pd.DataFrame({"Posición": range(1,5), "Modelo": [x[0] for x in items], "Error_%": errors, "Peso fijo": fixed_weights, "Peso dinámico": dynamic_weights})
    return fixed, dynamic, detail

def horizontal(forecast, forecast_type, model):
    return {"Tipo": forecast_type, "Modelo": model, **{f"M+{i+1}": int(v) for i, v in enumerate(forecast["Forecast"])}}

# ============================== INTERFAZ ==============================
EXAMPLE = """Fecha\tVentas
2025-01\t1200
2025-02\t0
2025-03\t980
2025-04\t0
2025-05\t1600
2025-06\t0
2025-07\t1700
2025-08\t0
2025-09\t1250
2025-10\t0
2025-11\t1550
2025-12\t0
2026-01\t1300
2026-02\t0
2026-03\t1500
2026-04\t1650"""
for key, value in {"raw": EXAMPLE, "counter": 0, "ready": False}.items():
    if key not in st.session_state: st.session_state[key] = value

def clear_history():
    st.session_state.raw = ""; st.session_state.counter += 1; st.session_state.ready = False

with st.sidebar:
    st.header("Configuración")
    start_month = st.text_input("Mes inicial si pegas solo ventas", "2025-01")
    st.markdown("**V2:** Croston, SBA, TSB, ensemble fijo y dinámico. ML desde 18 meses y modelos estacionales desde 24.")
    for name, ok in [("Prophet", PROPHET_OK), ("LightGBM", LGBM_OK), ("XGBoost", XGBOOST_OK), ("Random Forest", RFOREST_OK)]:
        st.write(f"{name}: {'✅' if ok else '❌'}")

raw_data = st.text_area("Pega Fecha y Ventas desde Excel", st.session_state.raw, height=270, key=f"data_{st.session_state.counter}")
col1, col2 = st.columns(2)
calculate = col1.button("🚀 Calcular", type="primary", use_container_width=True)
col2.button("🧹 Limpiar", on_click=clear_history, use_container_width=True)

if calculate:
    try:
        df = parse_pasted_data(raw_data, start_month)
        if len(df) < 16: raise ValueError("Se requieren al menos 16 meses.")
        results, valid, test = run_backtesting(df)
        st.session_state.update({"ready": True, "raw": raw_data, "df": df, "results": results, "valid": valid, "test": test})
    except Exception as exc:
        st.session_state.ready = False
        st.error(str(exc))

if st.session_state.ready:
    df, results, valid, test = st.session_state.df, st.session_state.results, st.session_state.valid, st.session_state.test
    best_model = valid.iloc[0]["Modelo"]
    best_error = float(valid.iloc[0]["Error selección_%"])
    diagnosis = demand_diagnosis(df)
    confiability = forecast_confidence(df, best_error)

    st.subheader("1. Diagnóstico")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tipo de demanda", diagnosis["Tipo de demanda"])
    c2.metric("Meses con venta cero", f"{diagnosis['Meses con venta cero %']:.1f}%")
    c3.metric("Mejor modelo", best_model)
    c4.metric("Confiabilidad", f"{confiability['Confiabilidad']}/100")
    c5.metric("Nivel", f"{confiability['Indicador']} {confiability['Nivel']}")
    st.info(diagnosis["Explicación"])
    st.warning(confiability["Interpretación"])

    with st.expander("¿Cómo se calcula e interpreta la confiabilidad?"):
        st.markdown("""
El índice combina:
- **50% precisión histórica**, basada en el error del backtesting.
- **30% estabilidad de la demanda**, basada en el coeficiente de variación.
- **20% cantidad de histórico disponible**.

Niveles:
- **90 a 100: Muy Alta**
- **75 a 89,9: Alta**
- **60 a 74,9: Media**
- **40 a 59,9: Baja**
- **0 a 39,9: Muy Baja**
        """)
        st.dataframe(pd.DataFrame([confiability]), use_container_width=True)
        st.dataframe(pd.DataFrame([diagnosis]), use_container_width=True)

    if len(df) < 18: st.info("LightGBM, XGBoost y Random Forest se activan desde 18 meses.")
    if len(df) < 24: st.info("Naive estacional y Holt-Winters se activan desde 24 meses.")

    manual_model = st.selectbox("Modelo alternativo", results[results["Estado"] == "OK"]["Modelo"].tolist(), index=0)
    try:
        automatic = future_forecast(df, best_model)
        manual = future_forecast(df, manual_model)
        fixed, dynamic, weights = ensemble_forecasts(df, valid)
        output = pd.DataFrame([horizontal(automatic, "Automático", best_model), horizontal(manual, "Manual", manual_model), horizontal(fixed, "Ensemble fijo Top 4", "0,4/0,3/0,2/0,1"), horizontal(dynamic, "Ensemble dinámico Top 4", "Inverso al error")])
        st.subheader("2. Forecast horizontal a 18 meses")
        st.dataframe(output, use_container_width=True)
        st.subheader("3. Pesos del ensemble")
        st.dataframe(weights, use_container_width=True)
        st.caption("Todos los forecasts se redondean a 0 decimales después de combinar los modelos.")

        st.subheader("4. Comparativa walk-forward")
        display = results.copy()
        for col in ["MAPE_%", "sMAPE_%", "WMAPE_%", "Error selección_%"]: display[col] = display[col].round(2)
        st.dataframe(display, use_container_width=True)

        winner = results[results["Modelo"] == best_model].iloc[0]
        backtest = test.copy()
        backtest["Forecast"] = [winner[f"BT{i}"] for i in range(1, 5)]
        backtest["Error %"] = np.where(backtest["Ventas"] != 0, abs(backtest["Ventas"] - backtest["Forecast"]) / backtest["Ventas"] * 100, np.nan)
        st.subheader("5. Backtesting del ganador")
        st.dataframe(backtest, use_container_width=True)

        if PLOTLY_OK:
            fig = go.Figure()
            fig.add_scatter(x=df["Fecha"], y=df["Ventas"], name="Histórico", mode="lines+markers")
            for forecast, name, dash in [(automatic, "Automático", None), (manual, "Manual", "dash"), (fixed, "Top 4 fijo", "dot"), (dynamic, "Top 4 dinámico", "dashdot")]:
                fig.add_scatter(x=forecast["Fecha"], y=forecast["Forecast"], name=name, mode="lines+markers", line=dict(dash=dash) if dash else None)
            fig.update_layout(template="plotly_white", height=520)
            st.plotly_chart(fig, use_container_width=True)

        d1, d2, d3 = st.columns(3)
        d1.download_button("Forecast CSV", output.to_csv(index=False, sep=";").encode("utf-8-sig"), "forecast_v2.csv", "text/csv", use_container_width=True)
        d2.download_button("Modelos CSV", results.to_csv(index=False, sep=";").encode("utf-8-sig"), "modelos_v2.csv", "text/csv", use_container_width=True)
        d3.download_button("Pesos CSV", weights.to_csv(index=False, sep=";").encode("utf-8-sig"), "pesos_v2.csv", "text/csv", use_container_width=True)
    except Exception as exc:
        st.error(str(exc))
