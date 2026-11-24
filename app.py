"""Laboratorio Forecast V3: comparación de algoritmos por referencia.

Historia efectiva de hasta 48 meses; entrenamiento mínimo general de 12 meses;
backtesting rolling de hasta cuatro ventanas de cuatro meses; forecast a 18 meses.
Las métricas se calculan sin redondear las predicciones. El ranking del ensemble
fijo y dinámico y de la comparativa walk-forward procede de la misma tabla.

No hay aprendizaje global entre referencias. El backtesting se reutiliza para
seleccionar hiperparámetros: sus errores son exploratorios, no una evaluación
independiente del proceso de selección.
"""

from __future__ import annotations

import io
import json
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

try:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False

try:
    from prophet import Prophet
    PROPHET_OK = True
except ImportError:
    PROPHET_OK = False

try:
    from lightgbm import LGBMRegressor
    LGBM_OK = True
except ImportError:
    LGBM_OK = False

try:
    from xgboost import XGBRegressor
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False

try:
    from sklearn.ensemble import RandomForestRegressor
    RFOREST_OK = True
except ImportError:
    RFOREST_OK = False


# ========================== POLÍTICA COMÚN ==========================

MIN_HISTORY_MONTHS = 12
MAX_HISTORY_MONTHS = 48
BACKTEST_HORIZON = 4
MAX_BACKTEST_WINDOWS = 4
FORECAST_HORIZON = 18
ML_LAGS = (1, 2, 3, 6, 12)
ML_WINDOWS = (3, 6, 12)
ML_FLOOR_RATIO = 0.25
ML_FLOOR_MONTHS = 6
RANDOM_STATE = 1337

ALL_MODELS = [
    "Naive", "Media móvil 3", "Media móvil 6", "Suavizado exponencial",
    "Holt", "Holt amortiguado", "ARIMA simple", "Prophet", "Croston",
    "SBA", "TSB", "LightGBM", "XGBoost", "Random Forest",
    "Naive estacional 12", "Holt-Winters estacional",
]
MODEL_MIN_TRAIN = {name: MIN_HISTORY_MONTHS for name in ALL_MODELS}
MODEL_MIN_TRAIN.update({"Naive estacional 12": 24, "Holt-Winters estacional": 24})

TOP_N = 4
FIXED_WEIGHTS = np.array([.4, .3, .2, .1])
MIN_WEIGHT_ERROR = .01  # Puntos porcentuales. Solo se aplica al calcular pesos.
RANKING_COLUMN = "Error ranking_%"
SELECTION_COLUMN = "Error selección_%"
RANKING_KEYS = [RANKING_COLUMN, SELECTION_COLUMN, "WMAPE_%"]


# ========================== DATOS ==========================

def clean_number(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).strip().replace(" ", "").replace("\xa0", "")
    if not s:
        return np.nan
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def normalize_history(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df["Ventas"] = df["Ventas"].map(clean_number)
    df = df.dropna(subset=["Fecha", "Ventas"])
    if df.empty:
        raise ValueError("No se han encontrado registros con fecha y ventas válidas.")
    if (~np.isfinite(df["Ventas"])).any():
        raise ValueError("Se han encontrado ventas no finitas.")
    if (df["Ventas"] < 0).any():
        raise ValueError("Hay ventas negativas. Revisar la serie antes de ejecutar.")
    df["Fecha"] = df["Fecha"].dt.to_period("M").dt.to_timestamp()
    df = df.sort_values("Fecha").drop_duplicates("Fecha", keep="last")
    dates = pd.date_range(df["Fecha"].min(), df["Fecha"].max(), freq="MS")
    # Política original: los meses intermedios sin registro se rellenan con cero.
    df = pd.DataFrame({"Fecha": dates}).merge(df, on="Fecha", how="left")
    missing_months = int(df["Ventas"].isna().sum())
    df["Ventas"] = df["Ventas"].fillna(0.0)
    nonzero = np.flatnonzero(df["Ventas"].to_numpy(float) != 0)
    if not len(nonzero):
        raise ValueError("La referencia no presenta demanda histórica distinta de cero.")
    df = df.iloc[int(nonzero[0]):].tail(MAX_HISTORY_MONTHS).reset_index(drop=True)
    if len(df) < MIN_HISTORY_MONTHS + BACKTEST_HORIZON:
        raise ValueError(
            f"Se requieren al menos {MIN_HISTORY_MONTHS + BACKTEST_HORIZON} meses "
            f"de historia efectiva para comparar modelos con un backtest de {BACKTEST_HORIZON} meses."
        )
    df.attrs["meses_rellenados_cero"] = missing_months
    return df


def parse_pasted_data(raw_text: str, start_month: str) -> pd.DataFrame:
    if not raw_text or not raw_text.strip():
        raise ValueError("Pega una serie de fechas y ventas desde Excel.")
    frame = None
    for sep in ("\t", ";"):
        try:
            candidate = pd.read_csv(io.StringIO(raw_text.strip()), sep=sep, header=None, dtype=str, engine="python")
            if candidate.shape[1] >= 2:
                frame = candidate
                break
        except Exception:
            continue
    if frame is None:
        frame = pd.DataFrame(raw_text.strip().splitlines())
    frame = frame.dropna(how="all").reset_index(drop=True)
    if frame.empty:
        raise ValueError("Los datos pegados están vacíos.")
    header = " ".join(str(v).lower() for v in frame.iloc[0].values)
    if any(word in header for word in ("fecha", "mes", "venta", "cantidad", "sales")):
        frame = frame.iloc[1:].reset_index(drop=True)
    if frame.shape[1] >= 2:
        dates = pd.to_datetime(frame.iloc[:, 0].astype(str).str.strip(), errors="coerce")
        return normalize_history(pd.DataFrame({"Fecha": dates, "Ventas": frame.iloc[:, 1]}))
    start = pd.to_datetime(start_month + "-01", format="%Y-%m-%d", errors="raise")
    return normalize_history(pd.DataFrame({
        "Fecha": pd.date_range(start, periods=len(frame), freq="MS"),
        "Ventas": frame.iloc[:, 0],
    }))


def parse_reference_csv(uploaded) -> tuple[pd.DataFrame, str]:
    raw = uploaded.getvalue()
    df = None
    for enc in ("utf-8-sig", "cp1252"):
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding=enc, dtype=str)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise ValueError("No se puede leer el CSV. Guardarlo como UTF-8.")
    df.columns = [str(c).strip() for c in df.columns]
    colmap = {str(c).lower(): c for c in df.columns}
    date_col = next((colmap[x] for x in ("fecha", "mes", "date", "ds") if x in colmap), None)
    value_col = next((colmap[x] for x in ("ventas", "venta", "realidad", "cantidad", "y") if x in colmap), None)
    reference_col = next((colmap[x] for x in ("referencia", "estadistico", "sku", "producto", "unique_id") if x in colmap), None)
    if not date_col or not value_col:
        raise ValueError("El CSV debe tener columnas Fecha y Ventas (formato largo).")
    if reference_col is None:
        df["Referencia"] = "Referencia única"
        reference_col = "Referencia"
    df[reference_col] = df[reference_col].fillna("Sin referencia").astype(str)
    return df.rename(columns={date_col: "Fecha", value_col: "Ventas"}), reference_col


# ========================== MÉTRICAS ==========================

def mape(actual: Any, pred: Any) -> float:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    mask = a != 0
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100) if mask.any() else np.nan


def smape(actual: Any, pred: Any) -> float:
    # Un par (0, 0) aporta error cero.
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    den = np.abs(a) + np.abs(p)
    den = np.where(den == 0.0, 1.0, den)
    return float(np.mean(2.0 * np.abs(a - p) / den) * 100)


def wmape(actual: Any, pred: Any) -> float:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    den = float(np.abs(a).sum())
    if den == 0:
        return 0.0 if np.allclose(p, 0.0) else 100.0
    return float(np.abs(a - p).sum() / den * 100)


def round_zero(values: Any) -> np.ndarray:
    a = np.asarray(values, float)
    if not np.isfinite(a).all():
        raise ValueError("Predicciones finales con NaN/inf.")
    return np.rint(np.clip(a, 0.0, None)).astype(int)


# ========================== DIAGNÓSTICO Y CONFIABILIDAD ==========================

def demand_diagnosis(df: pd.DataFrame) -> dict[str, Any]:
    y = df["Ventas"].astype(float)
    mean = y.mean()
    cv = y.std(ddof=0) / mean if mean else np.nan
    zero_ratio = (y == 0).mean()
    slope = np.polyfit(np.arange(len(y)), y, 1)[0] / mean if len(y) > 1 and mean else 0
    seasonal, strength = False, np.nan
    if len(y) >= 24 and mean:
        strength = df.assign(Mes=df["Fecha"].dt.month).groupby("Mes")["Ventas"].mean().std(ddof=0) / mean
        seasonal = strength >= .20
    first4, last4 = y.head(4).mean(), y.tail(4).mean()
    last3 = y.tail(3).mean()
    previous6 = y.iloc[-9:-3].mean() if len(y) >= 9 else np.nan
    if zero_ratio >= .40:
        kind, explanation = "Intermitente", "Proporción elevada de meses con venta cero; comparar Croston, SBA y TSB."
    elif len(y) >= 8 and first4 > 0 and last4 >= 1.8 * first4 and slope > .03:
        kind, explanation = "Posible lanzamiento", "La demanda reciente crece respecto del inicio; revisar distribución y ramp-up."
    elif len(y) >= 9 and previous6 > 0 and last3 <= .5 * previous6 and slope < -.03:
        kind, explanation = "Posible phase-out", "Caída reciente; validar descatalogación o sustitución."
    elif seasonal:
        kind, explanation = "Estacional", "Hay diferencias por mes calendario; revisar modelos estacionales."
    elif not pd.isna(cv) and cv <= .20:
        kind, explanation = "Estable", "La variabilidad histórica es baja."
    elif not pd.isna(cv) and cv >= .70:
        kind, explanation = "Volátil", "La variabilidad es elevada; revisar promociones y pedidos extraordinarios."
    elif slope >= .03:
        kind, explanation = "Tendencia creciente", "La pendiente histórica mensual es positiva."
    elif slope <= -.03:
        kind, explanation = "Tendencia decreciente", "La pendiente histórica mensual es negativa."
    else:
        kind, explanation = "Sin patrón dominante", "La serie no presenta un patrón dominante claro."
    return {"Tipo de demanda": kind, "Explicación": explanation, "CV": cv,
            "Meses con venta cero %": zero_ratio * 100, "Tendencia mensual %": slope * 100,
            "Estacionalidad": "Sí" if seasonal else "No", "Fuerza estacional": strength}


def confidence_level(score: float):
    if score >= 90:
        return "Muy Alta", "🟢", "Score heurístico alto; validar con Demand Planning."
    if score >= 75:
        return "Alta", "🟢", "Precisión y estabilidad favorables en el backtesting."
    if score >= 60:
        return "Media", "🟡", "Validar con información comercial."
    if score >= 40:
        return "Baja", "🟠", "Incertidumbre significativa; requiere revisión."
    return "Muy Baja", "🔴", "No utilizar automáticamente sin validación comercial."


def forecast_confidence(df: pd.DataFrame, error_value: float) -> dict[str, Any]:
    y, n = df["Ventas"].astype(float), len(df)
    cv = y.std(ddof=0) / y.mean() if y.mean() else np.nan
    error_score = max(0., 100. - 2. * error_value) if np.isfinite(error_value) else 0.
    stability_score = 0 if pd.isna(cv) else 100 if cv <= .20 else 0 if cv >= 1 else 100 * (1 - (cv - .20) / .80)
    history_score = (100 if n >= 48 else 85 + (n - 36) * 1.25 if n >= 36 else
                     70 + (n - 24) * 1.25 if n >= 24 else 50 + (n - 16) * 2.5)
    score = round(.5 * error_score + .3 * stability_score + .2 * history_score, 1)
    level, emoji, interpretation = confidence_level(score)
    return {"Confiabilidad": score, "Nivel": level, "Indicador": emoji, "Interpretación": interpretation,
            "Score precisión": round(error_score, 1), "Score estabilidad": round(stability_score, 1),
            "Score histórico": round(history_score, 1)}


# ========================== CONFIGURACIONES LIGERAS ==========================

def candidate_configs(name: str) -> list[dict[str, Any]]:
    if name == "Naive":
        return [{}]
    if name.startswith("Media móvil"):
        return [{"peso": x} for x in ("uniforme", "lineal", "exponencial")]
    if name == "Naive estacional 12":
        return [{"fuerza_estacional": x} for x in (1., .80, .60)]
    if name == "Suavizado exponencial":
        return [{"alpha": a} for a in (.15, .40, .70)]
    if name == "Holt":
        return [{"alpha": a, "beta": b} for a, b in ((.20, .05), (.40, .10), (.60, .20))]
    if name == "Holt amortiguado":
        return [{"alpha": .35, "beta": .10, "phi": p} for p in (.80, .90, .98)]
    if name == "Holt-Winters estacional":
        return [{"trend": t, "seasonal": s} for t, s in (("add", "add"), (None, "add"), ("add", "mul"))]
    if name == "ARIMA simple":
        return [{"order": order} for order in ((0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1))]
    if name == "Prophet":
        return [
            {"seasonality_mode": "additive", "changepoint_prior_scale": .05, "seasonality_prior_scale": 5., "use_half_seasonality": False},
            {"seasonality_mode": "multiplicative", "changepoint_prior_scale": .05, "seasonality_prior_scale": 10., "use_half_seasonality": False},
            {"seasonality_mode": "additive", "changepoint_prior_scale": .25, "seasonality_prior_scale": 10., "use_half_seasonality": True},
            {"seasonality_mode": "multiplicative", "changepoint_prior_scale": .25, "seasonality_prior_scale": 5., "use_half_seasonality": True},
        ]
    if name in ("Croston", "SBA"):
        return [{"alpha": a} for a in (.05, .15, .30)]
    if name == "TSB":
        return [{"alpha": a, "beta": b} for a, b in ((.05, .05), (.15, .10), (.30, .20))]
    if name == "LightGBM":
        return [
            {"n_estimators": 150, "learning_rate": .05, "num_leaves": 15, "max_depth": -1, "min_child_samples": 1, "reg_lambda": 0.},
            {"n_estimators": 300, "learning_rate": .03, "num_leaves": 31, "max_depth": -1, "min_child_samples": 1, "reg_lambda": .5},
            {"n_estimators": 200, "learning_rate": .06, "num_leaves": 15, "max_depth": 8, "min_child_samples": 5, "reg_lambda": .5},
        ]
    if name == "XGBoost":
        return [
            {"n_estimators": 150, "learning_rate": .05, "max_depth": 2, "min_child_weight": 1},
            {"n_estimators": 250, "learning_rate": .03, "max_depth": 3, "min_child_weight": 1},
            {"n_estimators": 200, "learning_rate": .06, "max_depth": 3, "min_child_weight": 3},
        ]
    if name == "Random Forest":
        return [
            {"n_estimators": 120, "max_depth": 3, "min_samples_leaf": 1},
            {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 2},
            {"n_estimators": 150, "max_depth": None, "min_samples_leaf": 3},
        ]
    raise ValueError(f"No existe grid para {name}")


def config_string(config: dict) -> str:
    return json.dumps(config, ensure_ascii=False, sort_keys=True)


# ========================== DEMANDA INTERMITENTE ==========================

def croston_rate(y: np.ndarray, alpha: float, variant: str, beta: float = .05) -> float:
    y = np.asarray(y, float)
    nz = np.flatnonzero(y > 0)
    if not len(nz):
        return 0.
    if variant == "TSB":
        size, probability = float(y[nz[0]]), 1.
        for v in y[nz[0] + 1:]:
            present = float(v > 0)
            probability = beta * present + (1 - beta) * probability
            if v > 0:
                size = alpha * v + (1 - alpha) * size
        return max(0., float(probability * size))
    size, interval, gap = float(y[nz[0]]), 1., 0
    for v in y[nz[0] + 1:]:
        gap += 1
        if v > 0:
            size = alpha * v + (1 - alpha) * size
            interval = alpha * gap + (1 - alpha) * interval
            gap = 0
    rate = size / max(interval, 1e-9)
    return max(0., float(rate * (1 - alpha / 2) if variant == "SBA" else rate))


# ========================== ML ==========================

def ml_features(history: np.ndarray, dates: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    y = np.asarray(history, float)
    ds = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    f = pd.DataFrame({"ds": ds, "y": y})
    f["month"] = f["ds"].dt.month.astype(int)
    f["quarter"] = f["ds"].dt.quarter.astype(int)
    f["year"] = f["ds"].dt.year.astype(int)
    f["mon_sin"] = np.sin(2 * np.pi * f["month"] / 12.)
    f["mon_cos"] = np.cos(2 * np.pi * f["month"] / 12.)
    for k in ML_LAGS:
        f[f"lag_{k}"] = f["y"].shift(k)
    for w in ML_WINDOWS:
        f[f"rollmean_{w}"] = f["y"].shift(1).rolling(w, min_periods=1).mean()
    f["expmean"] = f["y"].shift(1).expanding(min_periods=1).mean()
    wanted = ["month", "quarter", "year", "mon_sin", "mon_cos", "expmean"]
    wanted += [f"lag_{k}" for k in ML_LAGS]
    wanted += [f"rollmean_{w}" for w in ML_WINDOWS]
    usable = [c for c in wanted if f[c].notna().sum() >= 3 and f[c].nunique(dropna=True) >= 2]
    return f, usable or ["month", "mon_sin", "mon_cos"]


def ml_future_features(history: list[float], next_date: pd.Timestamp, features: list[str]) -> pd.DataFrame:
    m = next_date.month
    f = {"month": m, "quarter": next_date.quarter, "year": next_date.year,
         "mon_sin": np.sin(2 * np.pi * m / 12.), "mon_cos": np.cos(2 * np.pi * m / 12.),
         "expmean": float(np.mean(history)) if history else np.nan}
    for k in ML_LAGS:
        f[f"lag_{k}"] = history[-k] if len(history) >= k else np.nan
    for w in ML_WINDOWS:
        f[f"rollmean_{w}"] = float(np.mean(history[-w:])) if history else np.nan
    return pd.DataFrame([f])[features].astype("float32")


def ml_forecast(name: str, y: np.ndarray, dates: pd.Series, horizon: int, config: dict) -> np.ndarray:
    if name == "LightGBM" and not LGBM_OK:
        raise ImportError("LightGBM no instalado")
    if name == "XGBoost" and not XGBOOST_OK:
        raise ImportError("XGBoost no instalado")
    if name == "Random Forest" and not RFOREST_OK:
        raise ImportError("scikit-learn no instalado")
    train, features = ml_features(y, dates)
    X = train[features].astype("float32")
    if name == "Random Forest":
        X = X.fillna(-1.0)
    target = np.log1p(np.clip(np.asarray(y, float), 0., None))
    weights = np.linspace(1., 2., len(y))
    if name == "LightGBM":
        model = LGBMRegressor(**config, n_jobs=1, subsample=.8, colsample_bytree=.8,
                              min_data_in_bin=1, max_bin=63, verbosity=-1,
                              force_col_wise=True, random_state=RANDOM_STATE)
    elif name == "XGBoost":
        model = XGBRegressor(**config, n_jobs=1, objective="reg:squarederror",
                             subsample=.8, colsample_bytree=.8, reg_lambda=.5,
                             random_state=RANDOM_STATE, verbosity=0)
    else:
        model = RandomForestRegressor(**config, n_jobs=1, random_state=RANDOM_STATE)
    model.fit(X, target, sample_weight=weights)
    fitted = np.asarray(model.predict(X), float)
    smear = float(np.mean(np.exp(target - fitted)))
    if not np.isfinite(smear) or smear <= 0:
        smear = 1.
    values = list(np.asarray(y, float))
    last_date = pd.to_datetime(pd.Series(dates).iloc[-1])
    out = []
    for step in range(1, horizon + 1):
        xf = ml_future_features(values, last_date + pd.DateOffset(months=step), features)
        if name == "Random Forest":
            xf = xf.fillna(-1.0)
        yh = max(0., float(np.expm1(model.predict(xf)[0]) * smear))
        floor = ML_FLOOR_RATIO * float(np.median(values[-ML_FLOOR_MONTHS:]))
        yh = max(yh, floor)
        out.append(yh)
        values.append(yh)  # Predicción recursiva, sin usar observaciones futuras.
    return np.asarray(out, float)


# ========================== PROPHET ==========================

def prophet_growth(y: np.ndarray) -> str:
    if len(y) < 12:
        return "linear"
    recent = np.mean(y[-6:])
    previous = np.mean(y[:-6]) if len(y) > 6 else recent
    ratio = recent / previous if previous > 0 else 1.
    growth = (y[-1] - y[0]) / max(1., np.mean(y))
    if np.allclose(y[-6:], y[-1], rtol=.05):
        return "logistic"
    return "logistic" if ratio < 1.05 and growth < .2 else "linear"


def prophet_forecast(y: np.ndarray, dates: pd.Series, horizon: int, config: dict) -> np.ndarray:
    if not PROPHET_OK:
        raise ImportError("Prophet no instalado")
    growth = prophet_growth(y)
    df_train = pd.DataFrame({"ds": pd.to_datetime(pd.Series(dates)).values, "y": y})
    model = Prophet(growth=growth, yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, n_changepoints=10, interval_width=.8,
                    seasonality_mode=config["seasonality_mode"],
                    changepoint_prior_scale=config["changepoint_prior_scale"],
                    seasonality_prior_scale=config["seasonality_prior_scale"])
    cap = None
    if growth == "logistic":
        cap = max(1., float(np.max(y)) * 1.3)
        df_train["cap"] = cap
        df_train["floor"] = 0.
    if config["use_half_seasonality"]:
        model.add_seasonality(name="half", period=182.5, fourier_order=3)
    model.fit(df_train)
    future = model.make_future_dataframe(periods=horizon, freq="MS", include_history=False)
    if cap is not None:
        future["cap"] = cap
        future["floor"] = 0.
    return model.predict(future)["yhat"].to_numpy(float)


# ========================== EJECUCIÓN DE ALGORITMOS ==========================

def run_model(name: str, y: Any, dates: Any, horizon: int, config: dict) -> np.ndarray:
    y = np.asarray(y, float)
    if len(y) < MODEL_MIN_TRAIN[name]:
        raise ValueError(f"{name}: requiere {MODEL_MIN_TRAIN[name]} meses de entrenamiento")
    if name == "Naive":
        pred = np.repeat(y[-1], horizon)
    elif name.startswith("Media móvil"):
        w = 3 if name.endswith("3") else 6
        vals = y[-w:]
        if config["peso"] == "lineal":
            weights = np.arange(1, len(vals) + 1)
        elif config["peso"] == "exponencial":
            weights = np.exp(np.linspace(-1.5, 0, len(vals)))
        else:
            weights = np.ones(len(vals))
        pred = np.repeat(np.average(vals, weights=weights), horizon)
    elif name == "Naive estacional 12":
        recent = y[-12:]
        avg = float(recent.mean())
        strength = config["fuerza_estacional"]
        pred = np.asarray([avg + strength * (recent[i % 12] - avg) for i in range(horizon)])
    elif name in ("Croston", "SBA", "TSB"):
        pred = np.repeat(croston_rate(y, config["alpha"], name, config.get("beta", .05)), horizon)
    elif name == "Suavizado exponencial":
        if not STATSMODELS_OK:
            raise ImportError("statsmodels no instalado")
        pred = np.asarray(SimpleExpSmoothing(y, initialization_method="estimated")
                          .fit(smoothing_level=config["alpha"], optimized=False).forecast(horizon))
    elif name in ("Holt", "Holt amortiguado"):
        if not STATSMODELS_OK:
            raise ImportError("statsmodels no instalado")
        damped = name == "Holt amortiguado"
        fitted = Holt(y, damped_trend=damped, initialization_method="estimated").fit(
            smoothing_level=config["alpha"], smoothing_trend=config["beta"],
            damping_trend=config.get("phi") if damped else None, optimized=False)
        pred = np.asarray(fitted.forecast(horizon))
    elif name == "Holt-Winters estacional":
        if not STATSMODELS_OK:
            raise ImportError("statsmodels no instalado")
        if config["seasonal"] == "mul" and (y <= 0).any():
            raise ValueError("Estacionalidad multiplicativa incompatible con ventas 0")
        fit = ExponentialSmoothing(y, trend=config["trend"], seasonal=config["seasonal"],
                                   seasonal_periods=12, initialization_method="estimated").fit(optimized=True)
        pred = np.asarray(fit.forecast(horizon))
    elif name == "ARIMA simple":
        if not STATSMODELS_OK:
            raise ImportError("statsmodels no instalado")
        fit = SARIMAX(y, order=tuple(config["order"]), enforce_stationarity=False,
                      enforce_invertibility=False).fit(disp=False, maxiter=100)
        pred = np.asarray(fit.forecast(horizon))
    elif name == "Prophet":
        pred = prophet_forecast(y, pd.Series(dates), horizon, config)
    elif name in ("LightGBM", "XGBoost", "Random Forest"):
        pred = ml_forecast(name, y, pd.Series(dates), horizon, config)
    else:
        raise ValueError(f"Modelo no reconocido: {name}")
    pred = np.asarray(pred, float).ravel()
    if len(pred) != horizon or not np.isfinite(pred).all():
        raise ValueError(f"{name}: forecast incompleto o no finito")
    return np.clip(pred, 0.0, None)


def available_models(n_months: int) -> list[str]:
    names = [n for n in ALL_MODELS if n not in (
        "LightGBM", "XGBoost", "Random Forest", "Naive estacional 12", "Holt-Winters estacional"
    )]
    if n_months >= 18:
        names.extend(("LightGBM", "XGBoost", "Random Forest"))
    if n_months >= 24:
        names.extend(("Naive estacional 12", "Holt-Winters estacional"))
    return names


# ========================== BACKTEST Y RANKING COMÚN ==========================

def get_backtest_windows(n_history: int, max_windows: int = MAX_BACKTEST_WINDOWS) -> list[dict[str, int]]:
    possible = max(0, n_history - (MIN_HISTORY_MONTHS + BACKTEST_HORIZON) + 1)
    n_windows = min(max_windows, possible)
    windows = []
    for i, months_back in enumerate(range(n_windows - 1, -1, -1), start=1):
        valid_end = n_history - months_back
        valid_start = valid_end - BACKTEST_HORIZON
        windows.append({"window": i, "train_end": valid_start, "valid_start": valid_start, "valid_end": valid_end})
    return windows


@dataclass
class LabResult:
    models: pd.DataFrame
    valid: pd.DataFrame
    configs: pd.DataFrame
    backtesting: pd.DataFrame
    forecasts: dict[str, np.ndarray]
    window_count: int


def build_ranking(models: pd.DataFrame) -> pd.DataFrame:
    """Único punto de ordenación. No redondear antes de clasificar."""
    if RANKING_COLUMN not in models.columns:
        return pd.DataFrame()
    valid = (models.loc[models["Estado"].eq("OK") & models[RANKING_COLUMN].notna()]
             .sort_values(RANKING_KEYS, ascending=True)
             .reset_index(drop=True))
    valid.insert(0, "Posición", range(1, len(valid) + 1))
    return valid


def run_backtesting(df: pd.DataFrame, max_windows: int = MAX_BACKTEST_WINDOWS,
                    progress_callback: Callable[[int, int, str], None] | None = None) -> LabResult:
    windows = get_backtest_windows(len(df), max_windows)
    if not windows:
        raise ValueError("Sin ventanas evaluables para backtesting.")
    names = available_models(len(df))
    total = sum(len(candidate_configs(n)) for n in names)
    done = 0
    model_rows, config_rows, bt_rows = [], [], []
    forecasts: dict[str, np.ndarray] = {}

    for name in names:
        candidates = candidate_configs(name)
        eligible = all(w["train_end"] >= MODEL_MIN_TRAIN[name] for w in windows)
        if not eligible:
            reason = (f"No comparable: cada ventana exige >= {MODEL_MIN_TRAIN[name]} meses "
                      "de entrenamiento; manteniendo las mismas ventanas para todos.")
            model_rows.append({"Modelo": name, "Estado": "No comparable", "Detalle": reason,
                               "Configs probadas": 0, "Configs válidas": 0})
            for cfg in candidates:
                config_rows.append({"Modelo": name, "Configuración": config_string(cfg), "Estado": "No comparable",
                                    "Detalle": reason, "Ventanas válidas": 0})
                done += 1
                if progress_callback:
                    progress_callback(done, total, name)
            continue

        evaluated: list[dict[str, Any]] = []
        for cfg in candidates:
            cfg_text = config_string(cfg)
            evaluation_rows = []
            reason = ""
            for w in windows:
                train = df.iloc[:w["train_end"]]
                actual = df.iloc[w["valid_start"]:w["valid_end"]]
                try:
                    pred = run_model(name, train["Ventas"].values, train["Fecha"], BACKTEST_HORIZON, cfg)
                    for j in range(BACKTEST_HORIZON):
                        evaluation_rows.append({"Modelo": name, "Configuración": cfg_text,
                                                "Ventana": w["window"], "Horizonte": j + 1,
                                                "Fecha": actual["Fecha"].iloc[j],
                                                "Real": float(actual["Ventas"].iloc[j]),
                                                "Predicción": float(pred[j])})
                except Exception as exc:
                    reason = f"Ventana {w['window']}: {str(exc)[:200]}"
                    break

            if len(evaluation_rows) == len(windows) * BACKTEST_HORIZON:
                a = np.asarray([x["Real"] for x in evaluation_rows])
                p = np.asarray([x["Predicción"] for x in evaluation_rows])
                scores = {"MAPE_%": mape(a, p), "sMAPE_%": smape(a, p), "WMAPE_%": wmape(a, p)}
                evaluated.append({"config": cfg, "text": cfg_text, "metrics": scores, "rows": evaluation_rows})
                status = "OK"
            else:
                scores = {"MAPE_%": np.nan, "sMAPE_%": np.nan, "WMAPE_%": np.nan}
                status = "Error"

            config_rows.append({"Modelo": name, "Configuración": cfg_text, **scores, "Estado": status,
                                "Ventanas válidas": len(evaluation_rows) // BACKTEST_HORIZON, "Detalle": reason})
            done += 1
            if progress_callback:
                progress_callback(done, total, name)

        evaluated.sort(key=lambda item: (item["metrics"]["sMAPE_%"], item["metrics"]["WMAPE_%"]))
        selected = None
        final_issue = ""
        for item in evaluated:
            try:
                future = run_model(name, df["Ventas"].values, df["Fecha"], FORECAST_HORIZON, item["config"])
                selected = item
                forecasts[name] = future
                break
            except Exception as exc:
                final_issue = f"Forecast final fallido para config {item['text']}: {str(exc)[:150]}"

        if selected is None:
            detail = final_issue or next(
                (r["Detalle"] for r in reversed(config_rows) if r["Modelo"] == name and r["Detalle"]),
                "Sin configuraciones válidas",
            )
            model_rows.append({"Modelo": name, "Estado": "Error", "Detalle": detail,
                               "Configs probadas": len(candidates), "Configs válidas": len(evaluated)})
            continue

        errors = np.asarray([item["metrics"]["sMAPE_%"] for item in evaluated], dtype=float)
        median_error = float(np.median(errors))
        last_window = windows[-1]["window"]
        last_pred = [r["Predicción"] for r in selected["rows"] if r["Ventana"] == last_window]
        model_rows.append({
            "Modelo": name, **selected["metrics"], "Métrica": "sMAPE",
            SELECTION_COLUMN: selected["metrics"]["sMAPE_%"],
            RANKING_COLUMN: median_error,
            "Mediana sMAPE configs_%": median_error,
            "Peor sMAPE configs_%": float(np.max(errors)),
            "Desviación sMAPE configs_%": float(np.std(errors)),
            "Mejor configuración": selected["text"], "Configs probadas": len(candidates),
            "Configs válidas": len(evaluated), "Ventanas": len(windows),
            **{f"BT{i + 1}": last_pred[i] for i in range(BACKTEST_HORIZON)},
            "Estado": "OK", "Detalle": "",
        })
        bt_rows.extend(selected["rows"])

    models = pd.DataFrame(model_rows)
    valid = build_ranking(models)
    if valid.empty:
        raise ValueError("No hay algoritmos con backtesting completo y forecast final válido.")
    return LabResult(models, valid, pd.DataFrame(config_rows), pd.DataFrame(bt_rows), forecasts, len(windows))


def walk_forward_comparison(lab: LabResult) -> pd.DataFrame:
    """Mismo ranking y misma precisión que el ensemble; no reordena."""
    ranked = lab.valid
    unranked = lab.models.loc[~lab.models["Modelo"].isin(ranked["Modelo"])].copy()
    unranked["Posición"] = pd.Series(pd.NA, index=unranked.index, dtype="Int64")
    comparison = pd.concat([ranked, unranked], ignore_index=True)
    comparison["Posición"] = comparison["Posición"].astype("Int64")
    comparison.insert(2, "Top 4", comparison["Posición"].le(TOP_N).fillna(False))
    front = ["Posición", "Modelo", "Top 4", RANKING_COLUMN, SELECTION_COLUMN, "WMAPE_%"]
    return comparison.loc[:, front + [col for col in comparison if col not in front]]


# ========================== ENSEMBLES Y EXPORTACIONES ==========================

def future_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.date_range(df["Fecha"].max() + pd.DateOffset(months=1), periods=FORECAST_HORIZON, freq="MS")


def future_forecast(df: pd.DataFrame, name: str, forecasts: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame({"Fecha": future_dates(df), "Forecast": round_zero(forecasts[name])})


def ensemble_forecasts(df: pd.DataFrame, lab: LabResult):
    top = lab.valid.head(TOP_N)
    if len(top) < TOP_N:
        raise ValueError("El ensemble necesita al menos cuatro algoritmos con resultado válido.")
    errors = top[RANKING_COLUMN].to_numpy(dtype=float)
    inv = 1. / np.maximum(errors, MIN_WEIGHT_ERROR)
    dynamic_weights = inv / inv.sum()
    mat = np.vstack([lab.forecasts[name] for name in top["Modelo"]])
    dates = future_dates(df)
    fixed = pd.DataFrame({"Fecha": dates, "Forecast": round_zero(FIXED_WEIGHTS @ mat)})
    dynamic = pd.DataFrame({"Fecha": dates, "Forecast": round_zero(dynamic_weights @ mat)})
    detail = top[["Posición", "Modelo", RANKING_COLUMN]].copy()
    detail["Peso fijo"] = FIXED_WEIGHTS
    detail["Peso dinámico"] = dynamic_weights
    return fixed, dynamic, detail


def horizontal(forecast: pd.DataFrame, forecast_type: str, model: str) -> dict:
    return {"Tipo": forecast_type, "Modelo": model,
            **{f"M+{i + 1}": int(v) for i, v in enumerate(forecast["Forecast"])}}


def as_csv(data: pd.DataFrame) -> bytes:
    return data.to_csv(index=False, sep=";").encode("utf-8-sig")


# ========================== INTERFAZ STREAMLIT ==========================

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


def clear_history():
    st.session_state.raw = ""
    st.session_state.counter += 1
    st.session_state.ready = False


def show_results(df: pd.DataFrame, lab: LabResult, reference: str):
    valid = lab.valid
    best_model = str(valid.iloc[0]["Modelo"])
    best_error = float(valid.iloc[0][RANKING_COLUMN])
    diagnosis = demand_diagnosis(df)
    confidence = forecast_confidence(df, best_error)

    st.subheader("1. Diagnóstico")
    st.caption(f"Referencia: {reference} · {len(df)} meses efectivos · {lab.window_count} ventana(s) de cuatro meses · "
               "ranking por mediana de sMAPE entre configuraciones válidas.")
    if df.attrs.get("meses_rellenados_cero", 0):
        st.warning(f"Se han rellenado con cero {df.attrs['meses_rellenados_cero']} meses sin registro. "
                   "Comprobar si son ceros reales o datos ausentes.")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tipo de demanda", diagnosis["Tipo de demanda"])
    c2.metric("Meses con venta cero", f"{diagnosis['Meses con venta cero %']:.1f}%")
    c3.metric("Mejor modelo", best_model)
    c4.metric("Confiabilidad", f"{confidence['Confiabilidad']}/100")
    c5.metric("Nivel", f"{confidence['Indicador']} {confidence['Nivel']}")
    st.info(diagnosis["Explicación"])
    st.warning(confidence["Interpretación"])
    st.caption("La confiabilidad es un indicador heurístico basado en los mismos cortes utilizados para seleccionar parámetros; "
               "no es una probabilidad calibrada de acierto.")
    with st.expander("¿Cómo se calcula e interpreta la confiabilidad?"):
        st.markdown("""El índice combina **50% precisión histórica (sMAPE)**, **30% estabilidad (CV)** y **20% cantidad de historia**.

Niveles: Muy Alta (90–100), Alta (75–89,9), Media (60–74,9), Baja (40–59,9), Muy Baja (0–39,9).""")
        st.dataframe(pd.DataFrame([confidence]), use_container_width=True)
        st.dataframe(pd.DataFrame([diagnosis]), use_container_width=True)

    if len(df) < 18:
        st.info("LightGBM, XGBoost y Random Forest se activan desde 18 meses.")
    if len(df) < 24:
        st.info("Naive estacional y Holt-Winters se activan desde 24 meses.")
    if lab.models["Estado"].eq("No comparable").any():
        st.info("Algunos algoritmos no pueden competir todavía: necesitan suficiente entrenamiento en todas las ventanas. "
                "Se muestran en la comparativa con el motivo.")

    manual_model = st.selectbox("Modelo alternativo", valid["Modelo"].tolist(), index=0)
    automatic = future_forecast(df, best_model, lab.forecasts)
    manual = future_forecast(df, manual_model, lab.forecasts)
    output_rows = [horizontal(automatic, "Automático", best_model), horizontal(manual, "Manual", manual_model)]
    plot_series = [(automatic, "Automático", None), (manual, "Manual", "dash")]
    weights = None
    if len(valid) >= TOP_N:
        fixed, dynamic, weights = ensemble_forecasts(df, lab)
        output_rows.extend([
            horizontal(fixed, "Ensemble fijo Top 4", "0,4/0,3/0,2/0,1"),
            horizontal(dynamic, "Ensemble dinámico Top 4", "Inverso al error de ranking"),
        ])
        plot_series.extend([(fixed, "Top 4 fijo", "dot"), (dynamic, "Top 4 dinámico", "dashdot")])
    output = pd.DataFrame(output_rows)

    st.subheader("2. Forecast horizontal a 18 meses")
    st.dataframe(output, use_container_width=True)

    st.subheader("3. Pesos del ensemble")
    if weights is None:
        st.info("No hay cuatro modelos válidos. Se muestra la comparativa, pero no se calcula el ensemble Top 4.")
    else:
        weights_display = weights.copy()
        weights_display[RANKING_COLUMN] = weights_display[RANKING_COLUMN].round(2)
        st.dataframe(weights_display, use_container_width=True, hide_index=True)
        st.caption("El peso dinámico es inverso al error de ranking, normalizado para que los pesos sumen 1. "
                   "Se aplica un mínimo de 0,01 puntos porcentuales solo al calcular el peso; "
                   "el error mostrado es el mismo que en la comparativa.")
    st.caption("El forecast de salida se redondea a unidades; las métricas se calculan con las predicciones sin redondear.")

    st.subheader("4. Comparativa walk-forward")
    comparison = walk_forward_comparison(lab)
    display = comparison.copy()
    for col in ("MAPE_%", "sMAPE_%", "WMAPE_%", SELECTION_COLUMN, RANKING_COLUMN,
                "Mediana sMAPE configs_%", "Peor sMAPE configs_%", "Desviación sMAPE configs_%"):
        if col in display:
            display[col] = display[col].round(2)
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption("Mismo orden que el ensemble: menor Error ranking_% (mediana del sMAPE de las configuraciones válidas). "
               "Desempates: menor Error selección_% y después menor WMAPE_%. "
               "Error selección_% y sMAPE_% corresponden a la configuración elegida para el forecast final. "
               "El orden se determina antes del redondeo. La marca Top 4 identifica las primeras posiciones, "
               "aunque solo se calcula el ensemble si hay cuatro modelos válidos.")

    with st.expander("🔬 Barrido por configuración: todos los parámetros evaluados", expanded=False):
        cfg_display = lab.configs.copy()
        for col in ("MAPE_%", "sMAPE_%", "WMAPE_%"):
            if col in cfg_display:
                cfg_display[col] = cfg_display[col].round(2)
        st.dataframe(cfg_display, use_container_width=True)
        st.caption("Un algoritmo solo compite si una configuración completa todos los cortes de validación "
                   "y produce un forecast final válido. Las combinaciones no forman una búsqueda exhaustiva.")
    st.caption("Se seleccionan hiperparámetros y se calcula el ranking con el mismo rolling backtest. "
               "Es útil para explorar, pero el resultado puede ser optimista. Validar con meses nuevos o un holdout separado.")

    winner = lab.backtesting.loc[lab.backtesting["Modelo"].eq(best_model)]
    last_window = int(winner["Ventana"].max())
    backtest = winner.loc[winner["Ventana"].eq(last_window), ["Fecha", "Real", "Predicción", "Horizonte"]].copy()
    backtest = backtest.rename(columns={"Real": "Ventas", "Predicción": "Forecast"})
    backtest["Error %"] = (backtest["Ventas"] - backtest["Forecast"]).abs() / backtest["Ventas"].replace(0, np.nan) * 100
    st.subheader("5. Backtesting del ganador")
    st.caption("Última ventana: cuatro meses futuros pronosticados desde un único corte. "
               "El CSV adicional incluye todos los cortes.")
    st.dataframe(backtest, use_container_width=True)

    if PLOTLY_OK:
        fig = go.Figure()
        fig.add_scatter(x=df["Fecha"], y=df["Ventas"], name="Histórico", mode="lines+markers")
        for forecast, label, dash in plot_series:
            fig.add_scatter(x=forecast["Fecha"], y=forecast["Forecast"], name=label,
                            mode="lines+markers", line=dict(dash=dash) if dash else None)
        fig.update_layout(template="plotly_white", height=520)
        st.plotly_chart(fig, use_container_width=True)

    d1, d2, d3 = st.columns(3)
    d1.download_button("Forecast CSV", as_csv(output), "forecast_v3.csv", "text/csv", use_container_width=True)
    d2.download_button("Modelos CSV", as_csv(comparison), "modelos_v3.csv", "text/csv", use_container_width=True)
    if weights is not None:
        d3.download_button("Pesos CSV", as_csv(weights), "pesos_v3.csv", "text/csv", use_container_width=True)
    e1, e2 = st.columns(2)
    e1.download_button("Barrido parámetros CSV", as_csv(lab.configs),
                       "barrido_configuraciones_v3.csv", "text/csv", use_container_width=True)
    e2.download_button("Backtesting completo CSV", as_csv(lab.backtesting),
                       "backtesting_completo_v3.csv", "text/csv", use_container_width=True)


def main():
    st.set_page_config(page_title="Forecast automático V3", page_icon="📈", layout="wide")
    st.title("📈 Forecast automático de ventas V3")
    st.caption("Laboratorio por referencia · barrido ligero de hiperparámetros · backtesting rolling de 4 meses · ensembles Top 4")

    for key, value in {"raw": EXAMPLE, "counter": 0, "ready": False}.items():
        if key not in st.session_state:
            st.session_state[key] = value

    with st.sidebar:
        st.header("Configuración")
        start_month = st.text_input("Mes inicial si pegas solo ventas", "2025-01")
        max_windows = st.select_slider("Ventanas de backtesting (4 meses cada una)",
                                       options=list(range(1, MAX_BACKTEST_WINDOWS + 1)), value=MAX_BACKTEST_WINDOWS)
        st.markdown("**V3:** Barrido ligero por algoritmo; ranking y ensembles Top 4 por mediana de sMAPE.")
        st.caption("Historia efectiva: hasta 48 meses; al menos 16 para evaluar. Modelos ML desde 18 meses, "
                   "estacionales desde 24; para competir necesitan historia suficiente en todos los cortes.")
        for label, ok in (("statsmodels", STATSMODELS_OK), ("Prophet", PROPHET_OK),
                          ("LightGBM", LGBM_OK), ("XGBoost", XGBOOST_OK),
                          ("Random Forest", RFOREST_OK), ("Plotly", PLOTLY_OK)):
            st.write(f"{label}: {'✅' if ok else '❌'}")

    raw_data = st.text_area("Pega Fecha y Ventas desde Excel", st.session_state.raw,
                            height=270, key=f"data_{st.session_state.counter}")
    with st.expander("Opcional: CSV con varias referencias"):
        st.caption("Formato largo: Referencia;Fecha;Ventas (o Estadistico;Fecha;Ventas). Selecciona una referencia para analizar.")
        upload = st.file_uploader("Cargar CSV multirreferencia", type=["csv"])
        csv_frame, ref_col, selected_ref = None, None, None
        if upload is not None:
            try:
                csv_frame, ref_col = parse_reference_csv(upload)
                refs = sorted(csv_frame[ref_col].dropna().unique().tolist())
                selected_ref = st.selectbox("Referencia a analizar", refs)
            except Exception as exc:
                st.error(str(exc))
                csv_frame = None

    col1, col2 = st.columns(2)
    calculate = col1.button("🚀 Calcular", type="primary", use_container_width=True)
    col2.button("🧹 Limpiar", on_click=clear_history, use_container_width=True)
    if calculate:
        bar = None
        try:
            if upload is not None and csv_frame is None:
                raise ValueError("Corrige el CSV o elimínalo para usar los datos pegados.")
            if csv_frame is not None:
                frame = csv_frame.loc[csv_frame[ref_col].eq(selected_ref), ["Fecha", "Ventas"]]
                df = normalize_history(frame)
            else:
                df = parse_pasted_data(raw_data, start_month)

            bar = st.progress(0, text="Preparando evaluaciones...")

            def on_progress(done: int, total: int, name: str):
                bar.progress(min(1., done / total), text=f"Evaluando {name} · {done}/{total} configuraciones")

            lab = run_backtesting(df, max_windows=max_windows, progress_callback=on_progress)
            st.session_state.update({"ready": True, "raw": raw_data, "df": df, "lab": lab,
                                     "reference": selected_ref if csv_frame is not None else "Serie pegada"})
        except Exception as exc:
            st.session_state.ready = False
            st.error(f"No se pudo calcular: {exc}")
        finally:
            if bar is not None:
                bar.empty()

    if st.session_state.ready:
        try:
            show_results(st.session_state.df, st.session_state.lab, st.session_state.reference)
        except Exception as exc:
            st.error(f"No fue posible mostrar los resultados finales: {exc}")


if __name__ == "__main__":
    main()
