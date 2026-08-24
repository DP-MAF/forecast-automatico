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

st.set_page_config(page_title="Forecast automático de ventas", page_icon="📈", layout="wide")
st.title("📈 Forecast automático de ventas")
st.caption("Backtesting walk-forward, selección automática, alternativa manual y ensemble ponderado de los 4 mejores modelos.")

# ----------------------------- Datos -----------------------------
def clean_number(v):
    if pd.isna(v): return np.nan
    s = str(v).strip().replace(" ", "")
    if not s: return np.nan
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try: return float(s)
    except Exception: return np.nan

def parse_data(text, start_month):
    if not text or not text.strip(): raise ValueError("No se han introducido datos.")
    text = text.strip(); df = None
    for sep in ["\t", ";"]:
        try:
            t = pd.read_csv(io.StringIO(text), sep=sep, header=None, engine="python")
            if t.shape[1] >= 2: df = t; break
        except Exception: pass
    if df is None: df = pd.DataFrame(text.splitlines())
    df = df.dropna(how="all").reset_index(drop=True)
    if df.empty: raise ValueError("No se han podido interpretar los datos.")
    head = " ".join(str(x).lower() for x in df.iloc[0].values)
    if any(x in head for x in ["fecha", "mes", "venta", "cantidad", "sales"]):
        df = df.iloc[1:].reset_index(drop=True)
    if df.empty: raise ValueError("No quedan datos tras eliminar la cabecera.")
    if df.shape[1] >= 2:
        raw_dates = df.iloc[:, 0].astype(str).str.strip()
        dates = pd.to_datetime(raw_dates, errors="coerce")
        if dates.isna().all():
            dates = pd.to_datetime(raw_dates.str.replace("/", "-", regex=False) + "-01", errors="coerce")
        out = pd.DataFrame({"Fecha": dates, "Ventas": df.iloc[:, 1].map(clean_number)})
    else:
        vals = df.iloc[:, 0].map(clean_number)
        start = pd.to_datetime(start_month + "-01")
        out = pd.DataFrame({"Fecha": pd.date_range(start, periods=len(vals), freq="MS"), "Ventas": vals})
    out = out.dropna(subset=["Fecha", "Ventas"]).copy()
    out["Fecha"] = pd.to_datetime(out["Fecha"]).dt.to_period("M").dt.to_timestamp()
    out = out.sort_values("Fecha").drop_duplicates("Fecha", keep="last").reset_index(drop=True)
    if out.empty: raise ValueError("No se han obtenido filas válidas.")
    if (out["Ventas"] < 0).any(): raise ValueError("Hay ventas negativas. Revisa devoluciones o errores.")
    return out

def complete_months(df):
    idx = pd.date_range(df.Fecha.min(), df.Fecha.max(), freq="MS")
    return pd.DataFrame({"Fecha": idx}).merge(df, on="Fecha", how="left").fillna({"Ventas": 0})

# ----------------------------- Métricas -----------------------------
def mape(a, p):
    a=np.asarray(a,float); p=np.asarray(p,float); mask=a!=0
    return np.mean(np.abs((a[mask]-p[mask])/a[mask]))*100 if mask.any() else np.nan

def smape(a,p):
    a=np.asarray(a,float); p=np.asarray(p,float); d=(np.abs(a)+np.abs(p))/2; mask=d!=0
    return np.mean(np.abs(a[mask]-p[mask])/d[mask])*100 if mask.any() else np.nan

def wmape(a,p):
    a=np.asarray(a,float); p=np.asarray(p,float); d=np.abs(a).sum()
    return np.abs(a-p).sum()/d*100 if d else np.nan

def rounded(v):
    x=np.asarray(v,float); x=np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.rint(np.maximum(x,0)).astype(int)

# ----------------------------- Diagnóstico -----------------------------
def demand_diagnosis(df):
    y=df.Ventas.astype(float); mean=y.mean(); cv=(y.std(ddof=0)/mean) if mean else np.nan
    zero=(y==0).mean(); x=np.arange(len(y)); slope=np.polyfit(x,y,1)[0]/mean if len(y)>1 and mean else 0
    seasonal=False; strength=np.nan
    if len(y)>=24 and mean:
        tmp=df.assign(m=df.Fecha.dt.month); strength=tmp.groupby("m").Ventas.mean().std(ddof=0)/mean; seasonal=strength>=.20
    first=y.head(min(4,len(y))).mean(); last4=y.tail(min(4,len(y))).mean()
    last3=y.tail(min(3,len(y))).mean(); prev=y.iloc[-9:-3].mean() if len(y)>=9 else y.iloc[:-3].mean()
    if mean==0: kind="Sin demanda histórica"; exp="La media histórica es cero."; rec="Usar información comercial o de lanzamiento."
    elif zero>=.40: kind="Intermitente"; exp="Hay una proporción elevada de meses con venta cero."; rec="Revisar baja rotación y pedidos puntuales."
    elif len(y)>=8 and first>0 and last4>=1.8*first and slope>.03: kind="Posible lanzamiento"; exp="La demanda reciente supera claramente el inicio y crece."; rec="Validar pipeline, distribución y ramp-up."
    elif len(y)>=9 and prev>0 and last3<=.5*prev and slope<-.03: kind="Posible phase-out"; exp="La demanda reciente cae claramente frente al periodo anterior."; rec="Validar descatalogación o sustitución."
    elif seasonal: kind="Estacional"; exp="Se observan diferencias relevantes entre meses calendario."; rec="Priorizar modelos estacionales cuando sean elegibles."
    elif not pd.isna(cv) and cv<=.20: kind="Estable"; exp="La variabilidad relativa es baja."; rec="El forecast estadístico suele ser defendible."
    elif not pd.isna(cv) and cv>=.70: kind="Volátil"; exp="La variabilidad relativa es elevada."; rec="Revisar promociones, stockouts y pedidos extraordinarios."
    elif slope>=.03: kind="Tendencia creciente"; exp="La pendiente histórica mensual es positiva."; rec="Validar la continuidad del crecimiento."
    elif slope<=-.03: kind="Tendencia decreciente"; exp="La pendiente histórica mensual es negativa."; rec="Validar pérdida de distribución o declive estructural."
    else: kind="Sin patrón dominante"; exp="No se detecta un patrón dominante con estas reglas."; rec="Usar el resultado automático con revisión de negocio."
    return {"Tipo demanda":kind,"Explicación":exp,"Recomendación":rec,"CV":cv,"Meses cero %":zero*100,"Tendencia mensual %":slope*100,"Estacionalidad":"Sí" if seasonal else "No","Fuerza estacional":strength}

def reliability(df, error):
    y=df.Ventas; mean=y.mean(); cv=y.std(ddof=0)/mean if mean else np.nan; n=len(y)
    e=max(0,100-2*error) if not pd.isna(error) else 0
    s=0 if pd.isna(cv) else (100 if cv<=.2 else 0 if cv>=1 else 100*(1-(cv-.2)/.8))
    h=100 if n>=48 else 85+(n-36)*1.25 if n>=36 else 70+(n-24)*1.25 if n>=24 else 50+(n-16)*2.5 if n>=16 else 30
    score=round(.5*e+.3*s+.2*h,1)
    band="Muy alta" if score>=90 else "Alta" if score>=75 else "Media" if score>=60 else "Baja" if score>=40 else "Muy baja"
    return {"Índice":score,"Nivel":band,"Score error":round(e,1),"Score estabilidad":round(s,1),"Score histórico":round(h,1)}

# ----------------------------- Features ML -----------------------------
FEATURES=["lag_1","lag_2","lag_3","lag_6","lag_12","mean_3","mean_6","std_3","month_sin","month_cos","trend"]
def ml_table(values, dates):
    z=pd.DataFrame({"Fecha":pd.Series(pd.to_datetime(dates)).reset_index(drop=True),"y":pd.Series(np.asarray(values,float)).reset_index(drop=True)})
    for lag in [1,2,3,6,12]: z[f"lag_{lag}"]=z.y.shift(lag)
    z["mean_3"]=z.y.shift(1).rolling(3).mean(); z["mean_6"]=z.y.shift(1).rolling(6).mean(); z["std_3"]=z.y.shift(1).rolling(3).std(ddof=0)
    z["month_sin"]=np.sin(2*np.pi*z.Fecha.dt.month/12); z["month_cos"]=np.cos(2*np.pi*z.Fecha.dt.month/12); z["trend"]=np.arange(len(z))
    return z.dropna().reset_index(drop=True)

def recursive_ml(y, dates, horizon, kind):
    hist=list(np.asarray(y,float)); ds=list(pd.to_datetime(dates)); train=ml_table(hist,ds)
    if len(train)<4: raise ValueError(f"{kind} no dispone de suficientes filas entrenables.")
    if kind=="LightGBM":
        if not LGBM_OK: raise ValueError("LightGBM no está instalado.")
        model=LGBMRegressor(n_estimators=250,learning_rate=.04,num_leaves=7,max_depth=3,min_child_samples=1,random_state=42,verbosity=-1)
    elif kind=="XGBoost":
        if not XGBOOST_OK: raise ValueError("XGBoost no está instalado.")
        model=XGBRegressor(n_estimators=250,learning_rate=.04,max_depth=3,min_child_weight=1,subsample=.9,colsample_bytree=.9,objective="reg:squarederror",random_state=42,n_jobs=1,verbosity=0)
    else:
        if not RFOREST_OK: raise ValueError("Random Forest no está instalado.")
        model=RandomForestRegressor(n_estimators=300,max_depth=5,min_samples_leaf=1,max_features=.8,random_state=42,n_jobs=-1)
    model.fit(train[FEATURES],train.y)
    preds=[]; last=ds[-1]
    for step in range(1,horizon+1):
        nd=last+pd.DateOffset(months=step)
        def lag(k): return hist[-k] if len(hist)>=k else hist[0]
        row=pd.DataFrame([{ "lag_1":lag(1),"lag_2":lag(2),"lag_3":lag(3),"lag_6":lag(6),"lag_12":lag(12),
            "mean_3":np.mean(hist[-3:]),"mean_6":np.mean(hist[-6:]),"std_3":np.std(hist[-3:]),
            "month_sin":np.sin(2*np.pi*nd.month/12),"month_cos":np.cos(2*np.pi*nd.month/12),"trend":len(hist)}])
        p=max(0,float(model.predict(row[FEATURES])[0])); preds.append(p); hist.append(p)
    return np.asarray(preds)

# ----------------------------- Modelos -----------------------------
def available_models(n):
    out=["Naive","Media móvil 3 meses","Media móvil 6 meses","Suavizado exponencial","Holt","Holt amortiguado","ARIMA simple","Prophet","LightGBM","XGBoost","Random Forest"]
    if n>=24: out += ["Naive estacional 12 meses","Holt-Winters estacional"]
    return out

def run_model(name,y,dates,h):
    y=pd.Series(np.asarray(y,float)).reset_index(drop=True)
    if name=="Naive": return np.repeat(y.iloc[-1],h)
    if name.startswith("Media móvil"):
        w=3 if "3" in name else 6
        if len(y)<w: raise ValueError("Histórico insuficiente para la media móvil.")
        return np.repeat(y.tail(w).mean(),h)
    if name=="Naive estacional 12 meses":
        if len(y)<12: raise ValueError("Se requieren 12 meses.")
        base=y.tail(12).values; return np.array([base[i%12] for i in range(h)])
    if name=="Suavizado exponencial":
        if not STATSMODELS_OK: raise ValueError("statsmodels no está instalado.")
        return SimpleExpSmoothing(y,initialization_method="estimated").fit(optimized=True).forecast(h).values
    if name in ["Holt","Holt amortiguado"]:
        if not STATSMODELS_OK: raise ValueError("statsmodels no está instalado.")
        return Holt(y,damped_trend=name.endswith("amortiguado"),initialization_method="estimated").fit(optimized=True).forecast(h).values
    if name=="Holt-Winters estacional":
        if not STATSMODELS_OK or len(y)<24: raise ValueError("Holt-Winters requiere 24 meses y statsmodels.")
        return ExponentialSmoothing(y,trend="add",seasonal="add",seasonal_periods=12,initialization_method="estimated").fit(optimized=True).forecast(h).values
    if name=="ARIMA simple":
        if not STATSMODELS_OK: raise ValueError("statsmodels no está instalado.")
        best=None; aic=np.inf
        for order in [(0,1,0),(1,1,0),(0,1,1),(1,1,1),(2,1,1)]:
            try:
                m=SARIMAX(y,order=order,enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
                if m.aic<aic: best=m; aic=m.aic
            except Exception: pass
        if best is None: raise ValueError("ARIMA no ha podido ajustarse.")
        return best.forecast(h).values
    if name=="Prophet":
        if not PROPHET_OK: raise ValueError("Prophet no está instalado.")
        p=Prophet(yearly_seasonality=len(y)>=24,weekly_seasonality=False,daily_seasonality=False,seasonality_mode="additive",changepoint_prior_scale=.05,seasonality_prior_scale=5)
        p.fit(pd.DataFrame({"ds":pd.to_datetime(dates),"y":y.values}))
        return p.predict(p.make_future_dataframe(periods=h,freq="MS")).tail(h).yhat.values
    if name in ["LightGBM","XGBoost","Random Forest"]: return recursive_ml(y,dates,h,name)
    raise ValueError("Modelo no reconocido: "+name)

MIN_TRAIN={"Naive":1,"Media móvil 3 meses":3,"Media móvil 6 meses":6,"Suavizado exponencial":6,"Holt":8,"Holt amortiguado":8,"ARIMA simple":10,"Prophet":12,"LightGBM":16,"XGBoost":16,"Random Forest":16,"Naive estacional 12 meses":12,"Holt-Winters estacional":24}
RANK={n:i for i,n in enumerate(["Naive","Media móvil 3 meses","Media móvil 6 meses","Naive estacional 12 meses","Suavizado exponencial","Holt","Holt amortiguado","Holt-Winters estacional","ARIMA simple","Prophet","Random Forest","XGBoost","LightGBM"],1)}

def backtest(df):
    n=len(df); start=n-4
    if n<16: raise ValueError("Se requieren 16 meses para backtesting 12+4.")
    actual=df.iloc[start:].Ventas.values; rows=[]
    for name in available_models(n):
        if start<MIN_TRAIN.get(name,12): continue
        preds=[]; err=""
        try:
            for i in range(start,n): preds.append(rounded(run_model(name,df.iloc[:i].Ventas,df.iloc[:i].Fecha,1))[0])
            pred=np.asarray(preds,float); ma=mape(actual,pred); sm=smape(actual,pred); wm=wmape(actual,pred)
            metric="sMAPE" if np.any(actual==0) else "MAPE"; value=sm if metric=="sMAPE" else ma
            rows.append({"Modelo":name,"MAPE_%":ma,"sMAPE_%":sm,"WMAPE_%":wm,"Métrica selección":metric,"Valor métrica selección":value,
                         **{f"Backtest M+{i+1}":pred[i] for i in range(4)},"Estado":"OK","Error":""})
        except Exception as e:
            err=str(e); rows.append({"Modelo":name,"MAPE_%":np.nan,"sMAPE_%":np.nan,"WMAPE_%":np.nan,"Métrica selección":"","Valor métrica selección":np.nan,
                         **{f"Backtest M+{i+1}":np.nan for i in range(4)},"Estado":"Error","Error":err})
    res=pd.DataFrame(rows); valid=res[(res.Estado=="OK") & res["Valor métrica selección"].notna()].copy()
    if len(valid)<1: raise ValueError("Ningún modelo ha completado el backtesting.")
    valid["Complejidad"]=valid.Modelo.map(RANK).fillna(99); valid=valid.sort_values(["Valor métrica selección","Complejidad"])
    return valid.iloc[0].Modelo,float(valid.iloc[0]["Valor métrica selección"]),valid.iloc[0]["Métrica selección"],res,df.iloc[start:].copy(),valid

def future(df,name,h=18):
    p=rounded(run_model(name,df.Ventas,df.Fecha,h)); dates=pd.date_range(df.Fecha.max()+pd.DateOffset(months=1),periods=h,freq="MS")
    return pd.DataFrame({"Fecha":dates,"Mes":[f"M+{i}" for i in range(1,h+1)],"Forecast":p})

def ensemble_top4(df, valid, h=18):
    candidates=[]
    for name in valid.Modelo.tolist():
        try: candidates.append((name,future(df,name,h)))
        except Exception: pass
        if len(candidates)==4: break
    if len(candidates)<4: raise ValueError("No hay 4 modelos con backtesting y forecast final válidos para construir el ensemble.")
    weights=[.4,.3,.2,.1]; values=sum(w*f.Forecast.to_numpy(float) for w,(_,f) in zip(weights,candidates))
    out=candidates[0][1][["Fecha","Mes"]].copy(); out["Forecast"]=rounded(values)
    detail=pd.DataFrame({"Posición":[1,2,3,4],"Modelo":[x[0] for x in candidates],"Peso":weights,
                         "Error backtesting %":[float(valid.loc[valid.Modelo==x[0],"Valor métrica selección"].iloc[0]) for x in candidates]})
    return out,detail

def horizontal(f,kind,model):
    row={"Tipo forecast":kind,"Modelo":model}; row.update({f"M+{i+1}":int(v) for i,v in enumerate(f.Forecast)}); return row

def chart(df,a,m,e):
    if not PLOTLY_OK: return None
    fig=go.Figure(); fig.add_scatter(x=df.Fecha,y=df.Ventas,mode="lines+markers",name="Histórico")
    for f,n,d in [(a,"Automático",None),(m,"Manual","dash"),(e,"Ensemble Top 4","dot")]: fig.add_scatter(x=f.Fecha,y=f.Forecast,mode="lines+markers",name=n,line=dict(dash=d) if d else None)
    fig.update_layout(template="plotly_white",height=520,title="Histórico y previsiones",xaxis_title="Fecha",yaxis_title="Ventas",legend=dict(orientation="h")); return fig

# ----------------------------- Estado/UI -----------------------------
EXAMPLE="""Fecha\tVentas
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
for k,v in {"raw":EXAMPLE,"counter":0,"ready":False,"df":None,"res":None,"valid":None,"best":None,"metric":None,"metric_name":None,"test":None}.items():
    if k not in st.session_state: st.session_state[k]=v

def clear():
    st.session_state.raw=""; st.session_state.counter+=1; st.session_state.ready=False; st.session_state.df=None

with st.sidebar:
    st.header("⚙️ Configuración"); start_month=st.text_input("Mes inicial si pegas solo ventas","2025-01"); horizon=st.number_input("Horizonte",18,18,18)
    st.markdown("**Modelos:** Naive, medias móviles, suavizado, Holt, ARIMA, Prophet, LightGBM, XGBoost, Random Forest y estacionales con histórico suficiente.")
    st.markdown("---")
    for n,ok in [("statsmodels",STATSMODELS_OK),("Prophet",PROPHET_OK),("LightGBM",LGBM_OK),("XGBoost",XGBOOST_OK),("Random Forest",RFOREST_OK),("Plotly",PLOTLY_OK)]: st.write(f"{n}: {'✅' if ok else '❌'}")

st.subheader("1. Pega el histórico desde Excel")
raw=st.text_area("Fecha y Ventas, o una sola columna de ventas",value=st.session_state.raw,height=270,key=f"input_{st.session_state.counter}")
c1,c2=st.columns(2)
calc=c1.button("🚀 Calcular previsiones",type="primary",use_container_width=True)
c2.button("🧹 Limpiar histórico",on_click=clear,use_container_width=True)
if calc:
    try:
        df=complete_months(parse_data(raw,start_month)); st.session_state.raw=raw
        if len(df)<16: raise ValueError("Esta versión requiere al menos 16 meses para ejecutar el backtesting y el ensemble Top 4.")
        best,metric,metric_name,res,test,valid=backtest(df)
        st.session_state.update({"ready":True,"df":df,"best":best,"metric":metric,"metric_name":metric_name,"res":res,"test":test,"valid":valid})
    except Exception as e: st.session_state.ready=False; st.error(str(e))

if st.session_state.ready:
    df=st.session_state.df; best=st.session_state.best; res=st.session_state.res; valid=st.session_state.valid; test=st.session_state.test
    diag=demand_diagnosis(df); rel=reliability(df,st.session_state.metric)
    st.subheader("2. Diagnóstico")
    a,b,c,d=st.columns(4); a.metric("Meses",len(df)); b.metric("Tipo demanda",diag["Tipo demanda"]); c.metric("Confiabilidad",f"{rel['Índice']}/100"); d.metric("Nivel",rel["Nivel"])
    st.info(diag["Explicación"]); st.warning(diag["Recomendación"])
    with st.expander("Detalle de diagnóstico y confiabilidad"):
        st.dataframe(pd.DataFrame([diag]),use_container_width=True); st.dataframe(pd.DataFrame([rel]),use_container_width=True)
    st.subheader("3. Modelo automático y alternativa manual")
    x,y,z=st.columns(3); x.metric("Modelo ganador",best); y.metric(st.session_state.metric_name,f"{st.session_state.metric:.2f}%"); z.metric("Validación","Walk-forward 4 meses")
    successful=res[res.Estado=="OK"].Modelo.tolist(); manual=st.selectbox("Modelo alternativo",successful,index=successful.index(best))
    try:
        auto=future(df,best,18); man=future(df,manual,18); ens,ens_detail=ensemble_top4(df,valid,18)
        st.subheader("4. Forecast horizontal a 18 meses")
        out=pd.DataFrame([horizontal(auto,"Automático",best),horizontal(man,"Manual seleccionado",manual),horizontal(ens,"Ensemble ponderado Top 4","0,4 / 0,3 / 0,2 / 0,1")])
        st.dataframe(out,use_container_width=True)
        st.caption("El ensemble combina los cuatro mejores modelos válidos por error de backtesting: 40%, 30%, 20% y 10%. Cada resultado mensual se redondea a 0 decimales después de ponderar.")
        st.subheader("5. Composición del ensemble Top 4"); st.dataframe(ens_detail,use_container_width=True)
        st.subheader("6. Comparativa de modelos")
        show=res.copy();
        for col in ["MAPE_%","sMAPE_%","WMAPE_%","Valor métrica selección"]: show[col]=show[col].round(2)
        st.dataframe(show.sort_values(["Estado","Valor métrica selección"],ascending=[False,True]),use_container_width=True)
        st.subheader("7. Backtesting walk-forward del modelo ganador")
        br=res[res.Modelo==best].iloc[0]; bt=test.copy(); bt["Forecast ganador"]=[br[f"Backtest M+{i}"] for i in range(1,5)]; bt["Error absoluto"]=(bt.Ventas-bt["Forecast ganador"]).abs(); bt["Error %"]=np.where(bt.Ventas!=0,bt["Error absoluto"]/bt.Ventas*100,np.nan)
        st.dataframe(bt,use_container_width=True)
        st.subheader("8. Gráfico"); fig=chart(df,auto,man,ens); st.plotly_chart(fig,use_container_width=True) if fig else st.line_chart(pd.concat([df.set_index("Fecha").Ventas,auto.set_index("Fecha").Forecast],axis=1))
        st.subheader("9. Descargas")
        q1,q2,q3,q4=st.columns(4)
        q1.download_button("Forecast CSV",out.to_csv(index=False,sep=";").encode("utf-8-sig"),"forecast_18m.csv","text/csv",use_container_width=True)
        q2.download_button("Modelos CSV",res.to_csv(index=False,sep=";").encode("utf-8-sig"),"comparativa_modelos.csv","text/csv",use_container_width=True)
        q3.download_button("Backtest CSV",bt.to_csv(index=False,sep=";").encode("utf-8-sig"),"backtesting.csv","text/csv",use_container_width=True)
        q4.download_button("Ensemble CSV",ens_detail.to_csv(index=False,sep=";").encode("utf-8-sig"),"ensemble_top4.csv","text/csv",use_container_width=True)
    except Exception as e: st.error(f"No se han podido generar todas las previsiones: {e}")

st.markdown("---")
st.caption("La selección automática y el ensemble se basan en backtesting walk-forward. Revisar promociones, stockouts, lanzamientos, phase-out y eventos no contenidos en el histórico.")
