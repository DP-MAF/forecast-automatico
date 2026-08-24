import io, warnings
import numpy as np
import pandas as pd
import streamlit as st
warnings.filterwarnings("ignore")

try:
    import plotly.graph_objects as go
    PLOTLY=True
except Exception: PLOTLY=False
try:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SM=True
except Exception: SM=False
try:
    from prophet import Prophet
    PROPHET=True
except Exception: PROPHET=False
try:
    from lightgbm import LGBMRegressor
    LGBM=True
except Exception: LGBM=False
try:
    from xgboost import XGBRegressor
    XGB=True
except Exception: XGB=False
try:
    from sklearn.ensemble import RandomForestRegressor
    RF=True
except Exception: RF=False

st.set_page_config(page_title="Forecast automático",page_icon="📈",layout="wide")
st.title("📈 Forecast automático de ventas")
st.caption("Backtesting walk-forward, forecast automático, alternativa manual y ensemble ponderado Top 4.")

# DATOS
def number(v):
    if pd.isna(v): return np.nan
    s=str(v).strip().replace(" ","")
    if not s: return np.nan
    if "," in s and "." in s:
        s=s.replace(".","").replace(",",".") if s.rfind(",")>s.rfind(".") else s.replace(",","")
    elif "," in s: s=s.replace(",",".")
    try: return float(s)
    except: return np.nan

def parse(text,start_month):
    if not text or not text.strip(): raise ValueError("No se han introducido datos.")
    df=None
    for sep in ["\t",";"]:
        try:
            t=pd.read_csv(io.StringIO(text.strip()),sep=sep,header=None,engine="python")
            if t.shape[1]>=2: df=t; break
        except: pass
    if df is None: df=pd.DataFrame(text.strip().splitlines())
    df=df.dropna(how="all").reset_index(drop=True)
    if df.empty: raise ValueError("No se han interpretado datos.")
    head=" ".join(str(x).lower() for x in df.iloc[0].values)
    if any(x in head for x in ["fecha","mes","venta","cantidad","sales"]): df=df.iloc[1:].reset_index(drop=True)
    if df.shape[1]>=2:
        raw=df.iloc[:,0].astype(str).str.strip()
        dates=pd.to_datetime(raw,errors="coerce")
        if dates.isna().all(): dates=pd.to_datetime(raw.str.replace("/","-",regex=False)+"-01",errors="coerce")
        out=pd.DataFrame({"Fecha":dates,"Ventas":df.iloc[:,1].map(number)})
    else:
        vals=df.iloc[:,0].map(number); start=pd.to_datetime(start_month+"-01")
        out=pd.DataFrame({"Fecha":pd.date_range(start,periods=len(vals),freq="MS"),"Ventas":vals})
    out=out.dropna(subset=["Fecha","Ventas"]); out.Fecha=pd.to_datetime(out.Fecha).dt.to_period("M").dt.to_timestamp()
    out=out.sort_values("Fecha").drop_duplicates("Fecha",keep="last").reset_index(drop=True)
    if out.empty: raise ValueError("No hay filas válidas.")
    if (out.Ventas<0).any(): raise ValueError("Hay ventas negativas.")
    idx=pd.date_range(out.Fecha.min(),out.Fecha.max(),freq="MS")
    return pd.DataFrame({"Fecha":idx}).merge(out,on="Fecha",how="left").fillna({"Ventas":0})

# MÉTRICAS
def mape(a,p):
    a=np.asarray(a,float); p=np.asarray(p,float); q=a!=0
    return np.mean(np.abs((a[q]-p[q])/a[q]))*100 if q.any() else np.nan
def smape(a,p):
    a=np.asarray(a,float); p=np.asarray(p,float); d=(np.abs(a)+np.abs(p))/2; q=d!=0
    return np.mean(np.abs(a[q]-p[q])/d[q])*100 if q.any() else np.nan
def wmape(a,p):
    a=np.asarray(a,float); p=np.asarray(p,float); d=np.abs(a).sum()
    return np.abs(a-p).sum()/d*100 if d else np.nan
def round0(v): return np.rint(np.maximum(np.nan_to_num(np.asarray(v,float)),0)).astype(int)

# DIAGNÓSTICO
def diagnose(df):
    y=df.Ventas.astype(float); mean=y.mean(); cv=y.std(ddof=0)/mean if mean else np.nan
    zeros=(y==0).mean(); slope=np.polyfit(np.arange(len(y)),y,1)[0]/mean if len(y)>1 and mean else 0
    seas=False; strength=np.nan
    if len(y)>=24 and mean:
        strength=df.assign(m=df.Fecha.dt.month).groupby("m").Ventas.mean().std(ddof=0)/mean; seas=strength>=.20
    first=y.head(4).mean(); last4=y.tail(4).mean(); last3=y.tail(3).mean(); prev=y.iloc[-9:-3].mean() if len(y)>=9 else np.nan
    if mean==0: k,e,r="Sin demanda histórica","La media histórica es cero.","Usar información comercial."
    elif zeros>=.40: k,e,r="Intermitente","Hay muchos meses con venta cero.","Revisar baja rotación y pedidos puntuales."
    elif len(y)>=8 and first>0 and last4>=1.8*first and slope>.03: k,e,r="Posible lanzamiento","La demanda reciente crece frente al inicio.","Validar pipeline y ramp-up."
    elif len(y)>=9 and prev>0 and last3<=.5*prev and slope<-.03: k,e,r="Posible phase-out","La demanda reciente cae frente al periodo anterior.","Validar descatalogación o sustitución."
    elif seas: k,e,r="Estacional","Hay diferencias relevantes entre meses calendario.","Revisar modelos estacionales."
    elif not pd.isna(cv) and cv<=.20: k,e,r="Estable","La variabilidad relativa es baja.","El forecast estadístico suele ser defendible."
    elif not pd.isna(cv) and cv>=.70: k,e,r="Volátil","La variabilidad relativa es elevada.","Revisar promociones, roturas y pedidos extraordinarios."
    elif slope>=.03: k,e,r="Tendencia creciente","La pendiente mensual es positiva.","Validar continuidad del crecimiento."
    elif slope<=-.03: k,e,r="Tendencia decreciente","La pendiente mensual es negativa.","Validar pérdida de distribución o declive."
    else: k,e,r="Sin patrón dominante","No se detecta un patrón dominante.","Usar el automático con revisión de negocio."
    return {"Tipo demanda":k,"Explicación":e,"Recomendación":r,"CV":cv,"Meses cero %":zeros*100,"Tendencia mensual %":slope*100,"Estacionalidad":"Sí" if seas else "No","Fuerza estacional":strength}

def confidence(df,error):
    y=df.Ventas; mean=y.mean(); cv=y.std(ddof=0)/mean if mean else np.nan; n=len(y)
    es=max(0,100-2*error) if not pd.isna(error) else 0
    ss=0 if pd.isna(cv) else 100 if cv<=.2 else 0 if cv>=1 else 100*(1-(cv-.2)/.8)
    hs=100 if n>=48 else 85+(n-36)*1.25 if n>=36 else 70+(n-24)*1.25 if n>=24 else 50+(n-16)*2.5
    val=round(.5*es+.3*ss+.2*hs,1)
    band="Muy alta" if val>=90 else "Alta" if val>=75 else "Media" if val>=60 else "Baja" if val>=40 else "Muy baja"
    return {"Índice":val,"Nivel":band,"Score error":round(es,1),"Score estabilidad":round(ss,1),"Score histórico":round(hs,1)}

# ACTIVACIÓN POR HISTÓRICO TOTAL
def models(n):
    out=["Naive","Media móvil 3 meses","Media móvil 6 meses","Suavizado exponencial","Holt","Holt amortiguado","ARIMA simple","Prophet"]
    if n>=18: out += ["LightGBM","XGBoost","Random Forest"]
    if n>=24: out += ["Naive estacional 12 meses","Holt-Winters estacional"]
    return out
def blocked(n):
    out=[]
    if n<18: out += [("LightGBM",18),("XGBoost",18),("Random Forest",18)]
    if n<24: out += [("Naive estacional 12 meses",24),("Holt-Winters estacional",24)]
    return out

# FEATURES ML ADAPTATIVAS
BASE=["lag_1","lag_2","lag_3","lag_6","mean_3","mean_6","std_3","month_sin","month_cos","trend"]
def ml_table(y,dates):
    z=pd.DataFrame({"Fecha":pd.Series(pd.to_datetime(dates)).reset_index(drop=True),"y":pd.Series(np.asarray(y,float)).reset_index(drop=True)})
    for lag in [1,2,3,6]: z[f"lag_{lag}"]=z.y.shift(lag)
    f=BASE.copy()
    if len(z)>=24: z["lag_12"]=z.y.shift(12); f.insert(4,"lag_12")
    z["mean_3"]=z.y.shift(1).rolling(3).mean(); z["mean_6"]=z.y.shift(1).rolling(6).mean(); z["std_3"]=z.y.shift(1).rolling(3).std(ddof=0)
    z["month_sin"]=np.sin(2*np.pi*z.Fecha.dt.month/12); z["month_cos"]=np.cos(2*np.pi*z.Fecha.dt.month/12); z["trend"]=np.arange(len(z))
    return z.dropna().reset_index(drop=True),f

def ml_forecast(y,dates,h,kind):
    hist=list(np.asarray(y,float)); ds=list(pd.to_datetime(dates)); train,f=ml_table(hist,ds)
    if len(train)<6: raise ValueError(f"{kind}: filas entrenables insuficientes.")
    if kind=="LightGBM":
        if not LGBM: raise ValueError("LightGBM no está instalado.")
        model=LGBMRegressor(n_estimators=250,learning_rate=.04,num_leaves=7,max_depth=3,min_child_samples=1,random_state=42,verbosity=-1)
    elif kind=="XGBoost":
        if not XGB: raise ValueError("XGBoost no está instalado.")
        model=XGBRegressor(n_estimators=250,learning_rate=.04,max_depth=3,min_child_weight=1,subsample=.9,colsample_bytree=.9,objective="reg:squarederror",random_state=42,n_jobs=1,verbosity=0)
    else:
        if not RF: raise ValueError("Random Forest no está instalado.")
        model=RandomForestRegressor(n_estimators=300,max_depth=5,min_samples_leaf=1,max_features=.8,random_state=42,n_jobs=-1)
    model.fit(train[f],train.y); pred=[]; last=ds[-1]
    for step in range(1,h+1):
        nd=last+pd.DateOffset(months=step); lag=lambda k: hist[-k] if len(hist)>=k else hist[0]
        d={"lag_1":lag(1),"lag_2":lag(2),"lag_3":lag(3),"lag_6":lag(6),"mean_3":np.mean(hist[-3:]),"mean_6":np.mean(hist[-6:]),"std_3":np.std(hist[-3:]),"month_sin":np.sin(2*np.pi*nd.month/12),"month_cos":np.cos(2*np.pi*nd.month/12),"trend":len(hist)}
        if "lag_12" in f: d["lag_12"]=lag(12)
        p=max(0,float(model.predict(pd.DataFrame([d])[f])[0])); pred.append(p); hist.append(p)
    return np.asarray(pred)

# EJECUCIÓN MODELOS
def run(name,y,dates,h):
    y=pd.Series(np.asarray(y,float)).reset_index(drop=True)
    if name=="Naive": return np.repeat(y.iloc[-1],h)
    if name.startswith("Media móvil"):
        w=3 if "3" in name else 6; return np.repeat(y.tail(w).mean(),h)
    if name=="Naive estacional 12 meses":
        b=y.tail(12).values; return np.array([b[i%12] for i in range(h)])
    if name=="Suavizado exponencial":
        if not SM: raise ValueError("statsmodels no está instalado.")
        return SimpleExpSmoothing(y,initialization_method="estimated").fit(optimized=True).forecast(h).values
    if name in ["Holt","Holt amortiguado"]:
        if not SM: raise ValueError("statsmodels no está instalado.")
        return Holt(y,damped_trend=name.endswith("amortiguado"),initialization_method="estimated").fit(optimized=True).forecast(h).values
    if name=="Holt-Winters estacional":
        if not SM or len(y)<24: raise ValueError("Holt-Winters requiere 24 meses.")
        return ExponentialSmoothing(y,trend="add",seasonal="add",seasonal_periods=12,initialization_method="estimated").fit(optimized=True).forecast(h).values
    if name=="ARIMA simple":
        if not SM: raise ValueError("statsmodels no está instalado.")
        best=None; aic=np.inf
        for order in [(0,1,0),(1,1,0),(0,1,1),(1,1,1),(2,1,1)]:
            try:
                z=SARIMAX(y,order=order,enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
                if z.aic<aic: best,aic=z,z.aic
            except: pass
        if best is None: raise ValueError("ARIMA no ajustado.")
        return best.forecast(h).values
    if name=="Prophet":
        if not PROPHET: raise ValueError("Prophet no está instalado.")
        z=Prophet(yearly_seasonality=len(y)>=24,weekly_seasonality=False,daily_seasonality=False,changepoint_prior_scale=.05,seasonality_prior_scale=5)
        z.fit(pd.DataFrame({"ds":pd.to_datetime(dates),"y":y.values}))
        return z.predict(z.make_future_dataframe(periods=h,freq="MS")).tail(h).yhat.values
    if name in ["LightGBM","XGBoost","Random Forest"]: return ml_forecast(y,dates,h,name)
    raise ValueError("Modelo no reconocido.")

MIN={"Naive":1,"Media móvil 3 meses":3,"Media móvil 6 meses":6,"Suavizado exponencial":6,"Holt":8,"Holt amortiguado":8,"ARIMA simple":10,"Prophet":12,"LightGBM":12,"XGBoost":12,"Random Forest":12,"Naive estacional 12 meses":12,"Holt-Winters estacional":24}
ORDER={x:i for i,x in enumerate(["Naive","Media móvil 3 meses","Media móvil 6 meses","Naive estacional 12 meses","Suavizado exponencial","Holt","Holt amortiguado","Holt-Winters estacional","ARIMA simple","Prophet","Random Forest","XGBoost","LightGBM"],1)}

def backtest(df):
    n=len(df); cut=n-4; actual=df.iloc[cut:].Ventas.values; rows=[]
    for name in models(n):
        if cut<MIN[name]: continue
        try:
            pred=np.array([round0(run(name,df.iloc[:i].Ventas,df.iloc[:i].Fecha,1))[0] for i in range(cut,n)],float)
            ma,sm,wm=mape(actual,pred),smape(actual,pred),wmape(actual,pred); met="sMAPE" if np.any(actual==0) else "MAPE"; val=sm if met=="sMAPE" else ma
            rows.append({"Modelo":name,"MAPE_%":ma,"sMAPE_%":sm,"WMAPE_%":wm,"Métrica selección":met,"Valor métrica selección":val,**{f"Backtest M+{j+1}":pred[j] for j in range(4)},"Estado":"OK","Error":""})
        except Exception as e:
            rows.append({"Modelo":name,"MAPE_%":np.nan,"sMAPE_%":np.nan,"WMAPE_%":np.nan,"Métrica selección":"","Valor métrica selección":np.nan,**{f"Backtest M+{j+1}":np.nan for j in range(4)},"Estado":"Error","Error":str(e)})
    res=pd.DataFrame(rows); valid=res[(res.Estado=="OK")&res["Valor métrica selección"].notna()].copy()
    if valid.empty: raise ValueError("Ningún modelo completó el backtesting.")
    valid["Orden"]=valid.Modelo.map(ORDER); valid=valid.sort_values(["Valor métrica selección","Orden"])
    return valid.iloc[0].Modelo,float(valid.iloc[0]["Valor métrica selección"]),valid.iloc[0]["Métrica selección"],res,valid,df.iloc[cut:].copy()

def future(df,name):
    p=round0(run(name,df.Ventas,df.Fecha,18)); return pd.DataFrame({"Fecha":pd.date_range(df.Fecha.max()+pd.DateOffset(months=1),periods=18,freq="MS"),"Forecast":p})
def ensemble(df,valid):
    ok=[]
    for name in valid.Modelo:
        try: ok.append((name,future(df,name)))
        except: pass
        if len(ok)==4: break
    if len(ok)<4: raise ValueError("Se necesitan cuatro modelos válidos para el ensemble.")
    w=[.4,.3,.2,.1]; out=ok[0][1][["Fecha"]].copy(); out["Forecast"]=round0(sum(a*f.Forecast.to_numpy(float) for a,(_,f) in zip(w,ok)))
    detail=pd.DataFrame({"Posición":[1,2,3,4],"Modelo":[x[0] for x in ok],"Peso":w,"Error %":[float(valid.loc[valid.Modelo==x[0],"Valor métrica selección"].iloc[0]) for x in ok]})
    return out,detail
def horizontal(f,tipo,model): return {"Tipo forecast":tipo,"Modelo":model,**{f"M+{i+1}":int(x) for i,x in enumerate(f.Forecast)}}

# UI Y ESTADO
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
for k,v in {"raw":EXAMPLE,"counter":0,"ready":False}.items():
    if k not in st.session_state: st.session_state[k]=v
def clear(): st.session_state.raw=""; st.session_state.counter+=1; st.session_state.ready=False

with st.sidebar:
    st.header("⚙️ Configuración"); start=st.text_input("Mes inicial si pegas solo ventas","2025-01")
    st.markdown("**Activación por histórico total:**\n- 16-17 meses: clásicos + Prophet\n- 18-23 meses: añade LightGBM, XGBoost y Random Forest\n- 24+ meses: añade estacionales")
    st.markdown("---")
    for name,ok in [("statsmodels",SM),("Prophet",PROPHET),("LightGBM",LGBM),("XGBoost",XGB),("Random Forest",RF),("Plotly",PLOTLY)]: st.write(f"{name}: {'✅' if ok else '❌'}")

raw=st.text_area("1. Pega Fecha y Ventas desde Excel",value=st.session_state.raw,height=270,key=f"input_{st.session_state.counter}")
a,b=st.columns(2); calc=a.button("🚀 Calcular previsiones",type="primary",use_container_width=True); b.button("🧹 Limpiar",on_click=clear,use_container_width=True)
if calc:
    try:
        df=parse(raw,start)
        if len(df)<16: raise ValueError("Se requieren al menos 16 meses.")
        best,err,metric,res,valid,test=backtest(df)
        st.session_state.update({"ready":True,"raw":raw,"df":df,"best":best,"err":err,"metric":metric,"res":res,"valid":valid,"test":test})
    except Exception as e: st.session_state.ready=False; st.error(str(e))

if st.session_state.ready:
    df=st.session_state.df; best=st.session_state.best; res=st.session_state.res; valid=st.session_state.valid; test=st.session_state.test
    diag=diagnose(df); conf=confidence(df,st.session_state.err)
    st.subheader("2. Diagnóstico")
    c1,c2,c3,c4=st.columns(4); c1.metric("Meses",len(df)); c2.metric("Tipo demanda",diag["Tipo demanda"]); c3.metric("Confiabilidad",f"{conf['Índice']}/100"); c4.metric("Nivel",conf["Nivel"])
    st.info(diag["Explicación"]); st.warning(diag["Recomendación"])
    with st.expander("Detalle"): st.dataframe(pd.DataFrame([diag]),use_container_width=True); st.dataframe(pd.DataFrame([conf]),use_container_width=True)
    no=blocked(len(df))
    if no: st.info("Modelos no habilitados: "+"; ".join(f"{x} requiere {n} meses" for x,n in no)+". No aparecen en comparativa ni selector hasta alcanzar el mínimo.")
    st.subheader("3. Selección")
    q1,q2,q3=st.columns(3); q1.metric("Ganador",best); q2.metric(st.session_state.metric,f"{st.session_state.err:.2f}%"); q3.metric("Validación","Walk-forward 4 meses")
    good=res[res.Estado=="OK"].Modelo.tolist(); manual=st.selectbox("Modelo alternativo",good,index=good.index(best))
    try:
        auto=future(df,best); man=future(df,manual); ens,detail=ensemble(df,valid)
        out=pd.DataFrame([horizontal(auto,"Automático",best),horizontal(man,"Manual",manual),horizontal(ens,"Ensemble Top 4","0,4 / 0,3 / 0,2 / 0,1")])
        st.subheader("4. Forecast horizontal M+1 a M+18"); st.dataframe(out,use_container_width=True)
        st.caption("El ensemble pondera los cuatro mejores modelos válidos 40%, 30%, 20% y 10%. El redondeo a 0 decimales se realiza después de ponderar.")
        st.subheader("5. Composición Ensemble Top 4"); st.dataframe(detail,use_container_width=True)
        st.subheader("6. Comparativa de modelos"); show=res.copy()
        for col in ["MAPE_%","sMAPE_%","WMAPE_%","Valor métrica selección"]: show[col]=show[col].round(2)
        st.dataframe(show.sort_values(["Estado","Valor métrica selección"],ascending=[False,True]),use_container_width=True)
        st.subheader("7. Backtesting ganador"); row=res[res.Modelo==best].iloc[0]; bt=test.copy(); bt["Forecast"]=[row[f"Backtest M+{i}"] for i in range(1,5)]; bt["Error absoluto"]=(bt.Ventas-bt.Forecast).abs(); bt["Error %"]=np.where(bt.Ventas!=0,bt["Error absoluto"]/bt.Ventas*100,np.nan); st.dataframe(bt,use_container_width=True)
        if PLOTLY:
            fig=go.Figure(); fig.add_scatter(x=df.Fecha,y=df.Ventas,mode="lines+markers",name="Histórico")
            for f,n,d in [(auto,"Automático",None),(man,"Manual","dash"),(ens,"Ensemble","dot")]: fig.add_scatter(x=f.Fecha,y=f.Forecast,mode="lines+markers",name=n,line=dict(dash=d) if d else None)
            fig.update_layout(template="plotly_white",height=520); st.plotly_chart(fig,use_container_width=True)
        d1,d2,d3,d4=st.columns(4)
        d1.download_button("Forecast CSV",out.to_csv(index=False,sep=";").encode("utf-8-sig"),"forecast_18m.csv","text/csv",use_container_width=True)
        d2.download_button("Modelos CSV",res.to_csv(index=False,sep=";").encode("utf-8-sig"),"modelos.csv","text/csv",use_container_width=True)
        d3.download_button("Backtest CSV",bt.to_csv(index=False,sep=";").encode("utf-8-sig"),"backtest.csv","text/csv",use_container_width=True)
        d4.download_button("Ensemble CSV",detail.to_csv(index=False,sep=";").encode("utf-8-sig"),"ensemble.csv","text/csv",use_container_width=True)
    except Exception as e: st.error(f"Error generando previsiones: {e}")
