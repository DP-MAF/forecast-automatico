import io, warnings
import numpy as np
import pandas as pd
import streamlit as st
warnings.filterwarnings("ignore")

try:
 import plotly.graph_objects as go; PLOTLY=True
except: PLOTLY=False
try:
 from statsmodels.tsa.holtwinters import SimpleExpSmoothing,Holt,ExponentialSmoothing
 from statsmodels.tsa.statespace.sarimax import SARIMAX
 SM=True
except: SM=False
try:
 from prophet import Prophet
 PROPHET=True
except: PROPHET=False
try:
 from lightgbm import LGBMRegressor
 LGBM=True
except: LGBM=False
try:
 from xgboost import XGBRegressor
 XGB=True
except: XGB=False
try:
 from sklearn.ensemble import RandomForestRegressor
 RF=True
except: RF=False

st.set_page_config(page_title="Forecast V2",page_icon="📈",layout="wide")
st.title("📈 Forecast automático de ventas V2")
st.caption("Backtesting walk-forward, demanda intermitente, selección automática y ensembles Top 4.")

# ---------- entrada ----------
def num(v):
 s=str(v).strip().replace(" ","")
 if not s or s.lower()=="nan": return np.nan
 if "," in s and "." in s: s=s.replace(".","").replace(",",".") if s.rfind(",")>s.rfind(".") else s.replace(",","")
 elif "," in s: s=s.replace(",",".")
 try:return float(s)
 except:return np.nan

def read_data(text,start):
 if not text.strip(): raise ValueError("Pega al menos 16 meses de datos.")
 df=None
 for sep in ["\t",";"]:
  try:
   z=pd.read_csv(io.StringIO(text),sep=sep,header=None,engine="python")
   if z.shape[1]>=2: df=z; break
  except: pass
 if df is None: df=pd.DataFrame(text.splitlines())
 df=df.dropna(how="all").reset_index(drop=True)
 if any(w in " ".join(map(lambda x:str(x).lower(),df.iloc[0])) for w in ["fecha","mes","venta","sales"]): df=df.iloc[1:]
 if df.shape[1]>=2:
  d=pd.to_datetime(df.iloc[:,0].astype(str).str.strip(),errors="coerce")
  if d.isna().all(): d=pd.to_datetime(df.iloc[:,0].astype(str).str.strip()+"-01",errors="coerce")
  out=pd.DataFrame({"Fecha":d,"Ventas":df.iloc[:,1].map(num)})
 else:
  v=df.iloc[:,0].map(num); out=pd.DataFrame({"Fecha":pd.date_range(pd.to_datetime(start+"-01"),periods=len(v),freq="MS"),"Ventas":v})
 out=out.dropna().copy(); out.Fecha=pd.to_datetime(out.Fecha).dt.to_period("M").dt.to_timestamp(); out=out.sort_values("Fecha").drop_duplicates("Fecha",keep="last")
 if out.empty or (out.Ventas<0).any(): raise ValueError("Datos vacíos, no válidos o con ventas negativas.")
 idx=pd.date_range(out.Fecha.min(),out.Fecha.max(),freq="MS")
 return pd.DataFrame({"Fecha":idx}).merge(out,on="Fecha",how="left").fillna({"Ventas":0}).reset_index(drop=True)

# ---------- métricas ----------
def mape(a,p):
 a=np.asarray(a,float);p=np.asarray(p,float);q=a!=0
 return np.mean(np.abs((a[q]-p[q])/a[q]))*100 if q.any() else np.nan
def smape(a,p):
 a=np.asarray(a,float);p=np.asarray(p,float);d=(np.abs(a)+np.abs(p))/2;q=d!=0
 return np.mean(np.abs(a[q]-p[q])/d[q])*100 if q.any() else np.nan
def wmape(a,p):
 a=np.asarray(a,float);p=np.asarray(p,float);d=np.abs(a).sum()
 return np.abs(a-p).sum()/d*100 if d else np.nan
def r0(x): return np.rint(np.maximum(np.nan_to_num(np.asarray(x,float)),0)).astype(int)

# ---------- Croston / SBA / TSB ----------
def croston_rate(y,alpha=.1,variant="Croston",beta=.05):
 y=np.asarray(y,float); nz=np.flatnonzero(y>0)
 if len(nz)==0:return 0.
 if variant=="TSB":
  z=y[nz[0]]; p=1.0
  for v in y[nz[0]+1:]:
   occ=1.0 if v>0 else 0.0; p=beta*occ+(1-beta)*p
   if v>0:z=alpha*v+(1-alpha)*z
  return p*z
 z=y[nz[0]]; interval=1.; gap=0
 for v in y[nz[0]+1:]:
  gap+=1
  if v>0:
   z=alpha*v+(1-alpha)*z; interval=alpha*gap+(1-alpha)*interval; gap=0
 rate=z/max(interval,1e-9)
 return rate*(1-alpha/2) if variant=="SBA" else rate

def tune_intermittent(y,variant):
 best=(1e99,.1,.05)
 for a in [.05,.1,.15,.2,.3]:
  for b in ([.02,.05,.1,.2] if variant=="TSB" else [.05]):
   errs=[]
   for i in range(max(6,len(y)-4),len(y)):
    pred=croston_rate(y[:i],a,variant,b); errs.append(abs(y[i]-pred))
   score=np.mean(errs) if errs else 1e99
   if score<best[0]:best=(score,a,b)
 return best[1],best[2]

def intermittent_forecast(y,h,variant):
 a,b=tune_intermittent(np.asarray(y,float),variant); f=croston_rate(y,a,variant,b)
 return np.repeat(f,h)

# ---------- ML ----------
BASE=["l1","l2","l3","l6","ma3","ma6","sd3","sin","cos","trend"]
def ml_data(y,dates):
 z=pd.DataFrame({"d":pd.to_datetime(dates).reset_index(drop=True),"y":pd.Series(np.asarray(y,float))})
 for j in [1,2,3,6]:z[f"l{j}"]=z.y.shift(j)
 feats=BASE.copy()
 if len(z)>=24:z["l12"]=z.y.shift(12);feats.insert(4,"l12")
 z["ma3"]=z.y.shift(1).rolling(3).mean();z["ma6"]=z.y.shift(1).rolling(6).mean();z["sd3"]=z.y.shift(1).rolling(3).std(ddof=0)
 z["sin"]=np.sin(2*np.pi*z.d.dt.month/12);z["cos"]=np.cos(2*np.pi*z.d.dt.month/12);z["trend"]=np.arange(len(z))
 return z.dropna(),feats

def ml_forecast(y,dates,h,name):
 hist=list(np.asarray(y,float));ds=list(pd.to_datetime(dates));train,f=ml_data(pd.Series(hist),pd.Series(ds))
 if len(train)<6:raise ValueError("Pocas filas para "+name)
 if name=="LightGBM":
  if not LGBM:raise ValueError("LightGBM no instalado")
  model=LGBMRegressor(n_estimators=200,learning_rate=.04,num_leaves=7,max_depth=3,min_child_samples=1,verbosity=-1,random_state=42)
 elif name=="XGBoost":
  if not XGB:raise ValueError("XGBoost no instalado")
  model=XGBRegressor(n_estimators=200,learning_rate=.04,max_depth=3,objective="reg:squarederror",n_jobs=1,verbosity=0,random_state=42)
 else:
  if not RF:raise ValueError("Random Forest no instalado")
  model=RandomForestRegressor(n_estimators=250,max_depth=5,min_samples_leaf=1,n_jobs=-1,random_state=42)
 model.fit(train[f],train.y);out=[];last=ds[-1]
 for step in range(1,h+1):
  d=last+pd.DateOffset(months=step);lag=lambda k:hist[-k] if len(hist)>=k else hist[0]
  q={"l1":lag(1),"l2":lag(2),"l3":lag(3),"l6":lag(6),"ma3":np.mean(hist[-3:]),"ma6":np.mean(hist[-6:]),"sd3":np.std(hist[-3:]),"sin":np.sin(2*np.pi*d.month/12),"cos":np.cos(2*np.pi*d.month/12),"trend":len(hist)}
  if "l12" in f:q["l12"]=lag(12)
  p=max(0,float(model.predict(pd.DataFrame([q])[f])[0]));out.append(p);hist.append(p)
 return np.asarray(out)

# ---------- catálogo y modelos ----------
def model_list(n):
 x=["Naive","Media móvil 3","Media móvil 6","Suavizado exponencial","Holt","Holt amortiguado","ARIMA simple","Prophet","Croston","SBA","TSB"]
 if n>=18:x += ["LightGBM","XGBoost","Random Forest"]
 if n>=24:x += ["Naive estacional 12","Holt-Winters estacional"]
 return x

def run(name,y,dates,h):
 y=pd.Series(np.asarray(y,float)).reset_index(drop=True)
 if name=="Naive":return np.repeat(y.iloc[-1],h)
 if name.startswith("Media móvil"):
  w=3 if name.endswith("3") else 6;return np.repeat(y.tail(w).mean(),h)
 if name=="Naive estacional 12":
  b=y.tail(12).values;return np.array([b[i%12] for i in range(h)])
 if name in ["Croston","SBA","TSB"]:return intermittent_forecast(y,h,name)
 if name=="Suavizado exponencial":
  if not SM:raise ValueError("statsmodels no instalado")
  return SimpleExpSmoothing(y,initialization_method="estimated").fit().forecast(h).values
 if name in ["Holt","Holt amortiguado"]:
  if not SM:raise ValueError("statsmodels no instalado")
  return Holt(y,damped_trend=name.endswith("amortiguado"),initialization_method="estimated").fit().forecast(h).values
 if name=="Holt-Winters estacional":
  if not SM:raise ValueError("statsmodels no instalado")
  return ExponentialSmoothing(y,trend="add",seasonal="add",seasonal_periods=12,initialization_method="estimated").fit().forecast(h).values
 if name=="ARIMA simple":
  if not SM:raise ValueError("statsmodels no instalado")
  best=None;aic=np.inf
  for order in [(0,1,0),(1,1,0),(0,1,1),(1,1,1),(2,1,1)]:
   try:
    z=SARIMAX(y,order=order,enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
    if z.aic<aic:best,aic=z,z.aic
   except:pass
  if best is None:raise ValueError("ARIMA no ajustado")
  return best.forecast(h).values
 if name=="Prophet":
  if not PROPHET:raise ValueError("Prophet no instalado")
  z=Prophet(yearly_seasonality=len(y)>=24,weekly_seasonality=False,daily_seasonality=False)
  z.fit(pd.DataFrame({"ds":pd.to_datetime(dates),"y":y}));return z.predict(z.make_future_dataframe(periods=h,freq="MS")).tail(h).yhat.values
 if name in ["LightGBM","XGBoost","Random Forest"]:return ml_forecast(y,dates,h,name)
 raise ValueError("Modelo desconocido")

# ---------- validación y ensembles ----------
def backtest(df):
 cut=len(df)-4;actual=df.iloc[cut:].Ventas.values;rows=[]
 for name in model_list(len(df)):
  try:
   pred=np.array([r0(run(name,df.iloc[:i].Ventas,df.iloc[:i].Fecha,1))[0] for i in range(cut,len(df))],float)
   ma,sm,wm=mape(actual,pred),smape(actual,pred),wmape(actual,pred);metric="sMAPE" if np.any(actual==0) else "MAPE";value=sm if metric=="sMAPE" else ma
   rows.append({"Modelo":name,"MAPE_%":ma,"sMAPE_%":sm,"WMAPE_%":wm,"Métrica":metric,"Error selección_%":value,**{f"BT{i+1}":pred[i] for i in range(4)},"Estado":"OK","Detalle":""})
  except Exception as e:rows.append({"Modelo":name,"MAPE_%":np.nan,"sMAPE_%":np.nan,"WMAPE_%":np.nan,"Métrica":"","Error selección_%":np.nan,**{f"BT{i+1}":np.nan for i in range(4)},"Estado":"Error","Detalle":str(e)})
 res=pd.DataFrame(rows);valid=res[(res.Estado=="OK")&res["Error selección_%"].notna()].sort_values("Error selección_%")
 if len(valid)<4:raise ValueError("Se necesitan al menos cuatro modelos válidos.")
 return res,valid,df.iloc[cut:].copy()

def future(df,name):
 return pd.DataFrame({"Fecha":pd.date_range(df.Fecha.max()+pd.DateOffset(months=1),periods=18,freq="MS"),"Forecast":r0(run(name,df.Ventas,df.Fecha,18))})
def blends(df,valid):
 items=[]
 for name in valid.Modelo:
  try:items.append((name,future(df,name),float(valid.loc[valid.Modelo==name,"Error selección_%"].iloc[0])))
  except:pass
  if len(items)==4:break
 if len(items)<4:raise ValueError("No hay cuatro forecasts finales válidos.")
 fixed=np.array([.4,.3,.2,.1]);errors=np.array([max(x[2],.01) for x in items]);dynamic=(1/errors)/(1/errors).sum()
 mat=np.vstack([x[1].Forecast.values for x in items])
 base=items[0][1][["Fecha"]].copy();f1=base.copy();f2=base.copy();f1["Forecast"]=r0(fixed@mat);f2["Forecast"]=r0(dynamic@mat)
 detail=pd.DataFrame({"Posición":range(1,5),"Modelo":[x[0] for x in items],"Error_%":errors,"Peso fijo":fixed,"Peso dinámico":dynamic})
 return f1,f2,detail
def horiz(f,t,m):return {"Tipo":t,"Modelo":m,**{f"M+{i+1}":int(v) for i,v in enumerate(f.Forecast)}}

# ---------- interfaz ----------
EX="""Fecha\tVentas
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
for k,v in {"raw":EX,"counter":0,"ready":False}.items():
 if k not in st.session_state:st.session_state[k]=v
def clear():st.session_state.raw="";st.session_state.counter+=1;st.session_state.ready=False
with st.sidebar:
 st.header("Configuración");start=st.text_input("Mes inicial si pegas solo ventas","2025-01")
 st.markdown("**V2 incluye:** Croston, SBA, TSB y ensemble dinámico inverso al error. ML se activa desde 18 meses y estacionales desde 24.")
 for n,ok in [("Prophet",PROPHET),("LightGBM",LGBM),("XGBoost",XGB),("Random Forest",RF)]:st.write(f"{n}: {'✅' if ok else '❌'}")
raw=st.text_area("Pega Fecha y Ventas desde Excel",st.session_state.raw,height=270,key=f"x{st.session_state.counter}")
a,b=st.columns(2);go=a.button("🚀 Calcular",type="primary",use_container_width=True);b.button("🧹 Limpiar",on_click=clear,use_container_width=True)
if go:
 try:
  df=read_data(raw,start)
  if len(df)<16:raise ValueError("Se requieren al menos 16 meses.")
  res,valid,test=backtest(df);st.session_state.update({"ready":True,"raw":raw,"df":df,"res":res,"valid":valid,"test":test})
 except Exception as e:st.session_state.ready=False;st.error(str(e))
if st.session_state.ready:
 df,res,valid,test=st.session_state.df,st.session_state.res,st.session_state.valid,st.session_state.test;best=valid.iloc[0].Modelo;err=valid.iloc[0]["Error selección_%"]
 zero=(df.Ventas==0).mean();kind="Intermitente" if zero>=.4 else "Regular / no intermitente"
 st.subheader("Diagnóstico");c1,c2,c3=st.columns(3);c1.metric("Tipo demanda",kind);c2.metric("Meses cero",f"{zero*100:.1f}%");c3.metric("Mejor modelo",best)
 if len(df)<18:st.info("LightGBM, XGBoost y Random Forest se activan desde 18 meses.")
 if len(df)<24:st.info("Naive estacional y Holt-Winters se activan desde 24 meses.")
 manual=st.selectbox("Modelo alternativo",res[res.Estado=="OK"].Modelo.tolist(),index=0)
 try:
  auto,man=future(df,best),future(df,manual);fix,dyn,detail=blends(df,valid)
  out=pd.DataFrame([horiz(auto,"Automático",best),horiz(man,"Manual",manual),horiz(fix,"Ensemble fijo Top 4","0,4/0,3/0,2/0,1"),horiz(dyn,"Ensemble dinámico Top 4","Inverso al error")])
  st.subheader("Forecast horizontal a 18 meses");st.dataframe(out,use_container_width=True)
  st.subheader("Pesos del ensemble");st.dataframe(detail,use_container_width=True)
  st.caption("Peso dinámico = (1/error del modelo) dividido por la suma de los inversos de error. Forecasts redondeados a 0 decimales después de combinar.")
  st.subheader("Comparativa walk-forward");show=res.copy()
  for c in ["MAPE_%","sMAPE_%","WMAPE_%","Error selección_%"]:show[c]=show[c].round(2)
  st.dataframe(show,use_container_width=True)
  row=res[res.Modelo==best].iloc[0];bt=test.copy();bt["Forecast"]=[row[f"BT{i}"] for i in range(1,5)];bt["Error %"]=np.where(bt.Ventas!=0,abs(bt.Ventas-bt.Forecast)/bt.Ventas*100,np.nan)
  st.subheader("Backtesting ganador");st.dataframe(bt,use_container_width=True)
  if PLOTLY:
   fig=go.Figure();fig.add_scatter(x=df.Fecha,y=df.Ventas,name="Histórico",mode="lines+markers")
   for f,n,d in [(auto,"Automático",None),(man,"Manual","dash"),(fix,"Top 4 fijo","dot"),(dyn,"Top 4 dinámico","dashdot")]:fig.add_scatter(x=f.Fecha,y=f.Forecast,name=n,mode="lines+markers",line=dict(dash=d) if d else None)
   fig.update_layout(template="plotly_white",height=520);st.plotly_chart(fig,use_container_width=True)
  d1,d2,d3=st.columns(3);d1.download_button("Forecast CSV",out.to_csv(index=False,sep=";").encode("utf-8-sig"),"forecast_v2.csv","text/csv",use_container_width=True);d2.download_button("Modelos CSV",res.to_csv(index=False,sep=";").encode("utf-8-sig"),"modelos_v2.csv","text/csv",use_container_width=True);d3.download_button("Pesos CSV",detail.to_csv(index=False,sep=";").encode("utf-8-sig"),"pesos_v2.csv","text/csv",use_container_width=True)
 except Exception as e:st.error(str(e))
