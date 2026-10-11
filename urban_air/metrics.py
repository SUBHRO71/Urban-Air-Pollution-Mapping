from __future__ import annotations
import numpy as np

def regression(truth, prediction, baseline=None, mape_threshold=0.01):
    y, p = np.asarray(truth, dtype=float).ravel(), np.asarray(prediction, dtype=float).ravel()
    valid = np.isfinite(y) & np.isfinite(p)
    eligible = int(np.isfinite(y).sum()); y, p = y[valid], p[valid]
    n = len(y)
    if not n:
        return {"n": 0, "eligible": eligible, "coverage": 0.0, "mae": None, "mse": None, "rmse": None, "r2": None, "mape": None, "smape": None, "skill": None, "reason": "No observed targets with valid predictions"}
    err = p-y; mse=float(np.mean(err**2)); mae=float(np.mean(np.abs(err)))
    denom=float(np.sum((y-y.mean())**2))
    r2 = float(1-np.sum(err**2)/denom) if n>=2 and denom>1e-12 else None
    m = np.abs(y)>mape_threshold
    mape = float(100*np.mean(np.abs(err[m]/y[m]))) if m.sum()>=max(1, np.ceil(.8*n)) else None
    sd=np.abs(y)+np.abs(p)
    smape=float(np.mean(np.divide(200*np.abs(err), sd, out=np.zeros(n), where=sd>0)))
    skill=None
    if baseline is not None:
        b=np.asarray(baseline,dtype=float).ravel()[valid]
        bv=np.isfinite(b)
        if bv.all():
            base_mae=np.mean(np.abs(b-y))
            if base_mae>0:skill=float(1-mae/base_mae)
    return {"n": n,"eligible":eligible,"coverage":round(n/max(eligible,1),6),"mae":mae,"mse":mse,"rmse":float(np.sqrt(mse)),"r2":r2,"mape":mape,"mape_support":int(m.sum()),"mape_threshold":mape_threshold,"smape":smape,"skill":skill,"r2_reason":None if r2 is not None else "Constant target or fewer than two observations","mape_reason":None if mape is not None else "Less than 80% of valid observations exceed the saved denominator threshold"}

BREAKPOINTS = {
    "PM2.5": [0,30,60,90,120,250,500],
    "PM10": [0,50,100,250,350,430,1000],
    "NO2": [0,40,80,180,280,400,1000],
    "SO2": [0,40,80,380,800,1600,2500],
    "CO": [0,1,2,10,17,34,50],
    "O3": [0,50,100,168,208,748,1000],
}
AQI_CLASSES=["Good", "Satisfactory", "Moderate", "Poor", "Very poor", "Severe"]

def subindex(values, pollutant):
    """Continuous research interpolation; full AQI requires averaging/coverage."""
    a=np.asarray(values,dtype=float)
    breaks=BREAKPOINTS[pollutant]
    return np.interp(a,breaks,[0,50,100,200,300,400,500],left=0,right=500)

def categories(y, p):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=int)
    if not len(y): return None
    prec,rec,f1,support=precision_recall_fscore_support(y,p,labels=range(6),zero_division=0)
    hits=int(((y>=3)&(p>=3)).sum()); misses=int(((y>=3)&(p<3)).sum()); false=int(((y<3)&(p>=3)).sum())
    return {"accuracy":float(accuracy_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p)),"precision_macro":float(prec.mean()),"recall_macro":float(rec.mean()),"f1_macro":float(f1.mean()),"confusion":confusion_matrix(y,p,labels=range(6)).tolist(),"classes":AQI_CLASSES,"support":support.tolist(),"per_class":[{"class":n,"precision":float(a),"recall":float(b),"f1":float(c),"support":int(d)} for n,a,b,c,d in zip(AQI_CLASSES,prec,rec,f1,support)],"macro_policy":"Fixed six-category universe; zero-support classes contribute zero.","event_threshold":"AQI category Poor or worse", "pod":hits/(hits+misses) if hits+misses else None,"far":false/(hits+false) if hits+false else None,"csi":hits/(hits+misses+false) if hits+misses+false else None}

def aqi_from_hourly(frame, units):
    """CPCB running-window AQI with minimum three pollutants incl. PM.

    Requires 16/24 hourly measurements for daily-window species and 6/8
    for eight-hour species. Input must already be an hourly indexed table.
    """
    import pandas as pd
    indices={}
    for name, breaks in BREAKPOINTS.items():
        if name not in frame or name not in units:continue
        unit=units[name].replace('μ','µ').replace('ug','µg').replace('m3','m³')
        expected="mg/m³" if name=="CO" else "µg/m³"
        if unit!=expected:continue
        window=8 if name in {"CO","O3"} else 24
        avg=frame[name].rolling(window,min_periods=6 if window==8 else 16).mean()
        indices[name]=pd.Series(subindex(avg.to_numpy(),name),index=avg.index).where(avg.notna())
    if not indices:return pd.Series(np.nan,index=frame.index)
    vals=pd.DataFrame(indices)
    pm=vals[[c for c in ['PM2.5','PM10'] if c in vals]].notna().any(axis=1)
    return vals.max(axis=1).where((vals.notna().sum(axis=1)>=3)&pm)
