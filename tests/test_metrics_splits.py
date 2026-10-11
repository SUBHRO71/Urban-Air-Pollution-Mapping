import numpy as np
import pandas as pd
import pytest
from urban_air.metrics import regression,aqi_from_hourly,categories
from urban_air.splits import make_splits
from urban_air.engine import normalize

def test_known_regression_and_edge_cases():
    m=regression([1,2,3],[2,2,2])
    assert m['mae']==pytest.approx(2/3)
    assert m['mse']==pytest.approx(2/3)
    assert m['rmse']==pytest.approx(np.sqrt(2/3))
    assert m['r2']==pytest.approx(0)
    assert regression([4,4],[3,5])['r2'] is None
    assert regression([0,0],[0,0])['mape'] is None
    assert regression([0,0],[0,0])['smape']==0
    assert regression([1,2,3],[1,np.nan,3])['coverage']==pytest.approx(2/3,abs=1e-6)

def test_ten_folds_are_disjoint_and_purged():
    splits,cutoff=make_splits(1200,.8,6,'both')
    folds=splits[1:];assert len(folds)==10
    observed=[]
    for s in splits:
        assert s.train[-1]+6<s.validation[0]
        assert s.validation[-1]+6<s.test[0]
    for s in folds:
        assert s.test[-1]<cutoff
        observed.extend(s.test)
    assert len(observed)==len(set(observed))
    with pytest.raises(ValueError,match='ten|Ten|10'):make_splits(300,.8,1,'cv')

def test_future_sentinel_cannot_change_training_preprocessing():
    x=np.arange(800,dtype=np.float32).reshape(100,4,2,1);y=np.arange(200,dtype=np.float32).reshape(100,2)
    _,_,_,_,first=normalize(np.arange(40),x,y)
    changed=x.copy();changed[40:]=1e9
    _,_,_,_,second=normalize(np.arange(40),changed,y)
    assert first==second

def test_aqi_coverage_units_and_category_counts():
    frame=pd.DataFrame({'PM2.5':[30.]*24,'NO2':[40.]*24,'SO2':[40.]*24})
    valid=aqi_from_hourly(frame,{c:'µg/m³' for c in frame})
    assert np.isnan(valid.iloc[14]);assert valid.iloc[-1]==pytest.approx(50)
    assert aqi_from_hourly(frame[['NO2','SO2']],{'NO2':'µg/m³','SO2':'µg/m³'}).isna().all()
    assert aqi_from_hourly(frame,{c:'ppb' for c in frame}).isna().all()
    c=categories([0,1,5],[0,5,5]);assert c['accuracy']==pytest.approx(2/3);assert sum(c['support'])==3
    assert sum(map(sum,c['confusion']))==3
