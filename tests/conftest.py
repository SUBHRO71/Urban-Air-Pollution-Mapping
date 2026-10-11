import numpy as np
import pandas as pd
import pytest
from urban_air import storage,datasets
from urban_air.schemas import CSVMapping

@pytest.fixture
def workspace(tmp_path,monkeypatch):
    monkeypatch.setattr(storage,'HOME',tmp_path/'lab')
    return storage.HOME

@pytest.fixture
def grid_snapshot(workspace):
    # Clearly synthetic, independently generated cell observations for pipeline QA.
    times=pd.date_range('2024-01-01',periods=750,freq='h',tz='UTC');rows=[]
    for t,stamp in enumerate(times):
        for n,(lat,lon) in enumerate([(28.,77.),(28.,77.1),(28.1,77.),(28.1,77.1)]):
            rows.append({'timestamp':stamp,'station':f'QA cell {n}','NO2':40+8*np.sin(t/24)+n*2+.02*t,'temperature':20+np.sin(t/12),'latitude':lat,'longitude':lon})
    return datasets.save_snapshot(pd.DataFrame(rows),CSVMapping(name='Synthetic QA grid - not research evidence',source='Automated test fixture',features=['NO2','temperature'],targets={'NO2':'µg/m³'},latitude='latitude',longitude='longitude'))
