import io,json,zipfile
import numpy as np
import pytest
import torch
from urban_air import models,engine,storage,datasets,exports
from urban_air.schemas import RunConfig,BatchConfig

NEURAL=[m['id'] for m in models.catalog() if m['group']!='Reference baselines']

@pytest.mark.parametrize('kind',NEURAL)
def test_every_network_learns_from_actual_input_tensor(kind):
    torch.manual_seed(4);torch.set_num_threads(2)
    x=torch.randn(4,8,4,3);y=torch.randn(4,4)
    net=models.make_network(kind,3,8,4,8,[2,2],np.eye(4))
    before=[p.detach().clone() for p in net.parameters()]
    optimizer=torch.optim.Adam(net.parameters(),lr=.01)
    pred=net(x);assert pred.shape==(4,4);assert torch.isfinite(pred).all()
    loss=((pred-y)**2).mean();loss.backward();optimizer.step()
    assert any(not torch.equal(a,b) for a,b in zip(before,net.parameters()))

def test_shared_engine_fold_evidence_and_exports(grid_snapshot):
    cfg=RunConfig(dataset=grid_snapshot['id'],target='NO2',model='persistence',evaluation='both',lookback=24,max_origins=700)
    record=engine.submit(cfg);engine.run_experiment(record['id']);done=storage.get(record['id'])
    assert done['status']=='Completed',done['result']
    assert done['result']['cv']['folds']==10
    artifact=storage.artifact_dir(record['id']);manifest=json.loads((artifact/'manifest.json').read_text())
    assert len(manifest['splits'])==11
    import pandas as pd
    predictions=pd.read_parquet(artifact/'predictions.parquet');cv=predictions[predictions.fold!='Holdout']
    from urban_air.metrics import regression
    assert done['result']['cv']['pooled']['rmse']==pytest.approx(regression(cv.observed,cv.predicted)['rmse'])
    assert ((predictions.timestamp-predictions.origin).dt.total_seconds()==3600).all()
    workbook=exports.workbook(record['id']);assert workbook.startswith(b'PK')
    with zipfile.ZipFile(io.BytesIO(exports.bundle(record['id']))) as archive:assert record['id']+'/manifest.json' in archive.namelist()
    assert engine.submit(cfg)['id']==record['id']

def test_pause_resume_and_terminal_status_not_overwritten(grid_snapshot):
    record=engine.submit(RunConfig(dataset=grid_snapshot['id'],target='NO2',evaluation='holdout'))
    engine.control(record['id'],'pause');storage.update(record['id'],status='Training');assert storage.get(record['id'])['status']=='Paused'
    engine.control(record['id'],'resume');assert storage.get(record['id'])['status']=='Queued'
    engine.control(record['id'],'cancel');storage.update(record['id'],status='Evaluating');assert storage.get(record['id'])['status']=='Cancelled'

def test_batch_accounts_for_unprepared_sources_and_incompatible_models(grid_snapshot):
    config=BatchConfig(datasets=[grid_snapshot['id'],'openaq'],models=['persistence','unet'],settings={'evaluation':'holdout'})
    plan=engine.batch_preview(config);assert plan['total']==4;assert plan['compatible']==2;assert plan['unavailable']==2
    batch=engine.submit_batch(config);children=storage.records('run',batch['id']);assert len(children)==4
    assert sum(c['status']=='Unavailable' for c in children)==2

@pytest.mark.parametrize('kind',['gru','cnn_lstm','tft','gcn_lstm','stgnn','unet','convlstm','convlstm_attention','autoencoder_lstm'])
def test_trained_model_artifact_roundtrip_on_grid(kind,grid_snapshot):
    cfg=RunConfig(dataset=grid_snapshot['id'],target='NO2',model=kind,evaluation='holdout',lookback=8,epochs=1,hidden=8,max_origins=600)
    record=engine.submit(cfg);engine.run_experiment(record['id']);run=storage.get(record['id'])
    assert run['status']=='Completed',run['result']
    assert run['result']['holdout']['metrics']['coverage']==1
    assert (storage.artifact_dir(run['id'])/'00-model.pt').exists()
