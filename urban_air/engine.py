from __future__ import annotations

import copy
import importlib.metadata
import json
import os
import random
import subprocess
import time
import traceback

import joblib
import numpy as np
import pandas as pd
import torch

from . import datasets, models, storage as store
from .metrics import regression, categories
from .schemas import RunConfig, BatchConfig
from .splits import make_splits

TERMINAL={'Completed','Failed','Skipped incompatible','Unavailable','Cancelled'}
ACTIVE={'Validating','Preparing','Training','Evaluating'}

class StopRun(Exception):
    pass

def code_revision():
    try:return subprocess.check_output(['git','rev-parse','HEAD'],cwd=store.ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):return 'unversioned'

def preflight(config: RunConfig):
    manifest,frame=datasets.read(config.dataset)
    if config.target not in manifest['targets']:raise ValueError('Choose an available target from this dataset.')
    model=next((m for m in models.catalog(manifest) if m['id']==config.model),None)
    if not model:raise ValueError('Unknown algorithm ID.')
    if not model['compatible']:raise ValueError(model['reason'])
    if config.model=='seasonal_naive' and config.lookback<24:
        raise ValueError('Seasonal naive requires a lookback of at least 24 hourly observations.')
    if config.model=='seasonal_naive' and config.horizon>24:
        raise ValueError('The initial seasonal baseline supports horizons up to 24 hours.')
    if config.stations and not set(config.stations).issubset(manifest['stations']):raise ValueError('Selected station is absent from the dataset.')
    selected=config.stations or manifest['stations']
    if model['modality']=='graph' and len(selected)<2:raise ValueError('A graph experiment requires at least two selected stations.')
    if model['modality']=='grid' and sorted(selected)!=sorted(manifest['stations']):raise ValueError('Grid models need the complete observed grid. Select all grid stations.')
    frame=frame[frame.station.isin(selected)]
    times=frame.timestamp.nunique()
    if config.preset=='quick':times=min(times,config.max_origins+config.lookback+config.horizon)
    n=times-config.lookback-config.horizon+1
    splits,cutoff=make_splits(n,config.train_ratio,config.horizon,config.evaluation)
    return {'dataset':manifest,'model':model,'origins':n,'folds':len(splits),'holdout_cutoff':cutoff,'estimates':{'fits':len(splits),'epochs_per_fit':config.epochs,'device':config.device},'protocol':'Chronological; training labels purged before evaluation origins','label':'Quick check - reduced history' if config.preset=='quick' else 'Research run - full selected history'}

def submit(config: RunConfig,parent=None,force=False):
    manifest,_=datasets.read(config.dataset)
    fingerprint=store.digest({'config':config.model_dump(),'snapshot':manifest['checksum'],'engine':'1.0','code':code_revision()})
    if not force and parent is None:
        cached=next((r for r in store.records('run') if r['fingerprint']==fingerprint and r['status'] in {'Completed','Queued',*ACTIVE,'Paused','Interrupted'}),None)
        if cached:return cached
    preflight(config)
    record=store.create('run',config.model_dump(),parent=parent,fingerprint=fingerprint)
    store.event(record['id'],'Experiment queued. Dataset snapshot and evaluation settings frozen.')
    return record

def batch_preview(config: BatchConfig):
    data=datasets.catalog();ids=config.datasets or [d['id'] for d in data]
    modelids=config.models or [m['id'] for m in models.catalog()]
    if len(ids)*len(modelids)>1000:raise ValueError('Batch limit is 1,000 pairings.')
    pairs=[]
    for identifier in ids:
        dataset=next((d for d in data if d['id']==identifier),None)
        if not dataset:raise ValueError(f'Unknown dataset {identifier}')
        target=config.targets.get(identifier) or next(iter(dataset.get('targets',{'PM2.5':'µg/m³'})))
        for model in modelids:
            payload={**config.settings,'dataset':identifier,'model':model,'target':target}
            candidate=RunConfig.model_validate(payload)
            status='Queued';reason=None
            if dataset['status']!='Ready':status='Unavailable';reason=dataset['status']
            else:
                try:preflight(candidate)
                except ValueError as exc:status='Skipped incompatible';reason=str(exc)
            pairs.append({'config':candidate.model_dump(),'status':status,'reason':reason,'dataset_name':dataset['name']})
    return {'pairs':pairs,'total':len(pairs),'compatible':sum(p['status']=='Queued' for p in pairs),'unavailable':sum(p['status']!='Queued' for p in pairs),'fits':sum((11 if p['config']['evaluation']=='both' else 10 if p['config']['evaluation']=='cv' else 1) for p in pairs if p['status']=='Queued')}

def submit_batch(config: BatchConfig):
    preview=batch_preview(config)
    parent=store.create('batch',config.model_dump()|{'plan':preview},'Running')
    for pair in preview['pairs']:
        cfg=RunConfig.model_validate(pair['config'])
        if pair['status']=='Queued':submit(cfg,parent=parent['id'],force=True)
        else:store.create('run',cfg.model_dump(),pair['status'],parent=parent['id'],fingerprint=store.digest(pair['config']))
        child=next(r for r in store.records('run',parent['id']) if r['payload']['dataset']==cfg.dataset and r['payload']['model']==cfg.model)
        if pair['reason']:store.update(child['id'],result={'reason':pair['reason']})
    refresh_batch(parent['id'])
    return store.get(parent['id'])

def refresh_batch(identifier):
    batch=store.get(identifier)
    if not batch or batch['status'] in {'Paused','Cancelled'}:return
    children=store.records('run',identifier)
    counts={s:sum(c['status']==s for c in children) for s in {c['status'] for c in children}}
    result={'counts':counts,'completed':counts.get('Completed',0),'terminal':sum(c['status'] in TERMINAL for c in children),'total':len(children)}
    status='Running'
    if children and all(c['status'] in TERMINAL for c in children):status='Completed with issues' if any(c['status']!='Completed' for c in children) else 'Completed'
    store.update(identifier,status=status,result=result)

def control(identifier,action):
    if action not in {'pause','resume','cancel'}:raise ValueError('Unknown run control.')
    record=store.get(identifier)
    if not record:raise ValueError('Run not found.')
    if action=='pause':
        if record['status'] in TERMINAL or record['status']=='Completed with issues':return record
        store.update(identifier,status='Paused');store.event(identifier,'Paused. Neural checkpoints and completed folds are retained.')
    elif action=='cancel':
        if record['status']=='Completed':return record
        store.update(identifier,status='Cancelled');store.event(identifier,'Cancelled; completed evidence is retained.')
        if record['kind']=='batch':
            for child in store.records('run',identifier):
                if child['status'] not in TERMINAL:store.update(child['id'],status='Cancelled')
    elif action=='resume':
        if record['status'] not in {'Paused','Interrupted','Failed'}:return record
        store.update(identifier,status='Running' if record['kind']=='batch' else 'Queued')
        if record['kind']=='batch':
            for child in store.records('run',identifier):
                if child['status'] in {'Paused','Interrupted'}:store.update(child['id'],status='Queued')
        store.event(identifier,'Resumed with unchanged configuration; completed folds are reused.')
    return store.get(identifier)

def prepare(config,manifest,frame):
    stations=config.stations or manifest['grid_stations'] or manifest['stations']
    if config.model not in {'convlstm','convlstm_attention','unet'}:stations=config.stations or manifest['stations']
    frame=frame[frame.station.isin(stations)]
    times=pd.date_range(frame.timestamp.min(),frame.timestamp.max(),freq=manifest['frequency'],tz='UTC')
    if config.preset=='quick':times=times[-(config.max_origins+config.lookback+config.horizon):]
    features=config.feature_columns or manifest['features']
    if not set(features).issubset(manifest['features']):raise ValueError('Feature selection includes an unavailable column.')
    if config.target not in features:features=[config.target,*features]
    arrays=[]
    for station in stations:
        group=frame[frame.station==station].set_index('timestamp').reindex(times)
        arrays.append(group[features].to_numpy(dtype=np.float32))
    raw=np.stack(arrays,1)
    target_index=features.index(config.target)
    target=raw[:,:,target_index].copy()
    # Each window contains only observations at or before its forecast origin.
    starts=np.arange(len(times)-config.lookback-config.horizon+1)
    x=np.stack([raw[s:s+config.lookback] for s in starts])
    origins=starts+config.lookback-1;labels=origins+config.horizon
    y=target[labels].copy()
    historical_coverage=np.isfinite(x).any(-1).mean(1)
    y[historical_coverage<.7]=np.nan
    latest=x[:,-1,:,target_index].copy()
    # Carry the latest target within observed context only, never across an origin.
    for b in range(len(x)):
        for n in range(len(stations)):
            history=x[b,:,n,target_index];finite=history[np.isfinite(history)]
            latest[b,n]=finite[-1] if len(finite) else np.nan
    return x,y,latest,times,origins,labels,stations,features,target_index

def normalize(train,x,y):
    observed=x[train]
    with warnings_context():
        med=np.nanmedian(observed,axis=(0,1,2));mean=np.nanmean(observed,axis=(0,1,2));std=np.nanstd(observed,axis=(0,1,2))
    med=np.nan_to_num(med);mean=np.nan_to_num(mean);std=np.nan_to_num(std,nan=1);std[std<1e-6]=1
    filled=np.where(np.isfinite(x),x,med)
    normalized=((filled-mean)/std).astype(np.float32)
    train_y=y[train];finite=train_y[np.isfinite(train_y)]
    if len(finite)<24:raise ValueError('Training partition has fewer than 24 observed target values.')
    ym=float(finite.mean());ys=max(float(finite.std()),1e-6)
    return normalized,(y-ym)/ys,ym,ys,{'medians':med.tolist(),'means':mean.tolist(),'scales':std.tolist(),'target_mean':ym,'target_scale':ys,'fit_indices':[int(train[0]),int(train[-1])]}

def warnings_context():
    import warnings
    context=warnings.catch_warnings();context.__enter__();warnings.simplefilter('ignore',RuntimeWarning)
    class C:
        def __enter__(self):return self
        def __exit__(self,*args):context.__exit__(*args)
    return C()

def run_experiment(identifier):
    record=store.get(identifier);config=RunConfig.model_validate(record['payload']);folder=store.artifact_dir(identifier)
    started=time.perf_counter()
    def check():
        current=store.get(identifier);parent=store.get(current['parent']) if current['parent'] else None
        if current['status'] in {'Paused','Cancelled'} or (parent and parent['status'] in {'Paused','Cancelled'}):
            if parent and current['status'] not in {'Paused','Cancelled'}:store.update(identifier,status=parent['status'])
            raise StopRun()
    try:
        check();store.update(identifier,status='Preparing');store.event(identifier,'Preparing chronological forecast windows; targets remain observed-only.')
        manifest,frame=datasets.read(config.dataset);info=preflight(config)
        x,y,baseline,times,origins,labels,stations,features,target_index=prepare(config,manifest,frame)
        splits,cutoff=make_splits(len(x),config.train_ratio,config.horizon,config.evaluation)
        split_manifest=[]
        for split in splits:
            bounds={}
            for name,indices in [('train',split.train),('validation',split.validation),('test',split.test)]:
                bounds[name]={'origins':len(indices),'first_origin':times[origins[indices[0]]].isoformat(),'last_origin':times[origins[indices[-1]]].isoformat(),'last_label':times[labels[indices[-1]]].isoformat()}
            assert times[labels[split.train[-1]]] < times[origins[split.validation[0]]]
            assert times[labels[split.validation[-1]]] < times[origins[split.test[0]]]
            split_manifest.append({'name':split.name,**bounds,'purge_origins':config.horizon,'train_indices':split.train.tolist(),'validation_indices':split.validation.tolist(),'test_indices':split.test.tolist()})
        cohort=store.digest({'snapshot':manifest['checksum'],'target':config.target,'horizon':config.horizon,'lookback':config.lookback,'ratio':config.train_ratio,'preset':config.preset,'features':features,'stations':stations,'seed':config.seed,'splits':split_manifest})
        provenance={'config':config.model_dump(),'snapshot':manifest,'engine_version':'1.0','model_version':'1.0','code_revision':code_revision(),'cohort':cohort,'feature_columns':features,'stations':stations,'splits':split_manifest,'category_reason':'No valid full AQI/category target was selected; concentration accuracy is not percentage accuracy.','mape_threshold':.01,'environment':{p:importlib.metadata.version(p) for p in ['torch','numpy','pandas','scikit-learn','fastapi']},'gpu_nondeterminism':'GPU kernels may differ numerically; no bitwise equality is promised.'}
        store.atomic_json(folder/'manifest.json',provenance)
        all_results=[];all_predictions=[]
        for index,split in enumerate(splits):
            check();fold_path=folder/f'{index:02d}-result.json';pred_path=folder/f'{index:02d}-predictions.parquet'
            if fold_path.exists() and pred_path.exists():
                all_results.append(json.loads(fold_path.read_text()));all_predictions.append(pd.read_parquet(pred_path));store.event(identifier,f'{split.name}: reused completed fold.');continue
            store.update(identifier,status='Training',result={'progress':{'fold':index+1,'total':len(splits),'stage':split.name},'folds':all_results})
            store.event(identifier,f'{split.name}: fitting on earlier observations; inner validation controls stopping.')
            random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed)
            nx,ny,ym,ys,scaler=normalize(split.train,x,y)
            store.atomic_json(folder/f'{index:02d}-preprocessing.json',scaler)
            fit_start=time.perf_counter();losses=[];device='cpu';parameter_count=None
            kind=config.model
            if kind in {'persistence','seasonal_naive'}:
                def predict(indices):
                    if kind=='persistence':return baseline[indices].copy()
                    offset=24-config.horizon
                    if offset<0:return baseline[indices].copy()
                    return x[indices,max(0,config.lookback-1-offset),:,target_index].copy()
                store.atomic_json(folder/f'{index:02d}-model.json',{'model':kind,'rule':'Last observation' if kind=='persistence' else '24-hour seasonal cycle','version':'1.0'})
            elif kind in {'random_forest','gradient_boosting'}:
                estimator=models.fit_classical(kind,nx[split.train],ny[split.train],config.seed)
                joblib.dump(estimator,folder/f'{index:02d}-model.joblib')
                def predict(indices):return models.predict_classical(estimator,nx[indices])*ys+ym
            elif kind=='arima':
                from statsmodels.tsa.arima.model import ARIMA
                fitted=[]
                for n in range(len(stations)):
                    series=y[split.train,n]
                    if np.isfinite(series).sum()<24:fitted.append(None);continue
                    with warnings_context():fitted.append(ARIMA(series,order=(2,1,1)).fit())
                joblib.dump(fitted,folder/f'{index:02d}-model.joblib')
                def predict(indices):
                    out=np.full((len(indices),len(stations)),np.nan)
                    for b,i in enumerate(indices):
                        for n,model in enumerate(fitted):
                            if model is None:continue
                            history=x[i,:,n,target_index]
                            with warnings_context():out[b,n]=float(model.apply(history,refit=False).forecast(config.horizon)[-1])
                    return out
            else:
                adjacency=models.graph_from_training(nx[split.train]) if kind in {'stgnn','gcn_lstm'} else None
                if adjacency is not None:store.atomic_json(folder/f'{index:02d}-graph.json',{'recipe':'Positive feature-average station correlation fitted on outer training windows; row-normalized with self loops.','station_order':stations,'adjacency':adjacency.tolist(),'training_range':scaler['fit_indices']})
                net=models.make_network(kind,len(features),config.hidden,len(stations),config.lookback,manifest.get('grid_shape'),adjacency)
                def progress(epoch,tl,vl):
                    store.update(identifier,result={'progress':{'fold':index+1,'total':len(splits),'stage':split.name,'epoch':epoch,'epochs':config.epochs,'train_loss':tl,'validation_loss':vl},'folds':all_results})
                net,losses,device=models.train_neural(net,nx[split.train],ny[split.train],nx[split.validation],ny[split.validation],config,folder/f'{index:02d}-checkpoint.pt',check,progress)
                torch.save(net.state_dict(),folder/f'{index:02d}-model.pt');parameter_count=sum(p.numel() for p in net.parameters())
                def predict(indices):return models.predict_neural(net,nx[indices],device,config.batch_size)*ys+ym
            training_seconds=time.perf_counter()-fit_start;check();store.update(identifier,status='Evaluating')
            train_pred=predict(split.train);val_pred=predict(split.validation)
            tick=time.perf_counter();test_pred=predict(split.test);inference_ms=(time.perf_counter()-tick)*1000/max(len(split.test)*len(stations),1)
            metrics=regression(y[split.test],test_pred,baseline[split.test]);train_metrics=regression(y[split.train],train_pred,baseline[split.train]);validation_metrics=regression(y[split.validation],val_pred,baseline[split.validation])
            category=None
            if manifest['targets'][config.target]=='AQI index':
                mask=np.isfinite(y[split.test])&np.isfinite(test_pred)
                category=categories(np.clip(np.searchsorted([50,100,200,300,400],y[split.test][mask],side='left'),0,5),np.clip(np.searchsorted([50,100,200,300,400],test_pred[mask],side='left'),0,5))
            result={'name':split.name,'metrics':metrics,'train_metrics':train_metrics,'validation_metrics':validation_metrics,'category':category,'training_seconds':training_seconds,'inference_ms':inference_ms,'device':device,'parameter_count':parameter_count,'losses':losses,'split':split_manifest[index]}
            rows=[]
            for b,i in enumerate(split.test):
                for n,station in enumerate(stations):
                    rows.append({'run':identifier,'fold':split.name,'origin':times[origins[i]],'timestamp':times[labels[i]],'station':station,'target':config.target,'unit':manifest['targets'][config.target],'observed':float(y[i,n]),'predicted':float(test_pred[b,n]),'baseline':float(baseline[i,n])})
            predictions=pd.DataFrame(rows);predictions.to_parquet(pred_path,index=False);store.atomic_json(fold_path,result)
            all_results.append(result);all_predictions.append(predictions)
            store.event(identifier,f'{split.name} completed. MAE {metrics["mae"]:.4f}' if metrics['mae'] is not None else f'{split.name}: no scored targets.')
        predictions=pd.concat(all_predictions,ignore_index=True);predictions.to_parquet(folder/'predictions.parquet',index=False)
        holdout=next((r for r in all_results if r['name']=='Holdout'),None)
        cv=[r for r in all_results if r['name']!='Holdout'];cv_summary=None
        if cv:
            cvp=predictions[predictions.fold!='Holdout'];pooled=regression(cvp.observed,cvp.predicted,cvp.baseline)
            mean={};std={}
            for key in ['mae','mse','rmse','r2','mape','smape','skill']:
                values=[r['metrics'].get(key) for r in cv if r['metrics'].get(key) is not None]
                mean[key]=float(np.mean(values)) if values else None;std[key]=float(np.std(values,ddof=1)) if len(values)>1 else None
            cv_summary={'folds':len(cv),'mean':mean,'std':std,'pooled':pooled,'category':None}
            if manifest['targets'][config.target]=='AQI index':
                mask=np.isfinite(cvp.observed)&np.isfinite(cvp.predicted)
                cv_summary['category']=categories(np.searchsorted([50,100,200,300,400],cvp.observed[mask],side='left'),np.searchsorted([50,100,200,300,400],cvp.predicted[mask],side='left'))
        result={'holdout':holdout,'cv':cv_summary,'folds':all_results,'cohort':cohort,'dataset_name':manifest['name'],'unit':manifest['targets'][config.target],'elapsed_seconds':time.perf_counter()-started,'category_reason':provenance['category_reason'],'snapshot_checksum':manifest['checksum'],'memory_note':'Peak memory not sampled; no value fabricated.','selection_note':'Exploratory comparison; repeated holdout inspection is model-selection exposure.','rank_eligible':config.preset=='research' and all(r['metrics']['coverage']==1 and r['metrics']['n']>0 for r in all_results)}
        store.atomic_json(folder/'result.json',result)
        check();store.update(identifier,status='Completed',result=result);store.event(identifier,'Completed. Predictions, split manifests, model and fold-local preprocessing saved.')
    except StopRun:
        store.event(identifier,'Stopped at a safe boundary. Completed folds and neural checkpoints remain available.')
    except Exception as exc:
        existing=store.get(identifier)
        if existing['status'] not in {'Paused','Cancelled'}:store.update(identifier,status='Failed',result={'reason':str(exc),'folds':existing.get('result',{}).get('folds',[]) if existing.get('result') else []})
        (folder/'error.log').write_text(traceback.format_exc(),encoding='utf-8');store.event(identifier,f'Failed: {exc}')
    finally:
        parent=record.get('parent')
        if parent:refresh_batch(parent)
