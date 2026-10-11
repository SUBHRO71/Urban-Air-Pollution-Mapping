from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from contextlib import asynccontextmanager

import pandas as pd
import torch
from fastapi import FastAPI,UploadFile,File,Form,HTTPException
from fastapi.responses import Response,HTMLResponse,FileResponse
from fastapi.staticfiles import StaticFiles
from . import datasets,models,engine,exports,storage as store
from .schemas import RunConfig,BatchConfig,CSVMapping,OpenAQImport

@asynccontextmanager
async def lifespan(app):
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    worker=subprocess.Popen([sys.executable,'-m','urban_air.worker'],cwd=store.ROOT,creationflags=flags) if os.environ.get('URBAN_AIR_NO_WORKER')!='1' else None
    yield
    if worker:
        worker.terminate()
        try:worker.wait(timeout=10)
        except subprocess.TimeoutExpired:worker.kill()

app=FastAPI(title='Urban Air Lab',version='0.1.0',lifespan=lifespan)

@app.middleware('http')
async def protect_local_writes(request,call_next):
    from urllib.parse import urlparse
    from fastapi.responses import JSONResponse
    origin=request.headers.get('origin')
    if request.method not in {'GET','HEAD','OPTIONS'} and origin and urlparse(origin).hostname not in {'localhost','127.0.0.1'}:
        return JSONResponse({'detail':'External websites cannot modify this local research workspace.'},status_code=403)
    return await call_next(request)

@app.exception_handler(ValueError)
async def bad_input(request,exc):
    from fastapi.responses import JSONResponse
    return JSONResponse({'detail':str(exc)},status_code=400)

@app.get('/api/health')
def health():
    return {'status':'ok','version':'0.1.0','cpu_threads':os.cpu_count(),'cuda':torch.cuda.is_available(),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU','source':'Live local experiment engine'}

@app.get('/api/datasets')
def dataset_list():return datasets.catalog()

@app.get('/api/models')
def model_list(dataset:str|None=None):
    meta=store.get(dataset) if dataset else None
    return models.catalog(meta['payload'] if meta else None)

@app.post('/api/datasets/import/{source}')
def source_import(source:str):
    try:
        if source=='india':return datasets.download_india()
        if source in {'uci','beijing'}:return datasets.download_uci(source=='beijing')
        raise ValueError('Unknown downloadable source.')
    except Exception as exc:
        if isinstance(exc,ValueError):raise
        raise HTTPException(502,'Source download failed. Check your connection and retry; existing snapshots are preserved.') from exc

@app.post('/api/datasets/openaq')
def openaq(config:OpenAQImport):
    try:return datasets.import_openaq(config)
    except Exception as exc:
        if isinstance(exc,ValueError):raise
        raise HTTPException(502,'OpenAQ request failed. Verify API access, sensor IDs and dates.') from exc

@app.post('/api/datasets/csv')
def upload_csv(file:UploadFile=File(),mapping:str=Form()):
    if not file.filename or not file.filename.lower().endswith('.csv'):raise ValueError('Upload a CSV file.')
    content=file.file.read(50_000_001)
    try:return datasets.import_csv(content,CSVMapping.model_validate_json(mapping))
    except (KeyError,TypeError) as exc:raise ValueError(f'CSV mapping is invalid: {exc}') from exc

@app.get('/api/datasets/{identifier}/preview')
def preview(identifier:str,target:str|None=None):
    meta,frame=datasets.read(identifier);target=target or next(iter(meta['targets']))
    if target not in meta['targets']:raise ValueError('Unknown dataset target.')
    means=frame.groupby('timestamp')[target].mean().dropna();step=max(1,len(means)//240)
    points=[{'timestamp':t.isoformat(),'observed':float(v)} for t,v in means.iloc[::step].items()]
    station_summary=[]
    for station,group in frame.groupby('station'):
        coordinate=next((c for c in meta['coordinates'] if c['station']==station),{})
        value=group[target].dropna()
        station_summary.append({'station':station,**coordinate,'mean':float(value.mean()) if len(value) else None,'count':len(value)})
    return {'points':points,'stations':station_summary,'target':target,'unit':meta['targets'][target],'summary':'Historical observed concentrations; station mean is descriptive, not a forecast.'}

@app.post('/api/experiments/preview')
def experiment_preview(config:RunConfig):return engine.preflight(config)

@app.post('/api/experiments')
def experiment(config:RunConfig):return engine.submit(config)

@app.get('/api/runs')
def runs():return store.records('run')

@app.get('/api/batches')
def batches():return store.records('batch')

@app.post('/api/batches/preview')
def batch_preview(config:BatchConfig):return engine.batch_preview(config)

@app.post('/api/batches')
def batch_submit(config:BatchConfig):return engine.submit_batch(config)

@app.post('/api/jobs/{identifier}/{action}')
def job_control(identifier:str,action:str):return engine.control(identifier,action)

@app.get('/api/runs/{identifier}')
def run_detail(identifier:str):
    record=store.get(identifier)
    if not record:raise HTTPException(404,'Run not found')
    return record|{'events':store.events(identifier)}

@app.get('/api/runs/{identifier}/predictions')
def predictions(identifier:str,station:str|None=None,fold:str|None=None):
    if not store.get(identifier):raise HTTPException(404,'Run not found')
    path=store.artifact_dir(identifier)/'predictions.parquet'
    if not path.exists():
        partial=list(store.artifact_dir(identifier).glob('*-predictions.parquet'))
        if not partial:return []
        frame=pd.concat([pd.read_parquet(p) for p in partial])
    else:frame=pd.read_parquet(path)
    if station:frame=frame[frame.station==station]
    if fold:frame=frame[frame.fold==fold]
    if not station:frame=frame[frame.station==frame.station.iloc[0]] if len(frame) else frame
    frame=frame.sort_values('timestamp');frame=frame.iloc[::max(1,len(frame)//500)]
    return json.loads(frame.to_json(orient='records',date_format='iso'))

@app.get('/api/runs/{identifier}/manifest')
def manifest(identifier:str):
    path=store.artifact_dir(identifier)/'manifest.json'
    if not path.exists():raise HTTPException(404,'Manifest not yet available')
    return json.loads(path.read_text())

@app.get('/api/export/{format}')
def export(format:str,scope:str|None=None):
    if format=='xlsx':return Response(exports.workbook(scope),media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',headers={'Content-Disposition':'attachment; filename="urban-air-comparison.xlsx"'})
    if format=='csv':return Response(exports.tables(scope)['Summary'].to_csv(index=False),media_type='text/csv',headers={'Content-Disposition':'attachment; filename="urban-air-summary.csv"'})
    if format=='zip':return Response(exports.bundle(scope),media_type='application/zip',headers={'Content-Disposition':'attachment; filename="urban-air-evidence.zip"'})
    if format=='html':return HTMLResponse(exports.report(scope))
    raise ValueError('Choose csv, xlsx, zip or html.')

dist=store.ROOT/'frontend'/'dist'
if dist.exists():
    app.mount('/assets',StaticFiles(directory=dist/'assets'),name='assets')
    @app.get('/')
    def index():return FileResponse(dist/'index.html')
