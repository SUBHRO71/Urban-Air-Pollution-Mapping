from __future__ import annotations
import io
import json
import zipfile
import html
from pathlib import Path
import pandas as pd
from openpyxl.styles import Font,PatternFill
from . import storage as store

def scope(identifier=None):
    if not identifier:return store.records('run')
    record=store.get(identifier)
    if not record:raise ValueError('Experiment/batch not found.')
    return store.records('run',identifier) if record['kind']=='batch' else [record]

def safe(value):
    if isinstance(value,str) and value.startswith(('=','+','-','@')):return "'"+value
    return value

def tables(identifier=None):
    summary=[];folds=[];configs=[];quality=[]
    for run in scope(identifier):
        cfg=run['payload'];result=run['result'] or {}
        base={'run_id':run['id'],'dataset_id':cfg['dataset'],'dataset':result.get('dataset_name',cfg['dataset']),'model':cfg['model'],'target':cfg['target'],'unit':result.get('unit'),'horizon':cfg['horizon'],'preset':cfg['preset'],'status':run['status'],'reason':result.get('reason'),'seed':cfg['seed'],'cohort':result.get('cohort')}
        protocols=[]
        if result.get('holdout'):protocols.append(('Holdout',result['holdout']['metrics']))
        if result.get('cv'):protocols.append(('10-fold CV pooled',result['cv']['pooled']))
        if not protocols:protocols=[(cfg['evaluation'],{})]
        for protocol,metrics in protocols:
            row=base|{'protocol':protocol}|metrics
            cv=result.get('cv')
            if cv and protocol.startswith('10'):
                row.update({f'fold_mean_{k}':v for k,v in cv['mean'].items()});row.update({f'fold_std_{k}':v for k,v in cv['std'].items()});row['folds_completed']=cv['folds']
            row['elapsed_seconds']=result.get('elapsed_seconds');summary.append(row)
        for fold in result.get('folds',[]):
            bounds={f'{phase}_{k}':v for phase in ['train','validation','test'] for k,v in fold['split'][phase].items()}
            folds.append(base|{'fold':fold['name']}|fold['metrics']|bounds|{'training_seconds':fold['training_seconds'],'inference_ms':fold['inference_ms'],'parameter_count':fold['parameter_count']})
        configs.append({'run_id':run['id'],**cfg})
        snapshot=store.get(cfg['dataset'])
        if snapshot:
            meta=snapshot['payload']
            for feature,missing in meta['missing_percent'].items():quality.append({'dataset':meta['name'],'checksum':meta['checksum'],'feature':feature,'missing_percent':missing,'rows':meta['rows'],'frequency':meta['frequency'],'source_url':meta['source_url']})
    return {name:pd.DataFrame(rows).map(safe) if rows else pd.DataFrame() for name,rows in [('Summary',summary),('Folds',folds),('Configuration',configs),('Data Quality',quality)]}

def workbook(identifier=None):
    buffer=io.BytesIO()
    with pd.ExcelWriter(buffer,engine='openpyxl') as writer:
        for name,frame in tables(identifier).items():
            for col in frame:
                if frame[col].map(lambda v:isinstance(v,(list,dict))).any():frame[col]=frame[col].map(lambda v:json.dumps(v) if isinstance(v,(list,dict)) else v)
            frame.to_excel(writer,sheet_name=name,index=False)
            sheet=writer.sheets[name];sheet.freeze_panes='A2';sheet.auto_filter.ref=sheet.dimensions
            for cell in sheet[1]:cell.font=Font(color='FFFFFF',bold=True);cell.fill=PatternFill('solid',fgColor='14283F')
            for col in sheet.columns:
                letter=col[0].column_letter;sheet.column_dimensions[letter].width=min(42,max(12,max(len(str(c.value or '')) for c in col[:30])+2))
    return buffer.getvalue()

def report(identifier=None):
    table=tables(identifier)['Summary'].to_html(index=False,escape=True,na_rep='N/A')
    return '<!doctype html><html lang="en"><meta charset="utf-8"><title>Urban Air Lab - comparison report</title><style>body{font:14px/1.6 system-ui;margin:40px;color:#14283f}table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border:1px solid #d5dfe8}th{background:#14283f;color:white}tr:nth-child(even){background:#f1f5f9}h1{font-size:32px}@media print{body{margin:0}table{font-size:8px}}</style><h1>Urban Air Lab</h1><p>Chronological holdout and expanding-window ten-fold comparison. Metrics use observed targets in physical units. CV pooled metrics are recomputed from predictions; fold standard deviation is not a confidence interval. N/A is unavailable, never a fabricated zero. Quick checks use reduced history. Repeated holdout comparisons are exploratory.</p>'+table+'</html>'

def bundle(identifier=None):
    buffer=io.BytesIO();frames=tables(identifier)
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,frame in frames.items():archive.writestr(name.lower().replace(' ','_')+'.csv',frame.to_csv(index=False))
        archive.writestr('comparison.xlsx',workbook(identifier));archive.writestr('report.html',report(identifier))
        for run in scope(identifier):
            path=store.artifact_dir(run['id'])
            for pattern in ['manifest.json','result.json','predictions.parquet','*-preprocessing.json','*-graph.json','error.log']:
                for file in path.glob(pattern):archive.write(file,f"{run['id']}/{file.name}")
    return buffer.getvalue()
