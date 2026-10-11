import argparse,json,time
from pathlib import Path
from . import engine,storage as store
from .schemas import BatchConfig
from .worker import process_one
from .exports import bundle

def main():
    parser=argparse.ArgumentParser(description='Shared Urban Air Lab benchmark engine')
    parser.add_argument('--datasets',default='all');parser.add_argument('--models',default='all');parser.add_argument('--evaluation',choices=['holdout','cv','both'],default='both')
    parser.add_argument('--config',type=Path);parser.add_argument('--resume');parser.add_argument('--dry-run',action='store_true');parser.add_argument('--output',type=Path,default=Path('results/benchmark.zip'))
    args=parser.parse_args()
    if args.resume:
        batch=engine.control(args.resume,'resume')
    else:
        cfg=json.loads(args.config.read_text()) if args.config else {}
        cfg.setdefault('settings',{});cfg['settings']['evaluation']=args.evaluation
        if args.datasets!='all':cfg['datasets']=args.datasets.split(',')
        if args.models!='all':cfg['models']=args.models.split(',')
        config=BatchConfig.model_validate(cfg)
        preview=engine.batch_preview(config)
        print(json.dumps(preview,indent=2))
        if args.dry_run:return
        batch=engine.submit_batch(config)
    print('Batch:',batch['id'])
    while store.get(batch['id'])['status']=='Running':
        if not process_one():time.sleep(1)
        state=store.get(batch['id']);print(state['status'],state['result'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_bytes(bundle(batch['id']))
    print('Evidence saved:',args.output)

if __name__=='__main__':main()
