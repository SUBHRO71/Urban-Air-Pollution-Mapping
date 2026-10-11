"""Persistent single-worker queue. Expensive training stays outside web requests."""
import argparse
import time
from filelock import FileLock,Timeout
from . import storage as store
from .engine import ACTIVE,run_experiment,refresh_batch

def process_one():
    store.HOME.mkdir(parents=True,exist_ok=True)
    try:
        with FileLock(store.HOME/'training.lock',timeout=0):
            return _process_one()
    except Timeout:return False

def _process_one():
    for record in reversed(store.records('run')):
        if record['status']!='Queued':continue
        parent=store.get(record['parent']) if record['parent'] else None
        if parent and parent['status'] in {'Paused','Cancelled'}:continue
        with store.connect() as db:
            claimed=db.execute("UPDATE records SET status='Validating' WHERE id=? AND status='Queued'",(record['id'],)).rowcount
        if not claimed:continue
        run_experiment(record['id']);return True
    for batch in store.records('batch'):refresh_batch(batch['id'])
    return False

def recover():
    for run in store.records('run'):
        if run['status'] in ACTIVE:
            store.update(run['id'],status='Interrupted');store.event(run['id'],'Worker restarted. Resume to continue saved folds/checkpoints.')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');args=parser.parse_args()
    recover()
    while True:
        if not process_one():
            if args.once:break
            time.sleep(.8)

if __name__=='__main__':main()
