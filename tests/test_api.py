import os
from fastapi.testclient import TestClient
from urban_air.api import app

def test_api_contract_and_local_write_protection(workspace,monkeypatch):
    monkeypatch.setenv('URBAN_AIR_NO_WORKER','1')
    with TestClient(app) as client:
        assert client.get('/api/health').json()['status']=='ok'
        assert len(client.get('/api/models').json())==21
        assert client.post('/api/experiments',json={'dataset':'missing'}).status_code==400
        assert client.post('/api/datasets/import/unknown',headers={'Origin':'https://untrusted.example'}).status_code==403
        assert client.post('/api/datasets/csv',files={'file':('bad.exe',b'x')},data={'mapping':'{}'}).status_code==400
