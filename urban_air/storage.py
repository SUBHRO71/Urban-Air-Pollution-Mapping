from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(os.environ.get("URBAN_AIR_HOME", ROOT / "data"))
LOCK = threading.Lock()

def now() -> str:
    return datetime.now(timezone.utc).isoformat()

def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def connect():
    HOME.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(HOME / "lab.sqlite3", timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, kind TEXT, status TEXT, parent TEXT, fingerprint TEXT, created TEXT, updated TEXT, payload TEXT, result TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, run TEXT, time TEXT, message TEXT)")
    db.commit()
    return db

def decode(row):
    if row is None:
        return None
    item = dict(row)
    item["payload"] = json.loads(item["payload"])
    item["result"] = json.loads(item["result"]) if item["result"] else None
    return item

def create(kind, payload, status="Queued", parent=None, fingerprint=None, identifier=None):
    identifier = identifier or f"{kind[:3]}-{uuid.uuid4().hex[:12]}"
    timestamp = now()
    with LOCK, connect() as db:
        db.execute("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?)", (identifier, kind, status, parent, fingerprint, timestamp, timestamp, json.dumps(payload, allow_nan=False), None))
    return get(identifier)

def get(identifier):
    with connect() as db:
        return decode(db.execute("SELECT * FROM records WHERE id=?", (identifier,)).fetchone())

def records(kind=None, parent=None):
    clauses, args = [], []
    if kind:
        clauses.append("kind=?"); args.append(kind)
    if parent:
        clauses.append("parent=?"); args.append(parent)
    query = "SELECT * FROM records" + (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY created DESC"
    with connect() as db:
        return [decode(r) for r in db.execute(query, args).fetchall()]

def update(identifier, status=None, result=None, payload=None):
    assignments, values = ["updated=?"], [now()]
    for key, val in [("status", status), ("result", result), ("payload", payload)]:
        if val is not None:
            assignments.append(f"{key}=?")
            values.append(json.dumps(val, allow_nan=False) if key != "status" else val)
    with LOCK, connect() as db:
        db.execute(f"UPDATE records SET {', '.join(assignments)} WHERE id=?", [*values, identifier])
    return get(identifier)

def event(identifier, message):
    with LOCK, connect() as db:
        db.execute("INSERT INTO events(run,time,message) VALUES (?,?,?)", (identifier, now(), message))

def events(identifier):
    with connect() as db:
        return [dict(r) for r in db.execute("SELECT time,message FROM events WHERE run=? ORDER BY id DESC LIMIT 100", (identifier,)).fetchall()][::-1]

def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)

def artifact_dir(identifier):
    if not identifier.replace("-", "").isalnum():
        raise ValueError("Invalid artifact ID")
    p = HOME / "artifacts" / identifier
    p.mkdir(parents=True, exist_ok=True)
    return p
