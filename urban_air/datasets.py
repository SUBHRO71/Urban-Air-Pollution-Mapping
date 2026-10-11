from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
import gzip
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import storage as store
from .schemas import CSVMapping, OpenAQImport

SOURCES = [
    {"id": "uci-air-quality", "name": "UCI Air Quality", "source": "UCI · Italy", "description": "Hourly gas concentrations and sensor responses. A reproducible starting point.", "source_url": "https://archive.ics.uci.edu/dataset/360/air+quality", "status": "Not imported", "action": "uci"},
    {"id": "uci-beijing", "name": "Beijing Multi-Site", "source": "UCI · China", "description": "Twelve stations, six pollutants and meteorology. Supports temporal and statistical graph experiments.", "source_url": "https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data", "status": "Not imported", "action": "beijing"},
    {"id": "openaq", "name": "OpenAQ", "source": "Global monitoring", "description": "Import a bounded sensor history with an OpenAQ API key. Credentials remain server-side.", "source_url": "https://docs.openaq.org/", "status": "Needs configuration", "action": "openaq"},
    {"id": "india-cpcb-archive", "name": "India CPCB historical", "source": "CPCB via OpenAQ archive", "description": "Real hourly measurements from Aya Nagar, Delhi in 2020. A credential-free historical research snapshot.", "source_url": "https://docs.openaq.org/aws/about", "status": "Not imported", "action": "india"},
    {"id": "india-ogd", "name": "India Open Government Data", "source": "CPCB · data.gov.in", "description": "Import a verified historical export. The live OGD snapshot alone is insufficient for training.", "source_url": "https://www.data.gov.in/resource/real-time-air-quality-index-various-locations", "status": "Needs historical data", "action": "csv"},
]

def catalog():
    snapshots = [r["payload"] | {"id": r["id"], "status": r["status"]} for r in store.records("dataset") if r['status']!='Superseded']
    return snapshots + [s for s in SOURCES if not any(d.get("source_id") == s["id"] for d in snapshots)]

def save_snapshot(frame: pd.DataFrame, mapping: CSVMapping, source_id=None):
    frame = frame.copy()
    if "timestamp" not in frame or "station" not in frame:
        raise ValueError("A normalized dataset requires timestamp and station columns.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    rejected = int(frame["timestamp"].isna().sum())
    frame = frame.dropna(subset=["timestamp", "station"])
    frame["station"] = frame["station"].astype(str)
    features = mapping.features or [c for c in frame if c not in {"timestamp", "station", "latitude", "longitude"} and pd.api.types.is_numeric_dtype(frame[c])]
    targets = mapping.targets
    if not targets or not set(targets).issubset(features):
        raise ValueError("Map at least one numeric target with an explicit unit; targets must be included among historical features.")
    if not features or len(features) > 80:
        raise ValueError("Choose between 1 and 80 numeric features.")
    for c in features:
        frame[c] = pd.to_numeric(frame[c], errors="coerce").replace([np.inf, -np.inf, -200], np.nan)
    invalid_negative={}
    for c,unit in targets.items():
        if unit in {'µg/m³','μg/m³','ug/m3','mg/m³','mg/m3','ppb'}:
            invalid_negative[c]=int((frame[c]<0).sum())
            # Impossible negative concentrations are source-invalid, not severe episodes.
            frame.loc[frame[c]<0,c]=np.nan
    if len(frame) > 2_000_000 or frame["station"].nunique() > 256:
        raise ValueError("Local import limit: 2 million rows and 256 stations.")
    duplicate_count = int(frame.duplicated(["timestamp", "station"]).sum())
    coords = []
    if {"latitude", "longitude"}.issubset(frame):
        for station, group in frame.groupby("station"):
            lat = pd.to_numeric(group["latitude"], errors="coerce").dropna()
            lon = pd.to_numeric(group["longitude"], errors="coerce").dropna()
            if len(lat) and len(lon):
                if lat.nunique() != 1 or lon.nunique() != 1 or not (-90 <= lat.iloc[0] <= 90 and -180 <= lon.iloc[0] <= 180):
                    raise ValueError("Each station must have one valid, fixed coordinate pair.")
                coords.append({"station": str(station), "latitude": float(lat.iloc[0]), "longitude": float(lon.iloc[0])})
    frame = frame.groupby(["station", "timestamp"], as_index=False)[features].mean()
    parts = []
    for station, group in frame.groupby("station"):
        group = group.set_index("timestamp")[features].resample(mapping.frequency).mean()
        group["station"] = station
        parts.append(group.reset_index())
    frame = pd.concat(parts).sort_values(["timestamp", "station"]).reset_index(drop=True)
    if len(frame) < 50 or not any(frame[c].notna().sum() >= 24 for c in targets):
        raise ValueError("Too few usable historical observations. Import a time series, not a current snapshot.")
    raw = frame.to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%SZ").encode()
    checksum = hashlib.sha256(raw).hexdigest()
    identifier = f"ds-{checksum[:16]}"
    old = store.get(identifier)
    if old:
        payload=old['payload']|{'adapter_version':'1.1','invalid_negative_targets':invalid_negative}
        store.update(identifier,payload=payload)
        store.atomic_json(store.HOME/'snapshots'/identifier/'manifest.json',payload)
        return payload | {"id": identifier, "status": old["status"]}
    stations = sorted(frame.station.unique().tolist())
    grid_shape, grid_stations = None, None
    if len(coords) == len(stations):
        lats = sorted({p["latitude"] for p in coords}); lons = sorted({p["longitude"] for p in coords})
        regular = len(lats) >= 2 and len(lons) >= 2 and len(lats) * len(lons) == len(stations)
        for vals in [lats, lons]:
            if len(vals) > 2:
                regular = regular and bool(np.allclose(np.diff(vals), np.diff(vals)[0], atol=1e-6))
        if regular and len({(p['latitude'], p['longitude']) for p in coords}) == len(coords):
            grid_shape = [len(lats), len(lons)]
            grid_stations = [next(p["station"] for p in coords if p["latitude"] == a and p["longitude"] == b) for a in lats for b in lons]
    directory = store.HOME / "snapshots" / identifier
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / "observations.parquet", index=False)
    missing = {c: round(float(frame[c].isna().mean()) * 100, 2) for c in features}
    manifest = {
        "name": mapping.name, "source": mapping.source, "source_id": source_id,
        "source_url": mapping.source_url, "license": mapping.license, "checksum": checksum,
        "retrieved": store.now(), "rows": len(frame), "time_steps": frame.timestamp.nunique(),
        "start": frame.timestamp.min().isoformat(), "end": frame.timestamp.max().isoformat(),
        "features": features, "targets": targets, "stations": stations, "coordinates": coords,
        "frequency": mapping.frequency, "timezone": mapping.timezone, "rejected_timestamps": rejected,
        "duplicates_averaged": duplicate_count, "missing_percent": missing,
        "graph_ready": len(stations) > 1, "grid_shape": grid_shape, "grid_stations": grid_stations,
        "adapter_version": "1.1", "invalid_negative_targets":invalid_negative, "description": "Frozen historical observations; impossible negative concentrations are masked. Model inputs are processed within each training fold.",
    }
    store.atomic_json(directory / "manifest.json", manifest)
    store.create("dataset", manifest, "Ready", identifier=identifier)
    return manifest | {"id": identifier, "status": "Ready"}

def read(identifier):
    record = store.get(identifier)
    if not record or record["kind"] != "dataset":
        raise ValueError("Import a dataset snapshot before training.")
    return record["payload"], pd.read_parquet(store.HOME / "snapshots" / identifier / "observations.parquet")

def download_uci(beijing=False):
    source_id = "uci-beijing" if beijing else "uci-air-quality"
    existing = next((r for r in store.records("dataset") if r["payload"].get("source_id") == source_id), None)
    if existing:
        return existing["payload"] | {"id": existing["id"], "status": existing["status"]}
    url = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip" if beijing else "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
    response = requests.get(url, timeout=90); response.raise_for_status()
    if len(response.content) > 100_000_000:
        raise ValueError("Dataset archive exceeds the local import limit.")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        if beijing:
            names = [n for n in archive.namelist() if n.lower().endswith(".csv") and "PRSA" in n]
            if not names:
                nested = next(n for n in archive.namelist() if n.lower().endswith(".zip"))
                with zipfile.ZipFile(io.BytesIO(archive.read(nested))) as inner:
                    frame = pd.concat([pd.read_csv(io.BytesIO(inner.read(n))) for n in inner.namelist() if n.lower().endswith('.csv')], ignore_index=True)
            else:
                frame = pd.concat([pd.read_csv(io.BytesIO(archive.read(n))) for n in names], ignore_index=True)
            frame["timestamp"] = pd.to_datetime(frame[["year", "month", "day", "hour"]]).dt.tz_localize("Asia/Shanghai").dt.tz_convert("UTC")
            compass={name:i for i,name in enumerate(['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'])}
            angle=frame.wd.map(compass)*2*np.pi/16
            frame["wind_sin"] = np.sin(angle)
            frame["wind_cos"] = np.cos(angle)
            features = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM", "wind_sin", "wind_cos"]
            targets = {c: "µg/m³" for c in ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]}
            mapping = CSVMapping(name="Beijing · 12-station history", source="UCI Beijing Multi-Site", timezone="Asia/Shanghai", features=features, targets=targets, source_url=SOURCES[1]["source_url"], license="CC BY 4.0; cite Chen (2017), DOI 10.24432/C5RK5G")
        else:
            name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
            frame = pd.read_csv(io.BytesIO(archive.read(name)), sep=";", decimal=",").dropna(subset=["Date", "Time"])
            timestamps = pd.to_datetime(frame.Date + " " + frame.Time.str.replace(".", ":", regex=False), dayfirst=True, errors="coerce")
            frame["timestamp"] = timestamps.dt.tz_localize("Europe/Rome", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
            frame["station"] = "Italian sensor site"
            frame = frame.rename(columns={"CO(GT)": "CO", "NO2(GT)": "NO2", "NOx(GT)": "NOx", "C6H6(GT)": "Benzene", "NMHC(GT)": "NMHC"})
            features = [c for c in frame if c not in {"Date", "Time", "timestamp", "station"} and not c.startswith("Unnamed")]
            mapping = CSVMapping(name="UCI · Italian hourly sensors", source="UCI Air Quality", timezone="Europe/Rome", features=features, targets={"CO": "mg/m³", "NO2": "µg/m³", "NOx": "ppb", "Benzene": "µg/m³"}, source_url=SOURCES[0]["source_url"], license="Academic research; resource license wording requires review before redistribution")
    return save_snapshot(frame, mapping, source_id)

def import_csv(content: bytes, mapping: CSVMapping):
    if len(content) > 50_000_000:
        raise ValueError("CSV uploads are limited to 50 MB.")
    frame = pd.read_csv(io.BytesIO(content))
    if mapping.timestamp not in frame:
        raise ValueError(f"Timestamp column '{mapping.timestamp}' is missing.")
    timestamps = pd.to_datetime(frame[mapping.timestamp], errors="coerce")
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(mapping.timezone, ambiguous="NaT", nonexistent="NaT")
    frame["timestamp"] = timestamps.dt.tz_convert("UTC")
    frame["station"] = frame[mapping.station].astype(str) if mapping.station else "Uploaded station"
    for attr in ["latitude", "longitude"]:
        col = getattr(mapping, attr)
        if col:
            frame[attr] = pd.to_numeric(frame[col], errors="coerce")
    return save_snapshot(frame, mapping, "india-ogd" if "india" in mapping.source.lower() else None)

def import_openaq(config: OpenAQImport):
    key = config.api_key or os.environ.get("OPENAQ_API_KEY", "")
    if not key:
        raise ValueError("An OpenAQ API key is required. Add it in the import form or OPENAQ_API_KEY environment variable.")
    start, end = pd.to_datetime(config.date_from, utc=True), pd.to_datetime(config.date_to, utc=True)
    if end <= start or end-start > pd.Timedelta(days=730):
        raise ValueError("Choose a positive date range no longer than two years.")
    headers = {"X-API-Key": key}
    rows, units, sensors, coordinates = [], {}, [], []
    for location in config.location_ids:
        response=requests.get(f'https://api.openaq.org/v3/locations/{location}',headers=headers,timeout=45);response.raise_for_status()
        locations=response.json().get('results',[])
        if not locations:raise ValueError(f'OpenAQ location {location} was not found.')
        info=locations[0];station=f"OpenAQ location {info['id']}"
        coordinate=info.get('coordinates') or {}
        if coordinate.get('latitude') is not None and coordinate.get('longitude') is not None:coordinates.append({'station':station,**coordinate})
        for sensor in info.get('sensors',[]):
            if not config.sensor_ids or sensor['id'] in config.sensor_ids:sensors.append((sensor['id'],station))
    if not sensors:raise ValueError('No selected sensors belong to these verified locations.')
    if config.sensor_ids and set(config.sensor_ids)-{s[0] for s in sensors}:raise ValueError('One or more sensor IDs do not belong to the selected locations.')
    if len(sensors)>50:raise ValueError('Select fewer sensors; local import limit is 50.')
    for sensor,station in sensors:
        for page in range(1, 101):
            r = requests.get(f"https://api.openaq.org/v3/sensors/{sensor}/hours", headers=headers, params={"datetime_from": start.isoformat(), "datetime_to": end.isoformat(), "limit": 1000, "page": page}, timeout=45)
            r.raise_for_status(); records = r.json().get("results", [])
            for item in records:
                parameter = item.get("parameter", {})
                name = parameter.get("name", "unknown").upper().replace("PM25", "PM2.5")
                unit = parameter.get("units", "unknown")
                if name in units and units[name] != unit:
                    raise ValueError("Sensor units differ. Import separate unit-consistent snapshots.")
                units[name] = unit
                timestamp = item.get("period", {}).get("datetimeFrom", {}).get("utc")
                if not timestamp:
                    raise ValueError("Measurement timestamp missing in OpenAQ v3 response.")
                rows.append({"timestamp": timestamp, "station": station, "pollutant": name, "value": item.get("value")})
            if len(records) < 1000:
                break
        else:
            raise ValueError("OpenAQ page limit reached. Choose a shorter history.")
    if not rows:
        raise ValueError("No OpenAQ historical observations matched this selection.")
    long = pd.DataFrame(rows)
    frame = long.pivot_table(index=["timestamp", "station"], columns="pollutant", values="value", aggfunc="mean").reset_index()
    for col in ['latitude','longitude']:
        coordinate_map={item['station']:item[col] for item in coordinates}
        frame[col]=frame.station.map(coordinate_map)
    mapping = CSVMapping(name=config.name, source="OpenAQ", targets=units, features=list(units), source_url="https://docs.openaq.org/resources/measurements", license="OpenAQ provider-specific terms; consult source licenses before redistribution")
    return save_snapshot(frame, mapping, "openaq")

def download_india():
    """Bounded historical CPCB measurements, explicitly sourced via OpenAQ."""
    existing=next((r for r in store.records('dataset') if r['payload'].get('source_id')=='india-cpcb-archive'),None)
    if existing:return existing['payload']|{'id':existing['id'],'status':existing['status']}
    base='https://openaq-data-archive.s3.amazonaws.com/'
    response=requests.get(base,params={'list-type':'2','prefix':'records/csv.gz/locationid=5570/year=2020/','max-keys':1000},timeout=60);response.raise_for_status()
    tree=ET.fromstring(response.content)
    keys=[item.text for item in tree.findall('.//{*}Key')][:120]
    if not keys:raise ValueError('No archived CPCB daily files found for the selected history.')
    def fetch(key):
        r=requests.get(base+key,timeout=60);r.raise_for_status()
        return pd.read_csv(io.BytesIO(gzip.decompress(r.content)))
    with ThreadPoolExecutor(max_workers=4) as pool:long=pd.concat(pool.map(fetch,keys),ignore_index=True)
    units={}
    for name,group in long.groupby('parameter'):
        if group.units.nunique()!=1:raise ValueError('Archive contains mixed units; import a unit-specific selection instead.')
        units[str(name).upper().replace('PM25','PM2.5')]=group.units.iloc[0]
    long['pollutant']=long.parameter.str.upper().str.replace('PM25','PM2.5',regex=False)
    long['timestamp']=pd.to_datetime(long['datetime'],utc=True)
    frame=long.pivot_table(index=['timestamp','location'],columns='pollutant',values='value',aggfunc='mean').reset_index().rename(columns={'location':'station'})
    coordinates=long.groupby('location')[['lat','lon']].first()
    frame['latitude']=frame.station.map(coordinates.lat);frame['longitude']=frame.station.map(coordinates.lon)
    mapping=CSVMapping(name='Delhi · Aya Nagar CPCB history',source='India CPCB historical via OpenAQ',timezone='Asia/Calcutta',features=list(units),targets=units,source_url=base+'?list-type=2&prefix=records/csv.gz/locationid=5570/year=2020/',license='CPCB provider records via OpenAQ; provider-specific terms, attribution required')
    return save_snapshot(frame,mapping,'india-cpcb-archive')
