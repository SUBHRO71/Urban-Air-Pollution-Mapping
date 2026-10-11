from pydantic import BaseModel, Field
from typing import Literal

class RunConfig(BaseModel):
    dataset: str
    model: str = "random_forest"
    target: str = "CO"
    evaluation: Literal["holdout", "cv", "both"] = "both"
    lookback: int = Field(default=24, ge=2, le=168)
    horizon: int = Field(default=1, ge=1, le=72)
    train_ratio: float = Field(default=0.8, ge=0.6, le=0.9)
    epochs: int = Field(default=8, ge=1, le=300)
    patience: int = Field(default=3, ge=1, le=30)
    batch_size: int = Field(default=64, ge=8, le=512)
    hidden: int = Field(default=32, ge=8, le=128)
    learning_rate: float = Field(default=0.001, ge=0.00001, le=0.05)
    seed: int = Field(default=42, ge=0, le=2**31-1)
    preset: Literal["quick", "research"] = "quick"
    max_origins: int = Field(default=1200, ge=600, le=200000)
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    stations: list[str] = Field(default_factory=list)
    feature_columns: list[str] = Field(default_factory=list)

class BatchConfig(BaseModel):
    datasets: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    settings: dict = Field(default_factory=dict)
    targets: dict[str, str] = Field(default_factory=dict)

class CSVMapping(BaseModel):
    name: str = "Uploaded dataset"
    source: str = "User CSV"
    timestamp: str = "timestamp"
    station: str | None = None
    timezone: str = "UTC"
    features: list[str] = Field(default_factory=list)
    targets: dict[str, str] = Field(default_factory=dict)
    latitude: str | None = None
    longitude: str | None = None
    frequency: Literal["h", "D"] = "h"
    source_url: str = ""
    license: str = "User-declared research input; redistribution not verified"

class OpenAQImport(BaseModel):
    name: str = "OpenAQ historical snapshot"
    sensor_ids: list[int] = Field(min_length=1, max_length=20)
    date_from: str
    date_to: str
    api_key: str = ""

