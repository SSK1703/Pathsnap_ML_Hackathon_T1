from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class TripListResponse(BaseModel):
    trips: list[str]
    total: int


class PredictionRequest(BaseModel):
    trip_id: str = Field(min_length=1, max_length=128)


class EvaluationMetrics(BaseModel):
    top1_accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    log_loss: Optional[float] = None
    route_continuity: Optional[float] = None


class PredictionResponse(BaseModel):
    trip_id: str
    predicted_arc_ids: list[int]
    gps_points: list[list[float]] = Field(default_factory=list)
    predicted_route_segments: list[list[list[float]]] = Field(
        default_factory=list
    )
    metrics: Optional[EvaluationMetrics] = None
    processing_time_seconds: float
    message: str
