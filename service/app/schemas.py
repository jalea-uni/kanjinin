from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class SourceInfo(BaseModel):
    type: Literal["image"] = "image"
    dpi: int = 300
    normalized: str = "A4"

class ItemResult(BaseModel):
    id: str
    page: int = 1
    bbox: List[int]  # [x,y,w,h] in pixels
    expected_kanji: Optional[str] = None
    predicted_kanji: Optional[str] = None
    confidence: Optional[float] = None
    quality: Optional[float] = None
    assets: Optional[dict] = None
    notes: Optional[List[str]] = None

class Summary(BaseModel):
    total_boxes: int
    matched: int
    accuracy_top1: Optional[float] = None
    avg_quality: Optional[float] = None
    processing_ms: Optional[int] = None
    model_version: Optional[str] = None

class EvaluateResponse(BaseModel):
    session_id: str
    source: SourceInfo
    summary: Summary
    items: List[ItemResult]

