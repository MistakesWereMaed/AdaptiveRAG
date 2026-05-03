from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, validator


class QAItem(BaseModel):
    id: int
    question: str
    gold: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("question", "gold")
    def must_be_str(cls, v):
        if v is None:
            raise ValueError("field cannot be None")
        if not isinstance(v, str):
            raise ValueError("field must be a string")
        return v.strip()


class PredictionItem(BaseModel):
    id: int
    prediction: str
    gold: Optional[str] = None
    strategy: str

    @validator("prediction")
    def prediction_must_be_str(cls, v):
        if v is None:
            raise ValueError("prediction cannot be None")
        if not isinstance(v, str):
            raise ValueError("prediction must be a string")
        return v.strip()


class StrategyPredictions(BaseModel):
    no_rag: List[PredictionItem] = Field(..., alias="no-rag")
    single: List[PredictionItem]
    multi: List[PredictionItem]

    model_config = ConfigDict(populate_by_name=True)


class RetrievedDocument(BaseModel):
    doc_id: str
    title: str = ""
    text: str
    score: Optional[float] = None
    rank: Optional[int] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("doc_id", "title", "text", pre=True)
    def _strip_strings(cls, value):
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return value

    @validator("text")
    def _text_must_exist(cls, value):
        if value is None or not str(value).strip():
            raise ValueError("text cannot be empty")
        return str(value).strip()
