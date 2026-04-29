from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, validator


class QAItem(BaseModel):
    id: int
    question: str
    gold: str

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
