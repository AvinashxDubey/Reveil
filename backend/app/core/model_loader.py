import pandas as pd
import joblib
import json
from pathlib import Path
import numpy as np
from typing import List, Optional
from app.core.config import settings

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"


class BotDetectionModel:
    
    def __init__(self):
        self.scaler = None
        self.model = None
        self.feature_columns: Optional[List[str]] = None
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        scaler_path = ARTIFACTS_DIR / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")
        else:
            print(f"Scaler not found at {scaler_path}")
        
        model_path = ARTIFACTS_DIR / settings.active_model
        if model_path.exists():
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        features_path = ARTIFACTS_DIR / "feature_columns.json"
        if features_path.exists():
            with open(features_path, "r", encoding="utf-8") as f:
                self.feature_columns = json.load(f)
            print(f"Features loaded: {len(self.feature_columns)} columns")
        else:
            print(f"Features file not found at {features_path}")
    
    def predict(self, features: List[float]) -> dict:
        if self.feature_columns and len(features) != len(self.feature_columns):
            raise ValueError(
                f"Expected {len(self.feature_columns)} features, got {len(features)}"
            )
        
        if self.feature_columns:
            X = pd.DataFrame([features], columns=self.feature_columns)
        else:
            X = np.array(features).reshape(1, -1)
        
        
        if self.scaler:
            try:
                X = self.scaler.transform(X)
            except Exception as e:
                raise ValueError(f"Scaler error: {e}")
        
        if self.model is None:
            raise ValueError("Model not loaded")
        
        prediction = self.model.predict(X)[0]
        
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            confidence = float(max(proba))
        else:
            confidence = 0.5
        
        prediction_label = "bot" if int(prediction) == 1 else "human"
        
        return {
            "prediction": prediction_label,
            "confidence": round(confidence, 4)
        }


_model_instance: Optional[BotDetectionModel] = None


def get_model() -> BotDetectionModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = BotDetectionModel()
    return _model_instance