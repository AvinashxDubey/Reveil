from pydantic import BaseModel, Field
from datetime import datetime


class PredictionRequest(BaseModel):
    
    created_at: datetime = Field(..., description="Account creation timestamp")
    username: str = Field(..., min_length=1, max_length=50, description="Twitter username")
    tweet: str = Field(..., min_length=1, description="Tweet text content")
    hashtags: str = Field(default="", description="Space-separated hashtags")
    retweet_count: int = Field(..., ge=0, description="Number of retweets")
    mention_count: int = Field(..., ge=0, description="Number of mentions (@)")
    follower_count: int = Field(..., ge=0, description="Number of followers")
    verified: bool = Field(..., description="Is account verified")
    
    class Config:
        json_schema_extra = {
            "example": {
                "created_at": "2023-01-15T10:30:00Z",
                "username": "john_doe123",
                "tweet": "Just deployed my new FastAPI bot detection API! Check it out at example.com #python #api #machinelearning",
                "hashtags": "#python #api #machinelearning",
                "retweet_count": 5,
                "mention_count": 2,
                "follower_count": 5000,
                "verified": True
            }
        }


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    user_id: int
    timestamp: datetime
    features_calculated: dict
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "human",
                "confidence": 0.8547,
                "user_id": 1,
                "timestamp": "2024-11-07T10:30:00Z",
                "features_calculated": {
                    "account_age_days": 662,
                    "hashtag_count": 3,
                    "username_length": 12,
                    "digit_count": 3,
                    "tweet_length": 106,
                    "retweet_per_hashtag": 1.25,
                    "mention_count": 2,
                    "follower_count": 5000,
                    "verified": 1
                }
            }
        }