from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from app.core.database import get_db
from app.core.security import get_token_data
from app.core.model_loader import get_model
from app.schemas.predictSchema import PredictionRequest, PredictionResponse

router = APIRouter(prefix="/predict", tags=["Prediction"])


def extract_features(request: PredictionRequest) -> dict:
    now = datetime.now(timezone.utc)
    if request.created_at.tzinfo is None:
        created_at = request.created_at.replace(tzinfo=timezone.utc)
    else:
        created_at = request.created_at
    account_age_days = (now - created_at).days
    
    hashtag_list = [h.strip() for h in request.hashtags.split() if h.strip()]
    hashtag_count = len(hashtag_list)
    
    username_length = len(request.username)
    digit_count = sum(1 for c in request.username if c.isdigit())
    
    tweet_length = len(request.tweet)
    
    retweet_per_hashtag = request.retweet_count / (hashtag_count + 1)
    
    verified = 1 if request.verified else 0
    
    return {
        "account_age_days": account_age_days,
        "hashtag_count": hashtag_count,
        "username_length": username_length,
        "digit_count": digit_count,
        "tweet_length": tweet_length,
        "retweet_per_hashtag": retweet_per_hashtag,
        "mention_count": request.mention_count,
        "follower_count": request.follower_count,
        "verified": verified
    }


@router.post("/", response_model=PredictionResponse)
def predict_bot(
    request: PredictionRequest,
    token_data: dict = Depends(get_token_data),
    db: Session = Depends(get_db)
):
    user_id = token_data["user_id"]
    
    try:
        features_dict = extract_features(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing features: {str(e)}"
        )
    
    features = [
        float(features_dict["account_age_days"]),
        float(features_dict["hashtag_count"]),
        float(features_dict["username_length"]),
        float(features_dict["digit_count"]),
        float(features_dict["tweet_length"]),
        features_dict["retweet_per_hashtag"],
        float(features_dict["mention_count"]),
        float(features_dict["follower_count"]),
        float(features_dict["verified"])
    ]
    
    model = get_model()
    
    try:
        result = model.predict(features)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    return PredictionResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        user_id=user_id,
        timestamp=datetime.now(timezone.utc),
        features_calculated=features_dict
    )