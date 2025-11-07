from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
import bcrypt
from fastapi import Depends, status, HTTPException
from app.core.config import settings

ALGORITHM = "HS256"

bearer = HTTPBearer()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(
            plain_password.encode('utf-8'), 
            hashed_password.encode('utf-8')
        )
    except Exception:
        return False

def get_password_hash(password: str) -> str:
    try:
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters")
        if len(password) > 72:
            raise ValueError("Password cannot exceed 72 characters")
        
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Error hashing password: {e}")

def create_access_token(user_id: int, email: str, expires_delta: Optional[timedelta] = None):
    if not email:
        raise ValueError("Email is required to create token")
    if not user_id:
        raise ValueError("User ID is required to create token")
    
    to_encode = {
        "user_id": user_id,
        "email": email,
    }
    
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
    
        user_id: int = payload["user_id"]
        email: str = payload["email"]
        
        return {
            "user_id": user_id,
            "email": email
        }
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )    
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload"
        )
        
def get_token_data(credentials: HTTPAuthorizationCredentials = Depends(bearer)) -> dict:
    token = credentials.credentials
    return verify_token(token)