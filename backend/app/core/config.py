from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):

    allowed_origins: str = "http://localhost:3000"
    database_url: str = "sqlite:///./bot_detection.db"

    secret_key: str
    access_token_expire_minutes: int = 30
    active_model: str = 'mlp'
    
    debug: bool = False
    
    class Config:
        env_file = Path(__file__).parents[2] / ".env"
        env_file_encoding = "utf-8"

settings = Settings()