from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    
    secret_key: str
    access_token_expire_minutes: int
    active_model: str

    debug: True
    
    class Config:
        env_file = ".env"

settings = Settings()