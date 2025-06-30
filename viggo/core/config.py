# viggo/core/config.py
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    groq_api_key: str
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str
    mlflow_tracking_uri: str = "http://localhost:5000"
    wandb_project: str = "viggo-lore-companion"
    data_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")

    class Config:
        env_file = ".env"

settings = Settings()

