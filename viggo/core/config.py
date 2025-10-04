# viggo/core/config.py
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    groq_api_key: str
    
    # LLM Configuration
    llm_model: str = "llama-3.1-8b-instant"  # Current model
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1000
    
    # Database Configuration
    neo4j_uri: str = "bolt://20.216.195.227:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "viggo123"
    
    # Search Configuration
    elasticsearch_url: str = "https://viggo-search.search.windows.net"
    elasticsearch_api_key: str
    elasticsearch_index_prefix: str = "viggo"
    
    # Redis Configuration
    redis_host: str = "viggo-redis.redis.cache.windows.net"
    redis_port: int = 6380
    redis_password: str
    redis_ssl: bool = True
    redis_db: int = 0
    
    # Cache Configuration
    cache_ttl: int = 3600  # 1 hour default TTL
    cache_max_size: int = 1000  # Max cache entries
    
    # ML Configuration
    mlflow_tracking_uri: str = "http://localhost:5000"
    wandb_project: str = "viggo-lore-companion"
    data_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
    
    # Application Configuration
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra fields instead of raising validation errors

settings = Settings()

