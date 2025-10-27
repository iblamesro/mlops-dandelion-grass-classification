"""
Configuration module for MLOps project
"""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings"""
    
    # Project
    PROJECT_NAME: str = "MLOps Dandelion vs Grass"
    VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Database
    DB_USER: str = Field(default="mlops_user", env="DB_USER")
    DB_PASSWORD: str = Field(default="mlops_password", env="DB_PASSWORD")
    DB_HOST: str = Field(default="localhost", env="DB_HOST")
    DB_PORT: int = Field(default=5432, env="DB_PORT")
    DB_NAME: str = Field(default="mlops_db", env="DB_NAME")
    
    @property
    def database_url(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    # MinIO / S3
    MINIO_ENDPOINT: str = Field(default="localhost:9000", env="MINIO_ENDPOINT")
    MINIO_ACCESS_KEY: str = Field(default="minioadmin", env="MINIO_ACCESS_KEY")
    MINIO_SECRET_KEY: str = Field(default="minioadmin", env="MINIO_SECRET_KEY")
    MINIO_SECURE: bool = Field(default=False, env="MINIO_SECURE")
    
    # S3 Buckets
    S3_BUCKET_MODELS: str = "mlops-models"
    S3_BUCKET_DATA: str = "mlops-data"
    S3_BUCKET_ARTIFACTS: str = "mlops-artifacts"
    
    # MLflow
    MLFLOW_TRACKING_URI: str = Field(default="http://localhost:5000", env="MLFLOW_TRACKING_URI")
    MLFLOW_EXPERIMENT_NAME: str = "dandelion-grass-classification"
    
    # Model
    MODEL_NAME: str = "dandelion_grass_classifier"
    IMAGE_SIZE: int = 224
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 10
    LEARNING_RATE: float = 0.001
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    LOGS_DIR: Path = BASE_DIR / "logs"
    
    # API
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    
    # Monitoring
    PROMETHEUS_PORT: int = 9090
    GRAFANA_PORT: int = 3000
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()

# Create necessary directories
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
