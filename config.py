"""
config.py — Central configuration via environment variables or defaults.
Override any value by setting the corresponding env var (e.g. IMAGE_FOLDER=/data/imgs).
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Directories
    image_folder: Path = Path("static/images")
    faiss_index_path: Path = Path("data/faiss.index")
    image_paths_cache: Path = Path("data/image_paths.json")

    # CLIP model
    clip_model_name: str = "openai/clip-vit-base-patch32"
    embedding_dim: int = 512

    # Search
    default_top_k: int = 6

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def ensure_dirs(self):
        self.image_folder.mkdir(parents=True, exist_ok=True)
        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
