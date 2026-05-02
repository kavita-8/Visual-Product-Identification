"""
search.py — Core reverse image search engine.

Responsibilities:
  - Load and cache the CLIP model
  - Extract L2-normalised embeddings from images
  - Build/persist/load a FAISS IndexFlatIP (cosine similarity via inner product)
  - Serve ranked search results with similarity scores
"""

from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor

from config import settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# ---------------------------------------------------------------------------
# Embedding extractor (singleton-style, loaded once)
# ---------------------------------------------------------------------------

class CLIPEmbedder:
    """Wraps CLIP model + processor; produces L2-normalised float32 embeddings."""

    def __init__(self, model_name: str = settings.clip_model_name):
        logger.info(f"Loading CLIP model: {model_name}")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        logger.info(f"CLIP model loaded on {self.device}.")

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """
        Returns a 1-D float32 ndarray of shape (embedding_dim,).
        The vector is L2-normalised so that inner product == cosine similarity.
        """
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self.model.get_image_features(**inputs)

        # get_image_features returns a plain tensor (batch_size, dim)
        # unwrap ModelOutput if transformers wraps it
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output if hasattr(features, "pooler_output") \
                else features.last_hidden_state[:, 0, :]

        # features shape: (1, D) — flatten to (D,)
        vec = features.reshape(-1).cpu().numpy().astype("float32")
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_bytes(self, image_bytes: bytes) -> np.ndarray:
        try:
            image = Image.open(io.BytesIO(image_bytes))
        except UnidentifiedImageError as e:
            raise ValueError("Cannot decode image bytes.") from e
        return self.embed_image(image)

    def embed_path(self, path: Path) -> np.ndarray:
        try:
            image = Image.open(path)
        except (OSError, UnidentifiedImageError) as e:
            raise ValueError(f"Cannot open image at {path}") from e
        return self.embed_image(image)


# ---------------------------------------------------------------------------
# Search engine
# ---------------------------------------------------------------------------

class ImageSearchEngine:
    """
    Manages a FAISS index over a directory of images.

    Usage:
        engine = ImageSearchEngine()
        engine.build_index()          # build from scratch (or load cache)
        results = engine.search(img_bytes, top_k=6)
    """

    def __init__(self):
        self.embedder = CLIPEmbedder()
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(settings.embedding_dim)
        self.image_paths: List[Path] = []

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(self, force_rebuild: bool = False) -> None:
        """
        Build the FAISS index from the configured image folder.
        If a saved index exists (and force_rebuild is False), load it instead.
        """
        if not force_rebuild and self._load_cached_index():
            return

        logger.info(f"Building index from: {settings.image_folder}")
        image_files = self._collect_images(settings.image_folder)

        if not image_files:
            logger.warning("No images found — index will be empty.")
            return

        # Reset state before rebuild so stale paths can't leak into the new index
        self.index = faiss.IndexFlatIP(settings.embedding_dim)
        self.image_paths = []

        embeddings, valid_paths = self._embed_all(image_files)

        if embeddings:
            matrix = np.stack(embeddings)          # (N, D) float32
            actual_dim = matrix.shape[1]
            if actual_dim != settings.embedding_dim:
                logger.warning(
                    f"Embedding dim mismatch: config says {settings.embedding_dim}, "
                    f"actual is {actual_dim}. Using actual."
                )
            self.index = faiss.IndexFlatIP(actual_dim)
            self.index.add(matrix)
            self.image_paths = valid_paths
            self._save_index()
            logger.info(f"Index built: {self.index.ntotal} vectors.")
        else:
            logger.warning("No valid embeddings — index is empty.")

    def _embed_all(
        self, paths: List[Path]
    ) -> tuple[List[np.ndarray], List[Path]]:
        embeddings, valid = [], []
        for i, path in enumerate(paths, 1):
            try:
                emb = self.embedder.embed_path(path)
                embeddings.append(emb)
                valid.append(path)
            except Exception as exc:
                logger.warning(f"[{i}/{len(paths)}] Skipping {path.name}: {exc}")
            if i % 100 == 0:
                logger.info(f"  Embedded {i}/{len(paths)} images…")
        return embeddings, valid

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_index(self) -> None:
        faiss.write_index(self.index, str(settings.faiss_index_path))
        with open(settings.image_paths_cache, "w") as f:
            json.dump([str(p) for p in self.image_paths], f, indent=2)
        logger.info(f"Index saved to {settings.faiss_index_path}.")

    def _load_cached_index(self) -> bool:
        if not settings.faiss_index_path.exists() or not settings.image_paths_cache.exists():
            return False
        try:
            self.index = faiss.read_index(str(settings.faiss_index_path))
            with open(settings.image_paths_cache) as f:
                self.image_paths = [Path(p) for p in json.load(f)]
            logger.info(
                f"Loaded cached index: {self.index.ntotal} vectors "
                f"from {settings.faiss_index_path}."
            )
            return True
        except Exception as exc:
            logger.warning(f"Failed to load cached index ({exc}); rebuilding.")
            return False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, image_bytes: bytes, top_k: int = settings.default_top_k) -> List[Dict[str, Any]]:
        """
        Run similarity search against the index.

        Args:
            image_bytes: Raw bytes of the query image.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: rank, path, score.
        """
        if self.index.ntotal == 0:
            return []

        query = self.embedder.embed_bytes(image_bytes).reshape(1, -1)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx == -1:
                continue
            results.append(
                {
                    "rank": rank,
                    "path": str(self.image_paths[idx]),
                    "score": round(float(score), 6),  # cosine similarity in [-1, 1]
                }
            )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_images(folder: Path) -> List[Path]:
        return sorted(
            p for p in folder.rglob("*")
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )