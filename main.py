"""
Reverse Image Search Engine — FastAPI Backend
Uses CLIP embeddings + FAISS for fast similarity search.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from search import ImageSearchEngine
from config import settings
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing search engine...")
    engine = ImageSearchEngine()
    engine.build_index()
    app.state.engine = engine
    logger.info(f"Index ready — {engine.index.ntotal} images indexed.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Reverse Image Search API",
    description="CLIP + FAISS-powered reverse image search",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow browser to call the API from the HTML file
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve images from static/images/ at /images/<filename>
app.mount("/images", StaticFiles(directory=str(settings.image_folder)), name="images")


# --- Routes ---

@app.get("/", include_in_schema=False)
def index():
    return FileResponse("index.html")


@app.get("/health")
def health():
    engine: ImageSearchEngine = app.state.engine
    return {"status": "ok", "indexed_images": engine.index.ntotal}


@app.post("/search")
async def search(file: UploadFile = File(...), top_k: int = 6):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    if top_k < 1 or top_k > 50:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 50.")

    image_bytes = await file.read()
    engine: ImageSearchEngine = app.state.engine

    try:
        results = engine.search(image_bytes, top_k=top_k)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail="Search failed. See server logs.")

    return JSONResponse({"query_file": file.filename, "results": results})


@app.post("/index/rebuild")
def rebuild_index(background_tasks: BackgroundTasks):
    engine: ImageSearchEngine = app.state.engine
    background_tasks.add_task(engine.build_index, force_rebuild=True)
    return {"message": "Index rebuild started in the background."}


@app.get("/index/stats")
def index_stats():
    engine: ImageSearchEngine = app.state.engine
    return {
        "total_indexed": engine.index.ntotal,
        "image_folder": str(settings.image_folder),
        "index_path": str(settings.faiss_index_path),
        "embedding_dim": settings.embedding_dim,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)