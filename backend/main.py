import os
from contextlib import asynccontextmanager
from pathlib import Path
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from config.settings import load_retrieval_config
from models.ml_models import initialize_models
from routers import chat, dashboard
from services.chunk_service import load_initial_data
from utils.logging_utils import ensure_chat_log_file

BASE_DIR = Path(__file__).resolve().parent.parent

def startup():
    load_retrieval_config()
    initialize_models()
    ensure_chat_log_file()
    load_initial_data()
    print("Application startup complete")

@asynccontextmanager
async def lifespan(app: FastAPI):
    startup()
    yield

def create_app() -> FastAPI:
    app = FastAPI(title="UNH Catalog RAG API", lifespan=lifespan)
    
    # setup CORS
    public_url = os.getenv("PUBLIC_URL", "http://localhost:8003/")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[public_url],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # include routers (no prefix since routes already have paths)
    app.include_router(chat.router, tags=["chat"])
    app.include_router(dashboard.router)
    
    # mount frontend static files
    frontend_path = BASE_DIR / "frontend" / "out"
    if frontend_path.is_dir():
        app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
        print(f"Mounted frontend from: {frontend_path}")
    
    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003, reload=False)