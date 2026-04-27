import json
import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from api.routes import router
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

_CHROMA_PATH  = Path("data/chroma_db")
_TICKETS_PATH = Path("data/mock/tickets.json")
_EMPTY_TICKETS = {"CUST001": [], "CUST002": [], "CUST003": [], "CUST004": []}


def _reset_demo_memory():
    """Wipe ChromaDB vector store and ticket history for a clean demo slate."""
    if _CHROMA_PATH.exists():
        shutil.rmtree(_CHROMA_PATH)
        print("[DEMO] ChromaDB memory cleared.", flush=True)

    with open(_TICKETS_PATH, "w") as f:
        json.dump(_EMPTY_TICKETS, f, indent=2)
    print("[DEMO] Ticket history reset.", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _reset_demo_memory()      # ── on startup: fresh slate ──
    yield
    _reset_demo_memory()      # ── on shutdown: leave no trace ──


app = FastAPI(
    title="TeleSupport AI",
    description="Intelligent telecom customer support agent — thesis prototype",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="data"), name="static")
app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return FileResponse("telecom_support_prototype.html")


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        reload=True,
        reload_dirs=["agent", "api", "services", "models"],  # only watch source dirs
    )
