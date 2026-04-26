import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from api.routes import router
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="TeleSupport AI",
    description="Intelligent telecom customer support agent — thesis prototype",
    version="1.0.0",
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
