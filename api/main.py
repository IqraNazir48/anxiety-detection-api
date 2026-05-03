# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time

# ── Pipeline imports ──────────────────────────────────────
from pipeline.face_detector import FaceDetector
from pipeline.geometry      import compute_geometry
from pipeline.cnn_inference import CNNInference
from pipeline.score_fusion  import compute_score

# ── Global pipeline objects (loaded once at startup) ──────
detector = None
cnn      = None
start_time = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, cnn, start_time
    print("🚀 Loading pipeline...")
    detector   = FaceDetector()
    cnn        = CNNInference()
    start_time = time.time()
    print("✅ Pipeline ready")
    from db.database import create_db
    create_db()
    print("✅ Database ready")
    yield
    print("🛑 Shutting down")
    if detector:
        detector.close()

# ── App ───────────────────────────────────────────────────
app = FastAPI(
    title       = "Anxiety Detection API",
    description = "Real-Time Facial Tension & Anxiety Detection — DIP Project",
    version     = "1.0.0",
    lifespan    = lifespan
)

# ── CORS (allow mobile apps to call this API) ─────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Include routes ────────────────────────────────────────
from api.routes import router
app.include_router(router)