from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.health import router as health_router
from app.routes.zones import router as zones_router
from app.routes.recommendations import router as reco_router
from app.routes.places import router as places_router

app = FastAPI(title="OffPeak API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(zones_router)
app.include_router(reco_router)
app.include_router(places_router)
