# src/server.py
from fastapi import APIRouter, FastAPI
from fastapi.responses import RedirectResponse

from src.api.v1.endpoints.named_entity_recognition.ner_router import router as ner_router
from src.api.v1.endpoints.sentiment_analysis.sa_router import router as sentiment_analysis_router
from src.api.v1.endpoints.summarization.sum_router import router as summarization_router

app = FastAPI()

v1_router = APIRouter(prefix="/api/v1")


@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")


# Include all routers under v1
v1_router.include_router(sentiment_analysis_router)
v1_router.include_router(ner_router)
v1_router.include_router(summarization_router)

app.include_router(v1_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
