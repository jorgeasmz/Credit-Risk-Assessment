import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request

from app.schemas import CreditApplication, PredictionResponse
from app.utils import format_prediction, load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads the model once at startup and keeps it on the application state.

    A failure here is logged rather than raised: the service still starts and
    answers the health check, so an orchestrator can report why it is degraded
    instead of watching the container restart in a loop.
    """
    try:
        app.state.model = load_model()
        logger.info("Model loaded.")
    except Exception:
        logger.exception("Model failed to load; /predict will return 503.")
        app.state.model = None

    yield

    app.state.model = None


app = FastAPI(
    title="German Credit Risk API",
    description="Operationalizes a Random Forest model to predict credit risk.",
    version="1.0.0",
    lifespan=lifespan,
)

# Set before startup so the health check works even if lifespan never ran.
app.state.model = None


def get_model(request: Request):
    """Dependency that yields the loaded model or fails the request."""
    model = request.app.state.model
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check the server logs.",
        )
    return model


@app.get("/")
def root(request: Request):
    """Health check. Reports whether the model is available."""
    return {
        "message": "Credit Risk Prediction API is operational. Visit /docs.",
        "model_loaded": request.app.state.model is not None,
    }


# Annotated is the current FastAPI idiom and keeps Depends() out of a default
# argument, where it would be evaluated once at import time.
ModelDependency = Annotated[Any, Depends(get_model)]


@app.post("/predict", response_model=PredictionResponse)
def predict(application: CreditApplication, model: ModelDependency):
    """
    Scores one credit application.

    Returns the risk class, a readable label and the probability of default.
    """
    try:
        return format_prediction(model, application.model_dump())
    except Exception:
        # Logged in full server-side; the client gets no internals.
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail="Prediction failed.") from None


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
