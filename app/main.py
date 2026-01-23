from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.schemas import CreditApplication, PredictionResponse
from app.utils import load_model, format_prediction
import uvicorn
import os

# Global variable to hold the model in memory
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager that handles application startup and shutdown.
    
    1. Before 'yield': This code runs when the server starts (Loading the Model).
    2. yield: The application starts running and accepting requests.
    3. After 'yield': This code runs when the server stops (Cleanup, if needed).
    """
    global model
    # Define path relative to where python cmd is run
    model_path = os.path.join("model", "credit_risk_model.joblib")
    
    try:
        model = load_model(model_path)
        print("LifeSpan: Model loaded successfully.")
    except Exception as e:
        print(f"LifeSpan Startup Error: {e}")
    
    yield  # Control is passed to the application
    
    print("LifeSpan: Application shutting down.")
    model = None

# Initialize FastAPI app, passing the lifespan handler
app = FastAPI(
    title="German Credit Risk API",
    description="Operationalizes a Random Forest model to predict credit risk.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def root():
    """Health check endpoint."""
    return {"message": "Credit Risk Prediction API is operational. Visit /docs for Swagger UI."}

@app.post("/predict", response_model=PredictionResponse)
def predict(application: CreditApplication):
    """
    Endpoint to predict credit risk.
    
    - Receives a JSON body with credit application details.
    - Returns a JSON with the risk classification and probability.
    """
    global model
    
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model is not loaded. Please check server logs or restart the service."
        )
    
    try:
        result = format_prediction(model, application.model_dump())
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Prediction Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)