import joblib
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(path: str = "model/credit_risk_model.joblib"):
    """
    Loads the serialized scikit-learn pipeline from disk.
    
    Args:
        path (str): Relative path to the joblib file.
        
    Returns:
        The loaded model object or None if failed.
    """
    if not os.path.exists(path):
        logger.error(f"Model file not found at: {path}")
        raise FileNotFoundError(f"Model file not found at {path}. Please run 'python model/train.py' first.")
    
    try:
        logger.info(f"Loading model from {path}...")
        model = joblib.load(path)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

def format_prediction(model, input_data: dict) -> dict:
    """
    Prepares input data, runs inference, and structures the response.
    
    Args:
        model: The trained scikit-learn pipeline.
        input_data (dict): The raw input dictionary from the API request.
        
    Returns:
        dict: A dictionary matching the PredictionResponse schema.
    """
    # 1. Convert dictionary to DataFrame
    df = pd.DataFrame([input_data])
    
    # 2. Predict Class
    prediction = model.predict(df)[0]
    
    # 3. Predict Probability
    # predict_proba returns an array of shape (n_samples, n_classes).
    # We want the probability of class 1 (Risk/Bad Credit).
    probs = model.predict_proba(df)
    
    # Check shape to ensure we get the right index
    if probs.shape[1] == 2:
        probability = probs[0][1]
    else:
        # Edge case: if model only trained on one class
        probability = float(prediction)

    return {
        "risk_class": int(prediction),
        "risk_label": "High Risk" if prediction == 1 else "Low Risk",
        "probability": float(probability)
    }