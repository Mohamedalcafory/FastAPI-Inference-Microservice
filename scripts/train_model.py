"""
Training script for creating a sample sklearn model.
"""
import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Add the app directory to Python path
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

# Configure logging
configure_logging()
logger = get_logger("train_model")


def train_iris_model():
    """Train a simple iris classification model."""
    logger.info("Loading iris dataset...")
    
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")
    
    # Train model
    logger.info("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Add version attribute
    model.version = "v1.0.0"
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Model accuracy: {accuracy:.4f}")
    logger.info("Classification report:")
    logger.info(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    return model, iris


def save_model(model, model_path: str):
    """Save model to file."""
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    joblib.dump(model, model_path)
    logger.info(f"Model saved to: {model_path}")


def main():
    """Main training function."""
    try:
        # Get settings
        settings = get_settings()
        
        logger.info("Starting model training...")
        
        # Train model
        model, iris = train_iris_model()
        
        # Save model
        save_model(model, settings.MODEL_PATH)
        
        # Test model loading
        logger.info("Testing model loading...")
        loaded_model = joblib.load(settings.MODEL_PATH)
        
        # Test prediction
        test_input = np.array([[5.1, 3.5, 1.4, 0.2]])  # Sample iris data
        prediction = loaded_model.predict(test_input)
        logger.info(f"Test prediction: {prediction[0]} ({iris.target_names[prediction[0]]})")
        
        logger.info("Model training completed successfully!")
        
        # Print model info
        print("\n" + "="*50)
        print("MODEL TRAINING COMPLETED")
        print("="*50)
        print(f"Model saved to: {settings.MODEL_PATH}")
        print(f"Model version: {model.version}")
        print(f"Feature names: {iris.feature_names}")
        print(f"Target names: {iris.target_names}")
        print(f"Model type: {type(model).__name__}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
