"""
ONNX export script for converting sklearn models to ONNX format.
"""
import os
import sys
from pathlib import Path
import joblib
import numpy as np
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Add the app directory to Python path
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

# Configure logging
configure_logging()
logger = get_logger("export_onnx")


def load_sklearn_model(model_path: str):
    """Load sklearn model from file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    logger.info(f"Loaded sklearn model: {type(model).__name__}")
    return model


def export_to_onnx(model, onnx_path: str, n_features: int = 4):
    """Export sklearn model to ONNX format."""
    # Define input type
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    
    # Convert model
    logger.info("Converting model to ONNX...")
    onx = convert_sklearn(model, initial_types=initial_type)
    
    # Save ONNX model
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    with open(onnx_path, "wb") as f:
        f.write(onx.SerializeToString())
    
    logger.info(f"ONNX model saved to: {onnx_path}")
    return onx


def validate_onnx_model(onnx_path: str, sklearn_model, test_data: np.ndarray):
    """Validate ONNX model against sklearn model."""
    logger.info("Validating ONNX model...")
    
    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_path)
    
    # Get input/output names
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    
    # Make predictions with sklearn
    sklearn_pred = sklearn_model.predict(test_data)
    
    # Make predictions with ONNX
    onnx_pred = ort_session.run([output_name], {input_name: test_data.astype(np.float32)})[0]
    
    # Compare predictions
    if np.array_equal(sklearn_pred, onnx_pred):
        logger.info("✅ ONNX model validation successful - predictions match!")
        return True
    else:
        logger.error("❌ ONNX model validation failed - predictions don't match!")
        logger.error(f"Sklearn predictions: {sklearn_pred}")
        logger.error(f"ONNX predictions: {onnx_pred}")
        return False


def benchmark_models(sklearn_model, onnx_path: str, test_data: np.ndarray, n_runs: int = 100):
    """Benchmark sklearn vs ONNX performance."""
    logger.info(f"Benchmarking models with {n_runs} runs...")
    
    # Benchmark sklearn
    import time
    
    # Warmup
    for _ in range(10):
        sklearn_model.predict(test_data)
    
    # Benchmark sklearn
    start_time = time.time()
    for _ in range(n_runs):
        sklearn_model.predict(test_data)
    sklearn_time = (time.time() - start_time) / n_runs
    
    # Benchmark ONNX
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    
    # Warmup
    for _ in range(10):
        ort_session.run([output_name], {input_name: test_data.astype(np.float32)})
    
    # Benchmark ONNX
    start_time = time.time()
    for _ in range(n_runs):
        ort_session.run([output_name], {input_name: test_data.astype(np.float32)})
    onnx_time = (time.time() - start_time) / n_runs
    
    # Calculate speedup
    speedup = sklearn_time / onnx_time
    
    logger.info(f"Sklearn average time: {sklearn_time*1000:.3f} ms")
    logger.info(f"ONNX average time: {onnx_time*1000:.3f} ms")
    logger.info(f"ONNX speedup: {speedup:.2f}x")
    
    return {
        "sklearn_time_ms": sklearn_time * 1000,
        "onnx_time_ms": onnx_time * 1000,
        "speedup": speedup
    }


def main():
    """Main export function."""
    try:
        # Get settings
        settings = get_settings()
        
        logger.info("Starting ONNX export...")
        
        # Load sklearn model
        sklearn_model = load_sklearn_model(settings.MODEL_PATH)
        
        # Determine number of features
        if hasattr(sklearn_model, 'n_features_in_'):
            n_features = sklearn_model.n_features_in_
        else:
            # Default for iris dataset
            n_features = 4
        
        logger.info(f"Model has {n_features} features")
        
        # Export to ONNX
        onx = export_to_onnx(sklearn_model, settings.ONNX_PATH, n_features)
        
        # Generate test data
        test_data = np.random.random((10, n_features))
        
        # Validate ONNX model
        validation_success = validate_onnx_model(settings.ONNX_PATH, sklearn_model, test_data)
        
        if not validation_success:
            logger.error("ONNX export failed validation!")
            sys.exit(1)
        
        # Benchmark models
        benchmark_results = benchmark_models(sklearn_model, settings.ONNX_PATH, test_data)
        
        logger.info("ONNX export completed successfully!")
        
        # Print results
        print("\n" + "="*50)
        print("ONNX EXPORT COMPLETED")
        print("="*50)
        print(f"ONNX model saved to: {settings.ONNX_PATH}")
        print(f"Model features: {n_features}")
        print(f"Validation: {'✅ PASSED' if validation_success else '❌ FAILED'}")
        print(f"Sklearn time: {benchmark_results['sklearn_time_ms']:.3f} ms")
        print(f"ONNX time: {benchmark_results['onnx_time_ms']:.3f} ms")
        print(f"Speedup: {benchmark_results['speedup']:.2f}x")
        print("="*50)
        
    except Exception as e:
        logger.error(f"ONNX export failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
