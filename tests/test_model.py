from pathlib import Path
import joblib


# Build path to model file (two levels up from current file)
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "models"
    / "churn_model.pkl"
)


def test_model_file_exists():
    # Check if model file actually exists on disk
    assert MODEL_PATH.exists()


def test_model_loads():
    # Load model from file
    model = joblib.load(MODEL_PATH)

    # Verify model loaded and has predict method
    assert model is not None
    assert hasattr(model, "predict")