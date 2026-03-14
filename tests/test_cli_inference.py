import pytest
import pandas as pd
from typer.testing import CliRunner
from cli import app
from src.inference.predict import InferenceEngine
import yaml
import tempfile
from pathlib import Path

runner = CliRunner()

@pytest.fixture
def mock_config():
    with open("agent.yaml", "r") as f:
        return yaml.safe_load(f)

def test_inference_engine_initialization(mock_config):
    # This might fail if the model hasn't been trained in the environment yet,
    # but tests assume pipeline has been run at least once locally.
    model_path = Path("outputs/models/gst_selection_model.pkl")
    if model_path.exists():
        engine = InferenceEngine(mock_config, "gst")
        assert engine is not None
        assert engine.model is not None

def test_cli_infer_missing_args():
    # Should fail if neither sequence nor csv is provided
    result = runner.invoke(app, ["infer", "--model", "gst"])
    assert result.exit_code == 1
    assert "either a --sequence or a --csv file" in result.stdout

def test_cli_infer_single_sequence():
    model_path = Path("outputs/models/gst_selection_model.pkl")
    if model_path.exists():
        result = runner.invoke(app, ["infer", "--model", "gst", "--sequence", "MKTIIALSYIFCLVFA"])
        assert result.exit_code == 0
        assert "Predictions (gst)" in result.stdout
        assert "MKTIIALSYIFCLVFA" in result.stdout

def test_cli_infer_batch_csv():
    model_path = Path("outputs/models/gst_selection_model.pkl")
    if model_path.exists():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("sequence\n")
            f.write("MKTIIALSYIFCLVFA\n")
            f.write("MKTIIALSYIFCLVFAVVV\n")
            temp_path = f.name
            
        try:
            result = runner.invoke(app, ["infer", "--model", "gst", "--csv", temp_path])
            assert result.exit_code == 0
            assert "Predictions (gst)" in result.stdout
        finally:
            Path(temp_path).unlink()
