"""
Integration tests for API
"""

import pytest
import io
from PIL import Image
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def create_test_image():
    """Create a test image"""
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_model_info_endpoint():
    """Test model info endpoint"""
    response = client.get("/model/info")
    assert response.status_code in [200, 503]  # 503 if model not loaded
    
    if response.status_code == 200:
        data = response.json()
        assert "model_name" in data
        assert "classes" in data
        assert len(data["classes"]) == 2


def test_predict_endpoint():
    """Test prediction endpoint"""
    img_bytes = create_test_image()
    
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    
    # Should be 200 or 503 (if model not loaded)
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "predicted_class" in data
        assert "confidence" in data
        assert "prediction_time" in data
        assert data["predicted_class"] in ["dandelion", "grass"]
        assert 0 <= data["confidence"] <= 1


def test_predict_invalid_file():
    """Test prediction with invalid file"""
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    
    assert response.status_code in [400, 503]
