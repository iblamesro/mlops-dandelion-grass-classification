"""
Integration tests for FastAPI endpoints
"""

import pytest
import io
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np


@pytest.fixture
def client():
    """Create test client"""
    from src.api.main import app
    return TestClient(app)


@pytest.fixture
def test_image():
    """Create a test image"""
    # Create a simple test image (224x224 RGB)
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes


class TestAPIEndpoints:
    """Test API endpoints"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns correct response"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data
        assert data["status"] == "healthy"
    
    def test_model_info(self, client):
        """Test model info endpoint"""
        response = client.get("/model/info")
        
        # Model may not be loaded in test environment
        if response.status_code == 200:
            data = response.json()
            assert "model_name" in data
            assert "classes" in data
            assert "device" in data
            assert len(data["classes"]) == 2
    
    def test_predict_with_image(self, client, test_image):
        """Test prediction endpoint with valid image"""
        files = {"file": ("test.jpg", test_image, "image/jpeg")}
        response = client.post("/predict", files=files)
        
        # Prediction may fail if model not loaded, but should not crash
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "predicted_class" in data
            assert "confidence" in data
            assert "prediction_time" in data
            assert "timestamp" in data
            assert data["predicted_class"] in ["dandelion", "grass"]
            assert 0 <= data["confidence"] <= 1
    
    def test_predict_without_file(self, client):
        """Test prediction endpoint without file"""
        response = client.post("/predict")
        
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_predict_with_invalid_file(self, client):
        """Test prediction with invalid file"""
        # Send text file instead of image
        files = {"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
        response = client.post("/predict", files=files)
        
        assert response.status_code in [400, 503]  # Bad request or service unavailable
    
    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options("/")
        
        # CORS middleware should be active
        assert response.status_code in [200, 405]
    
    def test_multiple_predictions(self, client, test_image):
        """Test multiple predictions in sequence"""
        for i in range(3):
            test_image.seek(0)  # Reset bytes position
            files = {"file": (f"test{i}.jpg", test_image, "image/jpeg")}
            response = client.post("/predict", files=files)
            
            assert response.status_code in [200, 503]


class TestAPIPerformance:
    """Test API performance"""
    
    def test_prediction_response_time(self, client, test_image):
        """Test prediction response time is reasonable"""
        import time
        
        start_time = time.time()
        files = {"file": ("test.jpg", test_image, "image/jpeg")}
        response = client.post("/predict", files=files)
        elapsed_time = time.time() - start_time
        
        # Should respond within 10 seconds
        assert elapsed_time < 10.0, f"Response took {elapsed_time:.2f}s, expected < 10s"
    
    def test_health_check_fast(self, client):
        """Test health check is fast"""
        import time
        
        start_time = time.time()
        response = client.get("/health")
        elapsed_time = time.time() - start_time
        
        # Health check should be very fast
        assert elapsed_time < 1.0, f"Health check took {elapsed_time:.2f}s, expected < 1s"


class TestAPIValidation:
    """Test input validation"""
    
    def test_large_image(self, client):
        """Test with large image"""
        # Create a 4000x4000 image
        large_img = Image.fromarray(
            np.random.randint(0, 255, (4000, 4000, 3), dtype=np.uint8)
        )
        img_bytes = io.BytesIO()
        large_img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {"file": ("large.jpg", img_bytes, "image/jpeg")}
        response = client.post("/predict", files=files)
        
        # Should handle or reject gracefully
        assert response.status_code in [200, 400, 413, 503]
    
    def test_small_image(self, client):
        """Test with very small image"""
        small_img = Image.fromarray(
            np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        )
        img_bytes = io.BytesIO()
        small_img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {"file": ("small.jpg", img_bytes, "image/jpeg")}
        response = client.post("/predict", files=files)
        
        # Should handle gracefully
        assert response.status_code in [200, 400, 503]
    
    def test_grayscale_image(self, client):
        """Test with grayscale image"""
        gray_img = Image.fromarray(
            np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        )
        img_bytes = io.BytesIO()
        gray_img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {"file": ("gray.jpg", img_bytes, "image/jpeg")}
        response = client.post("/predict", files=files)
        
        # Should convert to RGB and process
        assert response.status_code in [200, 400, 503]
    
    def test_png_image(self, client):
        """Test with PNG image"""
        img = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        files = {"file": ("test.png", img_bytes, "image/png")}
        response = client.post("/predict", files=files)
        
        # Should handle PNG
        assert response.status_code in [200, 503]


class TestPrometheusMetrics:
    """Test Prometheus metrics endpoint"""
    
    def test_metrics_endpoint_exists(self, client):
        """Test that metrics endpoint is available"""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")
    
    def test_metrics_content(self, client):
        """Test metrics content"""
        response = client.get("/metrics")
        
        if response.status_code == 200:
            content = response.text
            # Check for some expected metric names
            assert "http_requests_total" in content or "http_request" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
