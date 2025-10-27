# Locust Load Testing Script

from locust import HttpUser, task, between
import random
import io
from PIL import Image


class APIUser(HttpUser):
    """
    Load testing for the prediction API
    """
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a user starts"""
        # Check if API is healthy
        response = self.client.get("/health")
        if response.status_code != 200:
            print("⚠️ API is not healthy!")
    
    @task(3)
    def predict_image(self):
        """Test prediction endpoint"""
        # Create a random test image
        img = Image.new('RGB', (224, 224), color=(
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        ))
        
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Send prediction request
        files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
        
        with self.client.post(
            "/predict",
            files=files,
            catch_response=True,
            name="/predict"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if 'predicted_class' in data and 'confidence' in data:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Test health endpoint"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def model_info(self):
        """Test model info endpoint"""
        self.client.get("/model/info", name="/model/info")


# To run:
# locust -f scripts/load_test.py --host http://localhost:8000
# Then open http://localhost:8089 in browser
