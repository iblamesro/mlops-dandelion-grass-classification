"""
Simple test script for the Plant Classification API
Tests the API endpoints with sample images
"""

import requests
import json
from pathlib import Path
import time

# API URL
API_URL = "http://localhost:8000"

def wait_for_api(max_retries=30):
    """Wait for API to be ready"""
    print("⏳ Waiting for API to start...")
    for i in range(max_retries):
        try:
            response = requests.get(f"{API_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✅ API is ready!")
                return True
        except:
            time.sleep(1)
            if (i + 1) % 5 == 0:
                print(f"   Still waiting... ({i+1}/{max_retries})")
    print("❌ API did not start in time")
    return False

def test_endpoints():
    """Test basic API endpoints"""
    print("\n" + "=" * 80)
    print("🧪 TESTING API ENDPOINTS")
    print("=" * 80)
    
    # Test root endpoint
    print("\n1️⃣  Testing GET /")
    try:
        response = requests.get(f"{API_URL}/")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test health endpoint
    print("\n2️⃣  Testing GET /health")
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test model info
    print("\n3️⃣  Testing GET /model/info")
    try:
        response = requests.get(f"{API_URL}/model/info")
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Model: {data.get('model_name')}")
        print(f"   Classes: {data.get('classes')}")
        print(f"   Device: {data.get('device')}")
        print(f"   Parameters: {data.get('num_parameters'):,}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

def test_prediction(image_path: str, expected_label: str = None):
    """Test prediction with an image"""
    print(f"\n4️⃣  Testing POST /predict with image: {Path(image_path).name}")
    
    if not Path(image_path).exists():
        print(f"   ⚠️  Image not found: {image_path}")
        return
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": (Path(image_path).name, f, "image/jpeg")}
            response = requests.post(f"{API_URL}/predict", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Status: {response.status_code}")
            print(f"   Predicted: {result['predicted_class']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Time: {result['prediction_time']:.3f}s")
            
            if expected_label and result['predicted_class'] == expected_label:
                print(f"   ✅ Correct prediction!")
            elif expected_label:
                print(f"   ⚠️  Expected {expected_label}")
        else:
            print(f"   ❌ Error: Status {response.status_code}")
            print(f"   {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

def main():
    """Main test function"""
    print("\n" + "🌿 " * 30)
    print("PLANT CLASSIFICATION API - TEST SUITE")
    print("🌿 " * 30)
    
    # Wait for API
    if not wait_for_api():
        print("\n❌ API is not running. Please start it first:")
        print("   cd /Users/sara/Desktop/AlbertSchool/MLOps/MLproject")
        print("   python3 run_api.py")
        return
    
    # Test endpoints
    test_endpoints()
    
    # Test predictions with sample images
    print("\n" + "=" * 80)
    print("🖼️  TESTING PREDICTIONS")
    print("=" * 80)
    
    # Find sample images
    data_dir = Path("data")
    if data_dir.exists():
        # Test dandelion
        dandelion_images = list((data_dir / "dandelion").glob("*.jpg"))
        if dandelion_images:
            test_prediction(str(dandelion_images[0]), "dandelion")
        
        # Test grass
        grass_images = list((data_dir / "grass").glob("*.jpg"))
        if grass_images:
            test_prediction(str(grass_images[0]), "grass")
    else:
        print("   ⚠️  No test images found in data/ directory")
    
    print("\n" + "=" * 80)
    print("✅ TESTING COMPLETE!")
    print("=" * 80)
    print(f"\n📚 View Swagger documentation at: {API_URL}/docs")
    print(f"📖 View ReDoc documentation at: {API_URL}/redoc")
    print("\n")

if __name__ == "__main__":
    main()
