#!/usr/bin/env python3
"""
Download and load model from MinIO (S3)
"""

from pathlib import Path
from minio import Minio
from minio.error import S3Error
import json
import torch
import torch.nn as nn
from torchvision import models


# MinIO Configuration
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
BUCKET_NAME = "mlops-models"


def create_minio_client():
    """Create MinIO client"""
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )


def list_available_models(client, bucket_name):
    """List all available models in the bucket"""
    try:
        objects = client.list_objects(bucket_name, prefix="models/", recursive=True)
        
        models_list = []
        for obj in objects:
            if obj.object_name.endswith('.pth'):
                models_list.append({
                    'name': obj.object_name,
                    'size_mb': obj.size / (1024 * 1024),
                    'last_modified': obj.last_modified
                })
        
        return models_list
    except S3Error as e:
        print(f"❌ Error listing models: {e}")
        return []


def download_metadata(client, bucket_name):
    """Download model metadata"""
    try:
        metadata_object = "models/model_metadata.json"
        response = client.get_object(bucket_name, metadata_object)
        metadata = json.loads(response.read())
        response.close()
        response.release_conn()
        
        return metadata
    except S3Error as e:
        print(f"⚠️  Could not download metadata: {e}")
        return None


def download_model(client, bucket_name, object_name, local_path):
    """Download model from MinIO"""
    try:
        print(f"📥 Downloading model from S3...")
        print(f"   Source: s3://{bucket_name}/{object_name}")
        print(f"   Destination: {local_path}")
        
        client.fget_object(bucket_name, object_name, str(local_path))
        
        file_size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model downloaded successfully! ({file_size_mb:.2f} MB)")
        
        return True
    except S3Error as e:
        print(f"❌ Error downloading model: {e}")
        return False


def create_model(num_classes=2):
    """Create ResNet18 model"""
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def load_model_from_checkpoint(model_path, device):
    """Load model from checkpoint"""
    model = create_model(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def main():
    """Main function"""
    print("="*60)
    print("📥 Download Model from MinIO (S3)")
    print("="*60)
    
    # Connect to MinIO
    print(f"\n🔌 Connecting to MinIO at {MINIO_ENDPOINT}...")
    try:
        client = create_minio_client()
        print("✅ Connected to MinIO")
    except Exception as e:
        print(f"❌ Failed to connect to MinIO: {e}")
        return
    
    # Download metadata
    print("\n📋 Fetching model metadata...")
    metadata = download_metadata(client, BUCKET_NAME)
    if metadata:
        print(f"✅ Metadata retrieved:")
        print(f"   Model: {metadata.get('model_name', 'Unknown')}")
        print(f"   Architecture: {metadata.get('architecture', 'Unknown')}")
        print(f"   Version: {metadata.get('version', 'Unknown')}")
        print(f"   Validation Accuracy: {metadata.get('metrics', {}).get('validation_accuracy', 0)*100:.2f}%")
    
    # List available models
    print(f"\n📦 Available models in bucket '{BUCKET_NAME}':")
    models_list = list_available_models(client, BUCKET_NAME)
    
    if not models_list:
        print("❌ No models found in bucket!")
        return
    
    print("="*60)
    for idx, model_info in enumerate(models_list, 1):
        print(f"{idx}. {model_info['name']}")
        print(f"   Size: {model_info['size_mb']:.2f} MB")
        print(f"   Last Modified: {model_info['last_modified']}")
        print()
    print("="*60)
    
    # Download the most recent model
    model_to_download = models_list[-1]['name']  # Most recent
    print(f"\n🎯 Downloading latest model: {model_to_download}")
    
    # Create download directory
    download_dir = Path("models/downloaded")
    download_dir.mkdir(parents=True, exist_ok=True)
    
    local_path = download_dir / "model_from_s3.pth"
    
    # Download
    if not download_model(client, BUCKET_NAME, model_to_download, local_path):
        return
    
    # Verify and load model
    print("\n🧠 Loading model to verify it works...")
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = load_model_from_checkpoint(str(local_path), device)
        print(f"✅ Model loaded successfully on {device}")
        
        # Test with dummy input
        test_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(test_input)
            probabilities = torch.softmax(output, dim=1)
            predicted = torch.argmax(probabilities, dim=1)
        
        print(f"✅ Model inference test passed!")
        print(f"   Output shape: {output.shape}")
        print(f"   Prediction: Class {predicted.item()}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Summary
    print("\n" + "="*60)
    print("✅ DOWNLOAD AND VERIFICATION COMPLETE!")
    print("="*60)
    print(f"📁 Model saved to: {local_path}")
    print(f"🎯 Ready to use for inference!")
    print("="*60)


if __name__ == "__main__":
    main()
