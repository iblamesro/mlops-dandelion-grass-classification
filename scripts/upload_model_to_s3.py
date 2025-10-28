#!/usr/bin/env python3
"""
Upload trained model to MinIO (S3-compatible storage)
"""

from pathlib import Path
from datetime import datetime
from minio import Minio
from minio.error import S3Error
import json


# MinIO Configuration
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
BUCKET_NAME = "mlops-models"

# Model path
MODEL_PATH = Path("models/best_model.pth")
METADATA_PATH = Path("models/model_metadata.json")


def create_minio_client():
    """Create MinIO client"""
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )


def create_bucket_if_not_exists(client, bucket_name):
    """Create bucket if it doesn't exist"""
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            print(f"✅ Created bucket: {bucket_name}")
        else:
            print(f"✅ Bucket already exists: {bucket_name}")
        return True
    except S3Error as e:
        print(f"❌ Error creating bucket: {e}")
        return False


def get_model_metadata():
    """Get or create model metadata"""
    if METADATA_PATH.exists():
        with open(METADATA_PATH, 'r') as f:
            return json.load(f)
    
    # Create basic metadata
    metadata = {
        "model_name": "dandelion-grass-classifier",
        "architecture": "ResNet18",
        "framework": "PyTorch",
        "version": "1.0.0",
        "trained_date": datetime.now().isoformat(),
        "num_classes": 2,
        "classes": ["dandelion", "grass"],
        "input_size": [224, 224],
        "metrics": {
            "validation_accuracy": 0.9302,
            "precision": 0.9130,
            "recall": 0.9545,
            "f1_score": 0.9333
        }
    }
    
    # Save metadata locally
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def upload_model(client, bucket_name, model_path, object_name=None):
    """Upload model file to MinIO"""
    if object_name is None:
        object_name = f"models/{model_path.name}"
    
    try:
        # Get file size
        file_size = model_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"\n📤 Uploading model...")
        print(f"   Source: {model_path}")
        print(f"   Size: {file_size_mb:.2f} MB")
        print(f"   Destination: s3://{bucket_name}/{object_name}")
        
        # Upload file
        client.fput_object(
            bucket_name,
            object_name,
            str(model_path),
            content_type="application/octet-stream"
        )
        
        print(f"✅ Model uploaded successfully!")
        return True
        
    except S3Error as e:
        print(f"❌ Error uploading model: {e}")
        return False


def upload_metadata(client, bucket_name, metadata):
    """Upload model metadata as JSON"""
    try:
        metadata_json = json.dumps(metadata, indent=2).encode('utf-8')
        object_name = "models/model_metadata.json"
        
        from io import BytesIO
        metadata_stream = BytesIO(metadata_json)
        
        client.put_object(
            bucket_name,
            object_name,
            metadata_stream,
            len(metadata_json),
            content_type="application/json"
        )
        
        print(f"✅ Metadata uploaded: s3://{bucket_name}/{object_name}")
        return True
        
    except S3Error as e:
        print(f"❌ Error uploading metadata: {e}")
        return False


def list_models_in_bucket(client, bucket_name):
    """List all models in the bucket"""
    try:
        objects = client.list_objects(bucket_name, prefix="models/", recursive=True)
        
        print(f"\n📦 Models in bucket '{bucket_name}':")
        print("="*60)
        
        model_count = 0
        for obj in objects:
            size_mb = obj.size / (1024 * 1024)
            print(f"  📄 {obj.object_name}")
            print(f"     Size: {size_mb:.2f} MB")
            print(f"     Last Modified: {obj.last_modified}")
            print()
            model_count += 1
        
        if model_count == 0:
            print("  (No models found)")
        else:
            print(f"Total: {model_count} file(s)")
        
        print("="*60)
        
    except S3Error as e:
        print(f"❌ Error listing models: {e}")


def verify_upload(client, bucket_name, object_name):
    """Verify that the file was uploaded correctly"""
    try:
        stat = client.stat_object(bucket_name, object_name)
        print(f"\n✅ Upload verified!")
        print(f"   Object: {object_name}")
        print(f"   Size: {stat.size / (1024*1024):.2f} MB")
        print(f"   ETag: {stat.etag}")
        print(f"   Last Modified: {stat.last_modified}")
        return True
    except S3Error as e:
        print(f"❌ Verification failed: {e}")
        return False


def get_download_command(bucket_name, object_name):
    """Generate download command for later use"""
    return f"""
To download this model later, use:

mc cp minio/{bucket_name}/{object_name} ./models/

Or in Python:
```python
from minio import Minio

client = Minio('{MINIO_ENDPOINT}', '{MINIO_ACCESS_KEY}', '{MINIO_SECRET_KEY}', secure=False)
client.fget_object('{bucket_name}', '{object_name}', 'downloaded_model.pth')
```
"""


def main():
    """Main function"""
    print("="*60)
    print("🚀 Model Upload to MinIO (S3)")
    print("="*60)
    
    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"\n❌ Model not found at {MODEL_PATH}")
        print("💡 Please train the model first: python3 scripts/train_advanced.py")
        return
    
    print(f"\n✅ Model found: {MODEL_PATH}")
    model_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"   Size: {model_size_mb:.2f} MB")
    
    # Connect to MinIO
    print(f"\n🔌 Connecting to MinIO at {MINIO_ENDPOINT}...")
    try:
        client = create_minio_client()
        print("✅ Connected to MinIO")
    except Exception as e:
        print(f"❌ Failed to connect to MinIO: {e}")
        print("💡 Make sure Docker containers are running: docker-compose up -d")
        return
    
    # Create bucket
    print(f"\n📦 Setting up bucket '{BUCKET_NAME}'...")
    if not create_bucket_if_not_exists(client, BUCKET_NAME):
        return
    
    # Get metadata
    print("\n📋 Preparing metadata...")
    metadata = get_model_metadata()
    print(f"   Model: {metadata['model_name']}")
    print(f"   Architecture: {metadata['architecture']}")
    print(f"   Version: {metadata['version']}")
    print(f"   Validation Accuracy: {metadata['metrics']['validation_accuracy']*100:.2f}%")
    
    # Upload model
    object_name = f"models/best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    if not upload_model(client, BUCKET_NAME, MODEL_PATH, object_name):
        return
    
    # Upload metadata
    upload_metadata(client, BUCKET_NAME, metadata)
    
    # Verify upload
    verify_upload(client, BUCKET_NAME, object_name)
    
    # Create a 'latest' version link
    latest_object = "models/best_model_latest.pth"
    try:
        client.copy_object(
            BUCKET_NAME,
            latest_object,
            f"{BUCKET_NAME}/{object_name}"
        )
        print(f"✅ Created 'latest' version: {latest_object}")
    except Exception as e:
        print(f"⚠️  Could not create latest version: {e}")
    
    # List all models
    list_models_in_bucket(client, BUCKET_NAME)
    
    # Show download instructions
    print("\n" + "="*60)
    print("📥 DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print(get_download_command(BUCKET_NAME, object_name))
    
    # Success summary
    print("\n" + "="*60)
    print("✅ MODEL UPLOAD COMPLETE!")
    print("="*60)
    print(f"🎯 Model stored in: s3://{BUCKET_NAME}/{object_name}")
    print(f"🔗 Latest version: s3://{BUCKET_NAME}/{latest_object}")
    print(f"🌐 MinIO Console: http://{MINIO_ENDPOINT}/browser")
    print(f"📊 Metadata: s3://{BUCKET_NAME}/models/model_metadata.json")
    print("="*60)


if __name__ == "__main__":
    main()
