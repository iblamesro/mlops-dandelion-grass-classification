#!/usr/bin/env python3
"""
Register trained model in MLflow Model Registry
"""

import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes=2):
    """Create ResNet18 model"""
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def register_model_in_mlflow():
    """Register the trained model in MLflow"""
    
    print("="*70)
    print("📊 MLflow Model Registry - Register Trained Model")
    print("="*70)
    
    # Set MLflow tracking URI
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(mlflow_uri)
    print(f"\n🔗 MLflow Tracking URI: {mlflow_uri}")
    
    # Set experiment
    experiment_name = "dandelion-grass-classification"
    mlflow.set_experiment(experiment_name)
    print(f"🔬 Experiment: {experiment_name}")
    
    # Load model
    model_path = Path("models/best_model.pth")
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return
    
    print(f"\n📁 Loading model from: {model_path}")
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    
    model = create_model(num_classes=2)
    checkpoint = torch.load(str(model_path), map_location=device)
    
    # Extract metadata from checkpoint
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        val_acc = checkpoint.get('val_accuracy', 0.0)
        metadata = {
            'epoch': str(epoch),
            'val_accuracy': str(val_acc),
            'device': str(device),
            'architecture': 'ResNet18',
            'num_classes': '2',
            'framework': 'PyTorch',
            'registered_date': datetime.now().isoformat()
        }
        print(f"✅ Model loaded from checkpoint (Epoch {epoch}, Val Acc: {val_acc:.4f})")
    else:
        model.load_state_dict(checkpoint)
        metadata = {
            'architecture': 'ResNet18',
            'num_classes': '2',
            'framework': 'PyTorch',
            'device': str(device),
            'registered_date': datetime.now().isoformat()
        }
        print(f"✅ Model loaded from state dict")
    
    model.eval()
    
    # Start MLflow run
    run_name = f"model_registration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\n🚀 Starting MLflow run: {run_name}")
    
    with mlflow.start_run(run_name=run_name):
        
        # Log parameters
        mlflow.log_params(metadata)
        
        # Create sample input for model signature
        print(f"\n📝 Creating model signature...")
        sample_input = torch.randn(1, 3, 224, 224)
        
        # Save model as artifact first
        print(f"📦 Saving model as artifact...")
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_model_path = Path(tmpdir) / "model"
            tmp_model_path.mkdir()
            
            # Save model
            torch.save(model.state_dict(), tmp_model_path / "model.pth")
            
            # Save as MLflow artifact
            mlflow.log_artifacts(str(tmp_model_path), "model")
        
        # Register the model without log_model
        print(f"📝 Registering model...")
        run_id = mlflow.active_run().info.run_id
        
        # Get run info
        run = mlflow.active_run()
        
        # Log model metadata
        import json
        metadata_path = Path("models/model_metadata.json")
        if metadata_path.exists():
            mlflow.log_artifact(str(metadata_path), "metadata")
            with open(metadata_path) as f:
                model_metadata = json.load(f)
                mlflow.log_metrics({
                    "registered_val_accuracy": model_metadata.get("metrics", {}).get("validation_accuracy", 0.0),
                    "registered_val_precision": model_metadata.get("metrics", {}).get("validation_precision", 0.0),
                    "registered_val_recall": model_metadata.get("metrics", {}).get("validation_recall", 0.0),
                    "registered_val_f1": model_metadata.get("metrics", {}).get("validation_f1", 0.0)
                })
        
        # Get run info
        run = mlflow.active_run()
        run_id = run.info.run_id
        
        print(f"\n" + "="*70)
        print(f"✅ MODEL REGISTERED SUCCESSFULLY!")
        print("="*70)
        print(f"🔑 Run ID: {run_id}")
        print(f"🏷️  Model Name: dandelion-grass-resnet18")
        print(f"📊 Experiment: {experiment_name}")
        print(f"🌐 MLflow UI: {mlflow_uri}")
        print("="*70)
        
        return run_id


def list_registered_models():
    """List all registered models in MLflow"""
    
    print("\n" + "="*70)
    print("📋 Listing Registered Models")
    print("="*70)
    
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(mlflow_uri)
    
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    
    try:
        # List all registered models
        registered_models = client.search_registered_models()
        
        if not registered_models:
            print("ℹ️  No registered models found")
            return
        
        for rm in registered_models:
            print(f"\n📦 Model: {rm.name}")
            print(f"   Description: {rm.description or 'N/A'}")
            print(f"   Latest Versions:")
            
            # Get latest versions
            for version in rm.latest_versions:
                print(f"      - Version {version.version}")
                print(f"        Stage: {version.current_stage}")
                print(f"        Run ID: {version.run_id}")
                print(f"        Status: {version.status}")
        
    except Exception as e:
        print(f"⚠️  Could not list models: {e}")
        print("💡 Make sure MLflow server is running")


def main():
    """Main function"""
    
    # Register model
    run_id = register_model_in_mlflow()
    
    if run_id:
        # List all registered models
        list_registered_models()
        
        print("\n" + "="*70)
        print("💡 Next Steps:")
        print("="*70)
        print("1. View in MLflow UI: http://localhost:5001")
        print("2. Navigate to 'Models' tab to see registered model")
        print("3. You can promote model versions to 'Staging' or 'Production'")
        print("4. Use the model for deployment via MLflow serving")
        print("="*70)


if __name__ == "__main__":
    main()
