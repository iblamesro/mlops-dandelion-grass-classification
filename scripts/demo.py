#!/usr/bin/env python3
"""
Quick demo script to test the trained model
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import sys


def create_model(num_classes=2):
    """Create ResNet18 model"""
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def load_model_from_checkpoint(model_path: str, device: torch.device):
    """Load model from checkpoint"""
    model = create_model(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"📊 Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"   Val Accuracy: {checkpoint.get('val_accuracy', 0):.4f}")
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def get_transforms():
    """Get image preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def predict_image(image_path: str, model, device, transform):
    """Predict class of an image"""
    # Load image
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"✅ Loaded image: {image_path}")
        print(f"   Size: {image.size}, Mode: {image.mode}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None
    
    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    class_names = ["Dandelion", "Grass"]
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item()
    
    probs = probabilities[0].cpu().numpy()
    
    return {
        "class": predicted_class,
        "confidence": confidence_score,
        "probabilities": {
            "Dandelion": probs[0],
            "Grass": probs[1]
        }
    }


def main():
    """Main demo function"""
    print("="*60)
    print("🌼 Dandelion vs Grass Classifier - Demo")
    print("="*60)
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple Silicon GPU (MPS)"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = f"CUDA GPU ({torch.cuda.get_device_name(0)})"
    else:
        device = torch.device("cpu")
        device_name = "CPU"
    
    print(f"\n💻 Device: {device_name}")
    
    # Load model
    model_path = Path("models/best_model.pth")
    
    if not model_path.exists():
        print(f"\n❌ Model not found at {model_path}")
        print("💡 Please train the model first: python3 scripts/train_advanced.py")
        return
    
    print(f"\n🧠 Loading model from {model_path}...")
    model = load_model_from_checkpoint(str(model_path), device)
    print("✅ Model loaded successfully!")
    
    # Get transforms
    transform = get_transforms()
    
    # Test with images from data directory
    print("\n" + "="*60)
    print("📸 Testing with sample images")
    print("="*60)
    
    data_dir = Path("data")
    test_images = []
    
    # Get 3 dandelion images
    dandelion_dir = data_dir / "dandelion"
    if dandelion_dir.exists():
        dandelion_images = list(dandelion_dir.glob("*.jpg"))[:3]
        test_images.extend([(img, "Dandelion") for img in dandelion_images])
    
    # Get 3 grass images
    grass_dir = data_dir / "grass"
    if grass_dir.exists():
        grass_images = list(grass_dir.glob("*.jpg"))[:3]
        test_images.extend([(img, "Grass") for img in grass_images])
    
    if not test_images:
        print("\n⚠️  No test images found in data/ directory")
        
        # Check if user provided image path
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
            if Path(image_path).exists():
                test_images = [(Path(image_path), "Unknown")]
            else:
                print(f"❌ Image not found: {image_path}")
                return
        else:
            print("💡 Usage: python3 scripts/demo.py [image_path]")
            return
    
    # Run predictions
    correct = 0
    total = 0
    
    for img_path, true_label in test_images:
        print(f"\n{'─'*60}")
        print(f"📷 Image: {img_path.name}")
        print(f"🏷️  True Label: {true_label}")
        print(f"{'─'*60}")
        
        result = predict_image(str(img_path), model, device, transform)
        
        if result:
            emoji = "🌼" if result["class"] == "Dandelion" else "🌿"
            print(f"\n🎯 Prediction: {emoji} {result['class']}")
            print(f"📊 Confidence: {result['confidence']*100:.2f}%")
            print(f"\n   Probabilities:")
            print(f"   🌼 Dandelion: {result['probabilities']['Dandelion']*100:.2f}%")
            print(f"   🌿 Grass:     {result['probabilities']['Grass']*100:.2f}%")
            
            if result["class"] == true_label:
                print(f"\n✅ CORRECT!")
                correct += 1
            else:
                print(f"\n❌ INCORRECT (expected {true_label})")
            
            total += 1
    
    # Summary
    if total > 0:
        accuracy = correct / total * 100
        print(f"\n{'='*60}")
        print(f"📊 SUMMARY")
        print(f"{'='*60}")
        print(f"Total predictions: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
