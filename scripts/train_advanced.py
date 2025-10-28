"""
Advanced training script with data augmentation and validation
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import random
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import mlflow
import mlflow.pytorch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(num_classes=2, pretrained=True):
    """Create ResNet18 model"""
    model = models.resnet18(pretrained=pretrained)
    # Modify final layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


class LocalPlantDataset(Dataset):
    """Dataset for loading images from local directory"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return blank image on error
            image = Image.new('RGB', (224, 224), color='gray')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label


def get_transforms(augment=True):
    """Get image transforms with optional data augmentation"""
    
    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def load_data_from_directory(data_dir):
    """Load images and labels from directory structure"""
    data_dir = Path(data_dir)
    
    image_paths = []
    labels = []
    
    # Load dandelion images (label=0)
    dandelion_dir = data_dir / "dandelion"
    if dandelion_dir.exists():
        for img_path in dandelion_dir.glob("*.jpg"):
            image_paths.append(str(img_path))
            labels.append(0)
    
    # Load grass images (label=1)
    grass_dir = data_dir / "grass"
    if grass_dir.exists():
        for img_path in grass_dir.glob("*.jpg"):
            image_paths.append(str(img_path))
            labels.append(1)
    
    print(f"📊 Loaded {len(image_paths)} images total")
    print(f"   - Dandelions: {labels.count(0)}")
    print(f"   - Grass: {labels.count(1)}")
    
    return image_paths, labels


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(dataloader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_precision = precision_score(all_labels, all_preds, average='binary')
    val_recall = recall_score(all_labels, all_preds, average='binary')
    val_f1 = f1_score(all_labels, all_preds, average='binary')
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'loss': val_loss,
        'accuracy': val_acc,
        'precision': val_precision,
        'recall': val_recall,
        'f1': val_f1,
        'confusion_matrix': cm
    }


def train_model(
    data_dir="data",
    output_dir="models",
    epochs=20,
    batch_size=32,
    learning_rate=0.001,
    test_size=0.2,
    val_size=0.1,
    early_stopping_patience=5
):
    """
    Main training function with validation and early stopping
    
    Args:
        data_dir: Directory containing images
        output_dir: Directory to save models
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        test_size: Proportion of data for testing
        val_size: Proportion of data for validation (from train set)
        early_stopping_patience: Epochs to wait before early stopping
    """
    
    print("🚀 Starting training pipeline...")
    set_seed(42)
    
    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Using device: {device}")
    
    # Load data
    print("\n📂 Loading data...")
    image_paths, labels = load_data_from_directory(data_dir)
    
    if len(image_paths) == 0:
        print("❌ No images found! Please run download script first.")
        return
    
    # Split data: train/test split first
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Then split train into train/val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=val_size, random_state=42, stratify=train_labels
    )
    
    print(f"\n📊 Data split:")
    print(f"   - Training: {len(train_paths)} images")
    print(f"   - Validation: {len(val_paths)} images")
    print(f"   - Test: {len(test_paths)} images")
    
    # Create datasets
    train_dataset = LocalPlantDataset(train_paths, train_labels, get_transforms(augment=True))
    val_dataset = LocalPlantDataset(val_paths, val_labels, get_transforms(augment=False))
    test_dataset = LocalPlantDataset(test_paths, test_labels, get_transforms(augment=False))
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Create model
    print("\n🧠 Creating model...")
    model = create_model(num_classes=2, pretrained=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # MLflow tracking
    mlflow.set_experiment("dandelion-grass-classification")
    
    with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("model", "ResNet18")
        mlflow.log_param("train_size", len(train_paths))
        mlflow.log_param("val_size", len(val_paths))
        mlflow.log_param("test_size", len(test_paths))
        mlflow.log_param("data_augmentation", "yes")
        
        # Training loop
        print("\n🏋️ Training model...\n")
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs}")
            print('='*60)
            
            # Train
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            print(f"📈 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            
            # Validate
            val_metrics = validate(model, val_loader, criterion, device)
            print(f"📊 Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"   Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")
            
            # Log metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_metrics['loss'], step=epoch)
            mlflow.log_metric("val_accuracy", val_metrics['accuracy'], step=epoch)
            mlflow.log_metric("val_precision", val_metrics['precision'], step=epoch)
            mlflow.log_metric("val_recall", val_metrics['recall'], step=epoch)
            mlflow.log_metric("val_f1", val_metrics['f1'], step=epoch)
            
            # Learning rate scheduler
            scheduler.step(val_metrics['loss'])
            
            # Early stopping and model checkpointing
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                patience_counter = 0
                
                # Save best model
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                model_path = output_path / "best_model.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': best_val_acc,
                }, model_path)
                print(f"💾 Saved best model (val_acc: {best_val_acc:.4f})")
                
            else:
                patience_counter += 1
                print(f"⏳ Early stopping patience: {patience_counter}/{early_stopping_patience}")
                
                if patience_counter >= early_stopping_patience:
                    print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Load best model for final evaluation
        print("\n" + "="*60)
        print("📊 Final Evaluation on Test Set")
        print("="*60)
        
        checkpoint = torch.load(output_path / "best_model.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_metrics = validate(model, test_loader, criterion, device)
        print(f"\n🎯 Test Results:")
        print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"   Precision: {test_metrics['precision']:.4f}")
        print(f"   Recall: {test_metrics['recall']:.4f}")
        print(f"   F1 Score: {test_metrics['f1']:.4f}")
        print(f"\n   Confusion Matrix:")
        print(f"   {test_metrics['confusion_matrix']}")
        
        # Log final test metrics
        mlflow.log_metric("test_accuracy", test_metrics['accuracy'])
        mlflow.log_metric("test_precision", test_metrics['precision'])
        mlflow.log_metric("test_recall", test_metrics['recall'])
        mlflow.log_metric("test_f1", test_metrics['f1'])
        
        # Log model
        mlflow.pytorch.log_model(model, "model")
        
        print("\n✅ Training completed successfully!")
        print(f"📁 Best model saved to: {model_path}")
        print(f"🎯 Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    train_model(
        data_dir="data",
        output_dir="models",
        epochs=20,
        batch_size=32,
        learning_rate=0.001,
        test_size=0.2,
        val_size=0.1,
        early_stopping_patience=5
    )
