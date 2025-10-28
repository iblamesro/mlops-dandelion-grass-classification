#!/usr/bin/env python3
"""
Data visualization script
"""

import os
from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

DATA_DIR = Path("data")


def plot_sample_images(num_samples=10):
    """Plot sample images from each class"""
    
    dandelion_dir = DATA_DIR / "dandelion"
    grass_dir = DATA_DIR / "grass"
    
    # Get random samples
    dandelion_images = list(dandelion_dir.glob("*.jpg"))
    grass_images = list(grass_dir.glob("*.jpg"))
    
    dandelion_samples = random.sample(dandelion_images, min(num_samples, len(dandelion_images)))
    grass_samples = random.sample(grass_images, min(num_samples, len(grass_images)))
    
    # Create figure
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 4))
    fig.suptitle('Sample Images from Dataset', fontsize=16, fontweight='bold')
    
    # Plot dandelions
    for i, img_path in enumerate(dandelion_samples):
        img = Image.open(img_path)
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Dandelion', fontweight='bold')
    
    # Plot grass
    for i, img_path in enumerate(grass_samples):
        img = Image.open(img_path)
        axes[1, i].imshow(img)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Grass', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('data_samples.png', dpi=150, bbox_inches='tight')
    print("✅ Sample images saved to: data_samples.png")
    plt.close()


def plot_dataset_distribution():
    """Plot distribution of classes"""
    
    dandelion_dir = DATA_DIR / "dandelion"
    grass_dir = DATA_DIR / "grass"
    
    dandelion_count = len(list(dandelion_dir.glob("*.jpg")))
    grass_count = len(list(grass_dir.glob("*.jpg")))
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    classes = ['Dandelion', 'Grass']
    counts = [dandelion_count, grass_count]
    colors = ['#FFD700', '#90EE90']
    
    bars = ax.bar(classes, counts, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax.set_title('Dataset Distribution', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(counts) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add total
    total = sum(counts)
    ax.text(0.5, 0.95, f'Total: {total} images', 
            transform=ax.transAxes,
            ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('dataset_distribution.png', dpi=150, bbox_inches='tight')
    print("✅ Distribution plot saved to: dataset_distribution.png")
    plt.close()


def analyze_image_properties():
    """Analyze image properties (sizes, dimensions, etc.)"""
    
    dandelion_dir = DATA_DIR / "dandelion"
    grass_dir = DATA_DIR / "grass"
    
    all_images = list(dandelion_dir.glob("*.jpg")) + list(grass_dir.glob("*.jpg"))
    
    sizes = []
    widths = []
    heights = []
    
    print("\n📊 Analyzing image properties...")
    for img_path in all_images[:100]:  # Sample first 100 for speed
        try:
            img = Image.open(img_path)
            widths.append(img.width)
            heights.append(img.height)
            sizes.append(img_path.stat().st_size / 1024)  # KB
        except Exception as e:
            print(f"Error with {img_path}: {e}")
    
    # Print statistics
    print("\n" + "="*50)
    print("📈 IMAGE STATISTICS")
    print("="*50)
    print(f"Sample size: {len(widths)} images")
    print(f"\nDimensions:")
    print(f"  Width:  {np.mean(widths):.1f} ± {np.std(widths):.1f} px")
    print(f"  Height: {np.mean(heights):.1f} ± {np.std(heights):.1f} px")
    print(f"\nFile size:")
    print(f"  Mean: {np.mean(sizes):.1f} KB")
    print(f"  Min:  {np.min(sizes):.1f} KB")
    print(f"  Max:  {np.max(sizes):.1f} KB")
    print("="*50)


def main():
    print("🎨 Starting data visualization...\n")
    
    # Check if data exists
    if not DATA_DIR.exists():
        print("❌ Data directory not found!")
        print("💡 Please run download_images.py first")
        return
    
    # Plot sample images
    print("📸 Plotting sample images...")
    plot_sample_images(num_samples=10)
    
    # Plot distribution
    print("\n📊 Plotting distribution...")
    plot_dataset_distribution()
    
    # Analyze properties
    analyze_image_properties()
    
    print("\n✅ Visualization complete!")
    print("📁 Check: data_samples.png and dataset_distribution.png")


if __name__ == "__main__":
    main()
