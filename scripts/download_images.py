#!/usr/bin/env python3
"""
Script to download and preprocess images from CSV files
"""

import os
import sys
import requests
from pathlib import Path
from PIL import Image
import io
from tqdm import tqdm

# Configuration
DATA_DIR = Path("data")
DANDELION_DIR = DATA_DIR / "dandelion"
GRASS_DIR = DATA_DIR / "grass"
IMAGE_SIZE = (224, 224)

# Create directories
DANDELION_DIR.mkdir(parents=True, exist_ok=True)
GRASS_DIR.mkdir(parents=True, exist_ok=True)


def read_urls_from_csv(csv_file):
    """Read URLs from CSV file (skip empty lines)"""
    with open(csv_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    return urls


def download_image(url, timeout=10):
    """Download image from URL"""
    try:
        response = requests.get(url, timeout=timeout, headers={
            'User-Agent': 'Mozilla/5.0'
        })
        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content))
            return image
        return None
    except Exception as e:
        return None


def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image: convert to RGB and resize"""
    try:
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        return image
    except Exception:
        return None


def process_images(csv_file, output_dir, label, max_images=None):
    """Download and process images from CSV"""
    print(f"\n🔄 Processing {label} images...")
    
    urls = read_urls_from_csv(csv_file)
    if max_images:
        urls = urls[:max_images]
    
    print(f"📊 Found {len(urls)} URLs")
    
    success_count = 0
    failed_count = 0
    
    for idx, url in enumerate(tqdm(urls, desc=f"Downloading {label}")):
        try:
            # Download
            image = download_image(url)
            if image is None:
                failed_count += 1
                continue
            
            # Preprocess
            image = preprocess_image(image, IMAGE_SIZE)
            if image is None:
                failed_count += 1
                continue
            
            # Save
            filename = f"{idx:08d}.jpg"
            filepath = output_dir / filename
            image.save(filepath, 'JPEG', quality=95)
            
            success_count += 1
            
        except Exception as e:
            failed_count += 1
            continue
    
    print(f"✅ {label}: {success_count} images downloaded")
    print(f"❌ {label}: {failed_count} images failed")
    
    return success_count, failed_count


def main():
    print("🚀 Starting image download and preprocessing...")
    
    # Process dandelions
    d_success, d_failed = process_images(
        "dandelion.csv", 
        DANDELION_DIR, 
        "dandelion",
        max_images=400  # Augmenté pour avoir plus d'images
    )
    
    # Process grass
    g_success, g_failed = process_images(
        "grass.csv", 
        GRASS_DIR, 
        "grass",
        max_images=400  # Augmenté pour avoir plus d'images
    )
    
    print("\n" + "="*50)
    print("📊 SUMMARY")
    print("="*50)
    print(f"Total downloaded: {d_success + g_success}")
    print(f"Total failed: {d_failed + g_failed}")
    print(f"Dandelions: {d_success}/{d_success + d_failed}")
    print(f"Grass: {g_success}/{g_success + g_failed}")
    print(f"\n📁 Images saved in: {DATA_DIR.absolute()}")
    print("="*50)


if __name__ == "__main__":
    main()
