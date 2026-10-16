#!/usr/bin/env python3
"""
Create a derived dataset by selecting alternating images from source dataset.
"""
import os
import shutil
from pathlib import Path

# Sampling rate: select 1 out of every M images
M = 6

# Source and destination directories
source_dir = Path("data/incoming/tomato-bed_2026-06-17/images")
dest_dir = Path("data/incoming/tomato-small/images")

# Create destination directory
dest_dir.mkdir(parents=True, exist_ok=True)

# Get all image files and sort by name
image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
all_images = sorted([
    f for f in source_dir.iterdir()
    if f.is_file() and f.suffix in image_extensions
])

print(f"Found {len(all_images)} images in source directory")

# Select 1 out of every M images
selected_images = all_images[::M]

print(f"Selecting {len(selected_images)} images (1 out of every {M})")

# Copy selected images to destination
for img in selected_images:
    dest_path = dest_dir / img.name
    shutil.copy2(img, dest_path)

print(f"\nDataset created at: {dest_dir.parent}")
print(f"Total images copied: {len(selected_images)}")
print(f"\nFirst 5 selected images:")
for img in selected_images[:5]:
    print(f"  - {img.name}")
