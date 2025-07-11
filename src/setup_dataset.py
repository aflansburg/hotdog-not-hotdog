#!/usr/bin/env python3
"""
Dataset Setup Script for Hot Dog / Not Hot Dog Classifier

This script downloads the Kaggle hot-dog-not-hot-dog dataset and organizes it
into the directory structure expected by the training code.

Requirements:
    - kaggle package: pip install kaggle
    - Kaggle API credentials configured (~/.kaggle/kaggle.json)

Usage:
    python setup_dataset.py

The script will:
1. Download the dataset from Kaggle
2. Extract it to a temporary directory
3. Organize images into the expected structure:
   data/hotdog/
   ├── train/
   │   ├── hot_dog/
   │   └── not_hot_dog/
   └── val/
       ├── hot_dog/
       └── not_hot_dog/
"""

import argparse
import shutil
import tempfile
from pathlib import Path

try:
    import kaggle
except ImportError:
    print("❌ Error: kaggle package not found.")
    print("Install dependencies with: uv sync")
    exit(1)


def check_kaggle_credentials() -> bool:
    """
    Check if Kaggle API credentials are configured.

    Returns:
        bool: True if credentials are configured, False otherwise.
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if not kaggle_json.exists():
        print("❌ Error: Kaggle API credentials not found.")
        print("\nTo set up Kaggle API credentials:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Download kaggle.json and place it at ~/.kaggle/kaggle.json")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False

    return True


def download_dataset(temp_dir: str) -> bool:
    """
    Download the hot dog dataset from Kaggle.

    Args:
        temp_dir: The temporary directory to download the dataset to.

    Returns:
        bool: True if dataset was downloaded successfully, False otherwise.
    """
    print("📥 Downloading hot-dog-not-hot-dog dataset from Kaggle...")

    try:
        kaggle.api.dataset_download_files(
            "dansbecker/hot-dog-not-hot-dog", path=temp_dir, unzip=True
        )
        print("✅ Dataset downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False


def create_directory_structure() -> Path:
    """
    Create the expected directory structure.

    Returns:
        Path: The base directory path.
    """
    base_dir = Path("data/hotdog")

    directories = [
        base_dir / "train" / "hot_dog",
        base_dir / "train" / "not_hot_dog",
        base_dir / "val" / "hot_dog",
        base_dir / "val" / "not_hot_dog",
    ]

    print("📁 Creating directory structure...")
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    return base_dir


def organize_images(temp_dir: str, base_dir: Path) -> bool:
    """
    Organize downloaded images into the expected structure.

    Args:
        temp_dir: The temporary directory containing the downloaded dataset.
        base_dir: The base directory path.

    Returns:
        bool: True if images were organized successfully, False otherwise.
    """
    temp_path = Path(temp_dir)

    train_hot_dog_src = temp_path / "train" / "hot_dog"
    train_not_hot_dog_src = temp_path / "train" / "not_hot_dog"
    test_hot_dog_src = temp_path / "test" / "hot_dog"
    test_not_hot_dog_src = temp_path / "test" / "not_hot_dog"

    if not train_hot_dog_src.exists():
        print("❌ Error: Expected dataset structure not found in downloaded files.")
        print(f"Looking for: {train_hot_dog_src}")
        print("Available files:")
        for item in temp_path.rglob("*"):
            if item.is_file():
                print(f"  {item}")
        return False

    print("📋 Organizing images...")

    # cp training images
    copy_images(train_hot_dog_src, base_dir / "train" / "hot_dog", "train/hot_dog")
    copy_images(
        train_not_hot_dog_src, base_dir / "train" / "not_hot_dog", "train/not_hot_dog"
    )

    # cp test images
    copy_images(test_hot_dog_src, base_dir / "val" / "hot_dog", "val/hot_dog")
    copy_images(
        test_not_hot_dog_src, base_dir / "val" / "not_hot_dog", "val/not_hot_dog"
    )

    return True


def copy_images(src_dir: Path, dest_dir: Path, category_name: str) -> None:
    """
    Copy images from source to destination directory.

    Args:
        src_dir: The source directory containing the images.
        dest_dir: The destination directory to copy the images to.
        category_name: The name of the category to copy the images to.

    Returns:
        None
    """
    if not src_dir.exists():
        print(f"⚠️  Warning: Source directory {src_dir} does not exist")
        return

    image_files = (
        list(src_dir.glob("*.jpg"))
        + list(src_dir.glob("*.jpeg"))
        + list(src_dir.glob("*.png"))
    )

    print(f"  Copying {len(image_files)} images to {category_name}...")

    for image_file in image_files:
        dest_file = dest_dir / image_file.name
        shutil.copy2(image_file, dest_file)

    print(f"  ✅ Copied {len(image_files)} images to {dest_dir}")


def verify_dataset(base_dir: Path) -> None:
    """
    Verify the dataset was organized correctly.

    Args:
        base_dir: The base directory path.

    Returns:
        None
    """
    print("🔍 Verifying dataset organization...")

    categories = [
        ("train/hot_dog", base_dir / "train" / "hot_dog"),
        ("train/not_hot_dog", base_dir / "train" / "not_hot_dog"),
        ("val/hot_dog", base_dir / "val" / "hot_dog"),
        ("val/not_hot_dog", base_dir / "val" / "not_hot_dog"),
    ]

    total_images = 0
    for category_name, category_path in categories:
        image_count = (
            len(list(category_path.glob("*.jpg")))
            + len(list(category_path.glob("*.jpeg")))
            + len(list(category_path.glob("*.png")))
        )
        print(f"  {category_name}: {image_count} images")
        total_images += image_count

    print("\n✅ Dataset verification complete!")
    print(f"📊 Total images organized: {total_images}")

    if total_images > 0:
        print("\n🎉 Dataset is ready for training!")
        print("You can now run: uv run hotdog-not-hotdog --train")
    else:
        print("\n❌ No images found. Please check the dataset download.")


def main():
    """
    Main function to set up the dataset.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Download and organize the Hot Dog / Not Hot Dog dataset from Kaggle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run hdnhd-setup         # Download and setup dataset
  uv run hdnhd-setup --help  # Show this help

Requirements:
  - Kaggle API credentials configured (~/.kaggle/kaggle.json)
  - Visit https://www.kaggle.com/account to get your API token

The script will:
  1. Download the dataset from Kaggle
  2. Extract it to a temporary directory  
  3. Organize images into the expected structure:
     data/hotdog/
     ├── train/
     │   ├── hot_dog/
     │   └── not_hot_dog/
     └── val/
         ├── hot_dog/
         └── not_hot_dog/
        """,
    )

    args = parser.parse_args()  # noqa: F841

    print("🌭 Hot Dog / Not Hot Dog Dataset Setup")
    print("=" * 40)

    if not check_kaggle_credentials():
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📂 Using temporary directory: {temp_dir}")

        if not download_dataset(temp_dir):
            return

        base_dir = create_directory_structure()

        if not organize_images(temp_dir, base_dir):
            return

        verify_dataset(base_dir)


if __name__ == "__main__":
    main()
