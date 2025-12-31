# feature_engineering.py
import os
import torch
import tarfile
from torchvision import datasets, transforms
from minio import Minio
from minio.error import S3Error
import pickle

print("🔧 Starting Feature Engineering Pipeline...")

# MinIO client
minio_client = Minio(
    "127.0.0.1:57888",  # Update if needed
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

# Step 1: Download raw data from MinIO
print("\n📥 Downloading raw dataset from MinIO...")
try:
    minio_client.fget_object(
        bucket_name="raw-data",
        object_name="cifar10-dataset.tar.gz",
        file_path="cifar10-raw.tar.gz"
    )
    print("✅ Downloaded cifar10-dataset.tar.gz")
except S3Error as e:
    print(f"❌ Error downloading: {e}")
    exit(1)

# Step 2: Extract dataset
print("\n📦 Extracting dataset...")
with tarfile.open("cifar10-raw.tar.gz", "r:gz") as tar:
    tar.extractall("./data_raw")
print("✅ Dataset extracted")

# Step 3: Apply feature engineering (normalization + augmentation)
print("\n🔧 Applying feature engineering...")

# Training transforms with augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Test transforms (no augmentation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load datasets
train_dataset = datasets.CIFAR10(root="./data_raw", train=True, download=False, transform=train_transform)
test_dataset = datasets.CIFAR10(root="./data_raw", train=False, download=False, transform=test_transform)

print(f"✅ Train samples: {len(train_dataset)}")
print(f"✅ Test samples: {len(test_dataset)}")

# Step 4: Save processed datasets
print("\n💾 Saving processed datasets...")
os.makedirs("./processed", exist_ok=True)

# Save train transform info
with open("./processed/train_transform.pkl", "wb") as f:
    pickle.dump(train_transform, f)

with open("./processed/test_transform.pkl", "wb") as f:
    pickle.dump(test_transform, f)

# Save metadata
metadata = {
    "train_size": len(train_dataset),
    "test_size": len(test_dataset),
    "num_classes": 10,
    "image_size": (32, 32),
    "normalization_mean": (0.4914, 0.4822, 0.4465),
    "normalization_std": (0.2023, 0.1994, 0.2010),
    "augmentation": ["RandomHorizontalFlip", "RandomCrop"]
}

with open("./processed/metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("✅ Processed datasets saved")

# Step 5: Create archive of processed data
print("\n📦 Creating processed data archive...")
archive_name = "cifar10-processed.tar.gz"

with tarfile.open(archive_name, "w:gz") as tar:
    tar.add("./data_raw/cifar-10-batches-py", arcname="cifar-10-batches-py")
    tar.add("./processed", arcname="processed")

print(f"✅ Archive created: {archive_name}")

# Step 6: Upload to MinIO processed-data bucket
print("\n☁️  Uploading to MinIO processed-data bucket...")
try:
    minio_client.fput_object(
        bucket_name="processed-data",
        object_name="cifar10-processed.tar.gz",
        file_path=archive_name,
    )
    print(f"✅ Uploaded {archive_name} to processed-data bucket")
    
    # Verify upload
    objects = minio_client.list_objects("processed-data")
    print("\n📂 Files in processed-data bucket:")
    for obj in objects:
        print(f"  - {obj.object_name} ({obj.size / (1024*1024):.2f} MB)")
    
except S3Error as e:
    print(f"❌ MinIO Error: {e}")
    exit(1)

# Cleanup
print("\n🧹 Cleaning up temporary files...")
os.remove("cifar10-raw.tar.gz")
os.remove(archive_name)
import shutil
shutil.rmtree("./data_raw")
shutil.rmtree("./processed")

print("\n✅ Feature Engineering Complete!")
print("📊 Summary:")
print(f"  - Input: raw-data/cifar10-dataset.tar.gz")
print(f"  - Output: processed-data/cifar10-processed.tar.gz")
print(f"  - Train samples: {metadata['train_size']}")
print(f"  - Test samples: {metadata['test_size']}")
print(f"  - Augmentations: {', '.join(metadata['augmentation'])}")