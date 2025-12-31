# upload_dataset.py
import os
import torch
from torchvision import datasets, transforms
from minio import Minio
from minio.error import S3Error
import tarfile
import shutil

print("📥 Downloading CIFAR-10 dataset...")

# Download CIFAR-10 dataset locally first
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

print("✅ CIFAR-10 downloaded successfully!")

# Create a tar.gz archive of the dataset
print("\n📦 Creating dataset archive...")
archive_name = "cifar10-dataset.tar.gz"

with tarfile.open(archive_name, "w:gz") as tar:
    tar.add("./data/cifar-10-batches-py", arcname="cifar-10-batches-py")

print(f"✅ Archive created: {archive_name}")

# Upload to MinIO
print("\n☁️  Uploading to MinIO...")

minio_client = Minio(
    "127.0.0.1:57888",  # Update with your actual URL
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

try:
    # Upload the archive
    minio_client.fput_object(
        bucket_name="raw-data",
        object_name="cifar10-dataset.tar.gz",
        file_path=archive_name,
    )
    print(f"✅ Uploaded {archive_name} to raw-data bucket")
    
    # Verify upload
    objects = minio_client.list_objects("raw-data")
    print("\n📂 Files in raw-data bucket:")
    for obj in objects:
        print(f"  - {obj.object_name} ({obj.size / (1024*1024):.2f} MB)")
    
    # Cleanup local archive
    os.remove(archive_name)
    print(f"\n🧹 Cleaned up local archive")
    
except S3Error as e:
    print(f"❌ MinIO Error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n✅ Dataset upload complete!")