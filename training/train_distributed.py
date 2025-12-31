from datetime import timedelta
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import models, datasets, transforms
from minio import Minio
import tarfile
import shutil
import mlflow
import pickle

def setup_distributed():
    """Initialize distributed training environment with retry logic"""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    
    if world_size > 1:
        import time
        
        # Get master address from environment
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "23456")
        
        print(f"🔗 Rank {rank}: Attempting connection to {master_addr}:{master_port}")
        print(f"   World size: {world_size}")
        
        # Wait a bit for network to stabilize
        if rank != 0:
            time.sleep(10)  # Workers wait for master
        
        # Initialize with retries
        max_retries = 5
        for attempt in range(max_retries):
            try:
                print(f"🔄 Rank {rank}: Connection attempt {attempt + 1}/{max_retries}")
                
                dist.init_process_group(
                    backend="gloo",
                    init_method=f"tcp://{master_addr}:{master_port}",
                    world_size=world_size,
                    rank=rank,
                    timeout=timedelta(minutes=15)
                )
                
                print(f"✅ Rank {rank}: Successfully initialized distributed training!")
                break
                
            except Exception as e:
                print(f"⚠️  Rank {rank}: Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"   Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Rank {rank}: All connection attempts failed!")
                    raise
    else:
        print("✅ Running in non-distributed mode (single worker)")
    
    return world_size, rank

def download_data_from_minio(minio_endpoint, rank):
    """Download processed data from MinIO"""
    if rank == 0:  # Only rank 0 downloads
        print(f"📥 Rank {rank}: Downloading data from MinIO...")
        
        minio_client = Minio(
            minio_endpoint,
            access_key="minioadmin",
            secret_key="minioadmin",
            secure=False
        )
        
        # Download processed data
        minio_client.fget_object(
            bucket_name="processed-data",
            object_name="cifar10-processed.tar.gz",
            file_path="cifar10-processed.tar.gz"
        )
        
        # Extract
        with tarfile.open("cifar10-processed.tar.gz", "r:gz") as tar:
            tar.extractall("./data")
        
        print(f"✅ Rank {rank}: Data downloaded and extracted")
    
    # Wait for rank 0 to finish downloading
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.barrier()

def train():
    # Setup
    world_size, rank = setup_distributed()
    device = torch.device("cpu")
    
    # MinIO endpoint (service name in Kubernetes)
    minio_endpoint = os.environ.get("MINIO_ENDPOINT", "minio.minio.svc.cluster.local:9000")
    
    # Download data
    download_data_from_minio(minio_endpoint, rank)
    
    # Load datasets with transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)
    
    # Distributed sampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    
    batch_size = 32 # 64
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = models.resnet18(num_classes=10)
    model = model.to(device)
    
    if world_size > 1:
        model = DDP(model)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    # MLflow tracking (only rank 0)
    if rank == 0:
        mlflow.set_tracking_uri("http://mlflow.mlflow.svc.cluster.local:5000")
        mlflow.set_experiment("cifar10-distributed-training")
        mlflow.start_run()
        mlflow.log_params({
            "model": "resnet18",
            "batch_size": batch_size,
            "world_size": world_size,
            "epochs": 10,
            "optimizer": "SGD",
            "lr": 0.01
        })
    
    # Training loop
    epochs = 10
    for epoch in range(epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        scheduler.step()
        
        train_acc = 100.0 * correct / total
        train_loss = running_loss / len(train_loader)
        
        if rank == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            mlflow.log_metrics({"train_loss": train_loss, "train_acc": train_acc}, step=epoch)
    
    # Evaluation (only rank 0)
    if rank == 0:
        model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_acc = 100.0 * correct / total
        test_loss = test_loss / len(test_loader)
        
        print(f"\n✅ Test Accuracy: {test_acc:.2f}%")
        print(f"✅ Test Loss: {test_loss:.4f}")
        
        mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc})
        
        # Save model to MinIO
        model_path = "resnet18_cifar10.pth"
        torch.save(model.state_dict() if world_size == 1 else model.module.state_dict(), model_path)
        
        minio_client = Minio(
            minio_endpoint,
            access_key="minioadmin",
            secret_key="minioadmin",
            secure=False
        )
        
        minio_client.fput_object(
            bucket_name="models",
            object_name="resnet18_cifar10.pth",
            file_path=model_path
        )
        
        print(f"✅ Model saved to MinIO: models/resnet18_cifar10.pth")
        
        mlflow.log_artifact(model_path)
        mlflow.end_run()
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    train()