# Advanced Project: Production Image Classifier

## Overview

Build a **production-grade** image classification system from scratch, including:
- Custom CNN architecture
- Data augmentation pipeline
- Transfer learning (fine-tune pre-trained model)
- Model optimization (quantization, pruning)
- REST API deployment
- Monitoring & observability

**Goal**: Deploy a real-time image classifier with <100ms latency.

## Tech Stack

- **Training**: PyTorch
- **Serving**: FastAPI + TorchServe
- **Monitoring**: Prometheus + Grafana
- **Storage**: S3 (dataset), Docker Registry (models)
- **Deployment**: Docker + Kubernetes

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Training Pipeline                         │
│                                                              │
│  Raw Data → Preprocessing → Augmentation → Training →        │
│  Validation → Model Export → Model Registry                  │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                    Serving Pipeline                          │
│                                                              │
│  API Request → Image Preprocessing → Model Inference →       │
│  Postprocessing → Response (with confidence scores)          │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                    Monitoring Pipeline                       │
│                                                              │
│  Metrics Collection → Prometheus → Grafana Dashboard         │
│  (Latency, Throughput, Accuracy, Drift Detection)            │
└──────────────────────────────────────────────────────────────┘
```

## Project Structure

```
advanced_project/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── kubernetes/
│   ├── deployment.yaml
│   └── service.yaml
├── data/
│   ├── download_dataset.py
│   ├── augmentation.py
│   └── data_loader.py
├── models/
│   ├── custom_cnn.py
│   ├── resnet_finetune.py
│   └── model_utils.py
├── training/
│   ├── train.py
│   ├── configs.yaml
│   └── callbacks.py
├── optimization/
│   ├── quantization.py
│   ├── pruning.py
│   └── distillation.py
├── serving/
│   ├── api.py
│   ├── inference.py
│   └── preprocessing.py
├── monitoring/
│   ├── prometheus_metrics.py
│   └── drift_detection.py
├── tests/
│   ├── test_model.py
│   ├── test_api.py
│   └── load_test.py
└── notebooks/
    ├── 01_eda.ipynb
    ├── 02_model_experiments.ipynb
    └── 03_optimization_analysis.ipynb
```

## Phase 1: Dataset & Exploration (Week 1)

### Task 1.1: Choose Dataset

Pick one:
- **CIFAR-100** (100 classes, 32x32)
- **Food-101** (101 food categories, varies)
- **Stanford Dogs** (120 dog breeds, varies)
- **Custom**: Collect your own dataset (Recommended!)

### Task 1.2: EDA & Data Pipeline

```python
# data/data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import albumentations as A

class CustomDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.data = self._load_data(root, split)
        self.transform = transform
    
    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image=np.array(image))['image']
        return image, label
    
    def __len__(self):
        return len(self.data)

def get_data_loaders(root, batch_size=64):
    # Training augmentation
    train_transform = A.Compose([
        A.RandomResizedCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Validation: minimal preprocessing
    val_transform = A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    train_dataset = CustomDataset(root, 'train', train_transform)
    val_dataset = CustomDataset(root, 'val', val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader
```

**Deliverables**:
- [ ] Dataset downloaded and organized
- [ ] EDA notebook with class distribution, image stats
- [ ] Data loader with augmentation
- [ ] Visualization of augmented samples

## Phase 2: Model Development (Week 2-3)

### Task 2.1: Custom CNN Architecture

```python
# models/custom_cnn.py
import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    \"\"\"Modern CNN with residual connections and efficient design.\"\"\"
    
    def __init__(self, num_classes=100):
        super().__init__()
        
        # Stem: aggressive downsampling
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 128, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, 512, blocks=2, stride=2)
        
        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
```

### Task 2.2: Transfer Learning

```python
# models/resnet_finetune.py
import torch
import torch.nn as nn
from torchvision import models

def get_resnet50_pretrained(num_classes=100, freeze_backbone=True):
    \"\"\"Load pre-trained ResNet50 and adapt for your task.\"\"\"
    model = models.resnet50(weights='IMAGENET1K_V2')
    
    # Freeze backbone (optional)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace classifier
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model
```

### Task 2.3: Training Script

```python
# training/train.py
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

def train_model(model, train_loader, val_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Initialize wandb
    wandb.init(project='image-classifier', config=config)
    
    best_val_acc = 0
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss, train_acc = 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == labels).float().mean().item()
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Logging
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': scheduler.get_last_lr()[0]
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        scheduler.step()
    
    return model

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc = 0, 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            total_acc += (outputs.argmax(1) == labels).float().mean().item()
    
    return total_loss / len(loader), total_acc / len(loader)
```

**Deliverables**:
- [ ] Custom CNN implementation
- [ ] Transfer learning baseline
- [ ] Training script with mixed precision
- [ ] Experiment tracking with WandB
- [ ] Model comparison (custom vs transfer learning)

## Phase 3: Optimization (Week 4)

### Task 3.1: Quantization

```python
# optimization/quantization.py
import torch
from torch.quantization import quantize_dynamic, quantize_static, prepare_qat, convert

def post_training_quantization(model, dataloader, backend='fbgemm'):
    \"\"\"Dynamic or static quantization after training.\"\"\"
    
    # Dynamic quantization (simple, works for many models)
    quantized_model = quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )
    
    return quantized_model

def quantization_aware_training(model, train_loader, config):
    \"\"\"QAT: Train with quantization in mind.\"\"\"
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model = prepare_qat(model)
    
    # Train model (same as before)
    train_model(model, train_loader, config)
    
    # Convert to quantized
    quantized_model = convert(model)
    return quantized_model
```

### Task 3.2: Pruning

```python
# optimization/pruning.py
import torch.nn.utils.prune as prune

def prune_model(model, amount=0.3):
    \"\"\"Prune weights to reduce model size.\"\"\"
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
    
    return model
```

**Target Metrics**:
- Model size: <50MB
- Inference latency: <50ms (CPU), <10ms (GPU)
- Accuracy drop: <2% compared to full precision

**Deliverables**:
- [ ] Quantized model (INT8)
- [ ] Pruned model (30% sparsity)
- [ ] Benchmark comparison (size, latency, accuracy)

## Phase 4: Deployment (Week 5)

### Task 4.1: FastAPI Service

```python
# serving/api.py
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import io

app = FastAPI(title=\"Image Classifier API\")

# Load model once at startup
model = load_model('best_model_quantized.pth')
model.eval()

@app.post(\"/predict\")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Preprocess
    tensor = preprocess(image)
    
    # Inference
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        top5_prob, top5_idx = torch.topk(probs, 5)
    
    # Response
    results = [
        {\"class\": idx2label[idx.item()], \"confidence\": prob.item()}
        for prob, idx in zip(top5_prob[0], top5_idx[0])
    ]
    
    return {\"predictions\": results}

@app.get(\"/health\")
async def health():
    return {\"status\": \"healthy\"}
```

### Task 4.2: Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run
CMD [\"uvicorn\", \"serving.api:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - \"8000:8000\"
    environment:
      - MODEL_PATH=/models/best_model.pth
    volumes:
      - ./models:/models
  
  prometheus:
    image: prom/prometheus
    ports:
      - \"9090:9090\"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
  
  grafana:
    image: grafana/grafana
    ports:
      - \"3000:3000\"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

**Deliverables**:
- [ ] REST API with FastAPI
- [ ] Dockerfile & docker-compose
- [ ] Load testing (100 concurrent requests)
- [ ] Latency <100ms at P99

## Phase 5: Monitoring & Production (Week 6)

### Task 5.1: Metrics Collection

```python
# monitoring/prometheus_metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency')
PREDICTION_CONFIDENCE = Histogram('prediction_confidence', 'Prediction confidence')
MODEL_VERSION = Gauge('model_version', 'Current model version')

@app.middleware(\"http\")
async def monitor_requests(request, call_next):
    REQUEST_COUNT.inc()
    
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time
    
    REQUEST_LATENCY.observe(latency)
    
    return response
```

### Task 5.2: Drift Detection

```python
# monitoring/drift_detection.py
from scipy.stats import wasserstein_distance

class DriftDetector:
    def __init__(self, reference_dist):
        self.reference_dist = reference_dist
    
    def detect_drift(self, current_dist, threshold=0.1):
        \"\"\"Detect distribution shift using Wasserstein distance.\"\"\"
        distance = wasserstein_distance(self.reference_dist, current_dist)
        return distance > threshold, distance
```

**Deliverables**:
- [ ] Prometheus metrics endpoint
- [ ] Grafana dashboard (latency, throughput, accuracy)
- [ ] Drift detection alerting
- [ ] Performance report

## Evaluation Criteria

| Category | Weight | Criteria |
|----------|--------|----------|
| **Model Quality** | 25% | Accuracy, robustness, error analysis |
| **Optimization** | 20% | Model size, inference speed, quality retention |
| **API Design** | 15% | RESTful, documented, error handling |
| **Performance** | 20% | Latency, throughput, scalability |
| **Monitoring** | 10% | Metrics, logging, alerting |
| **Documentation** | 10% | README, API docs, deployment guide |

## Success Metrics

**Minimum Viable Product**:
- ✅ Test accuracy >85%
- ✅ REST API deployed
- ✅ Latency <200ms

**Production Ready**:
- 🌟 Test accuracy >90%
- 🌟 Latency <100ms at P99
- 🌟 Monitoring dashboard
- 🌟 Docker deployment

**Industry Grade**:
- 🏆 Test accuracy >95%
- 🏆 Latency <50ms at P99
- 🏆 Kubernetes deployment
- 🏆 A/B testing framework
- 🏆 Automated retraining pipeline

## Resources

- [PyTorch Production Tutorial](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/naming/)
- [Model Serving Patterns](https://ml-ops.org/content/serving)

This project will give you hands-on experience with the entire ML lifecycle — from research to production!
