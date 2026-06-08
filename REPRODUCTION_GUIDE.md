# Reproduction Guide: Historical 89.39% Accuracy

**Target:** Reproduce DenseNet121 + NONE = 89.39% validation accuracy  
**Approach:** Minimum viable changes to revert performance regression

---

## SIDE-BY-SIDE COMPARISON

### 1. NUMBER OF FRAMES (CRITICAL)

**Current (73% accuracy):**
```yaml
# config.yaml
training:
  num_frames: 64
```

**Historical (89.39% accuracy):**
```yaml
# config.yaml
training:
  num_frames: 16
```

**Change:** Replace `64` with `16` in config.yaml

---

### 2. DATA AUGMENTATIONS (CRITICAL)

**Current (73% accuracy):**
```python
# src/data_prep/dataset.py lines 43-54
if split == "train":
    self.transform = transforms.Compose([
        transforms.RandomResizedCrop((self.frame_size, self.frame_size), 
                                     scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                              saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.0)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
```

**Historical (89.39% accuracy):**
```python
# src/data_prep/dataset.py
if split == "train":
    self.transform = transforms.Compose([
        transforms.Resize((self.frame_size, self.frame_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
```

**Change:** Remove augmentations for training split

---

### 3. MODEL OUTPUT: VIDEO-LEVEL vs PER-FRAME (CRITICAL)

**Current (73% accuracy):**
```python
# src/models/temporal_brain.py line 95
def forward(self, x):
    # ... temporal processing ...
    out = self.temporal(x)  
    return self.classifier(out)  # Returns (B, T, num_classes)
```

**Historical (89.39% accuracy):**
```python
# _corrupted_backup/models.py
def forward(self, x):
    B, T, C, H, W = x.shape
    x_frames = x.view(B * T, C, H, W)
    feats = self.backbone(x_frames)
    feats = feats.view(B, T, -1)
    
    if self.temporal in ["bilstm", "lstm", "gru"]:
        out, _ = self.temporal_module(feats)
        clip_feat = out.mean(dim=1)  # ← TEMPORAL POOLING
        logits = self.classifier(clip_feat)
        return logits  # Returns (B, num_classes)
```

**Change:** Add temporal pooling in temporal_brain.py

```python
# src/models/temporal_brain.py (modified forward method)
def forward(self, x):
    """Returns video-level predictions (not per-frame)"""
    if self.temporal_type in ["tcn", "ms-tcn", "none"]:
        out = self.temporal(x)
    else:
        out, _ = self.temporal(x)
    
    # ← ADD THIS: Temporal average pooling
    clip_feat = out.mean(dim=1)  # (B, T, hidden) → (B, hidden)
    return self.classifier(clip_feat)  # (B, num_classes)
```

---

### 4. TRAINING LOOP: VIDEO-LEVEL LOSS (CRITICAL)

**Current (73% accuracy):**
```python
# src/training/train.py lines 104-112
features = spatial_model(clips)
logits = temporal_model(features)  # (B, 64, 12)

if labels.dim() == 1:
    labels = labels.unsqueeze(1).repeat(1, logits.size(1))  # (B,) → (B, 64)

loss = criterion(
    logits.reshape(-1, num_classes),     # (512, 12)
    labels.reshape(-1)                    # (512,)
)
```

**Historical (89.39% accuracy):**
```python
# _corrupted_backup/train.py
logits = model(clips)  # (B, 12)
loss = criterion(logits, labels)  # labels: (B,)
```

**Change:** Revert to direct loss computation

```python
# src/training/train.py (modified)
features = spatial_model(clips)
logits = temporal_model(features)  # (B, 12) after pooling
loss = criterion(logits, labels)   # labels: (B,)
```

**Also update validation:**
```python
# src/training/train.py validate() function
def validate(spatial, temporal, loader, device, criterion, config):
    spatial.eval()
    temporal.eval()
    metrics = MetricsTracker()
    
    with torch.no_grad():
        for batch in loader:
            if batch is None: continue
            
            clips, labels = batch
            clips, labels = clips.to(device), labels.to(device)
            
            with autocast(enabled=config["training"]["mixed_precision"]):
                features = spatial(clips)
                logits = temporal(features)  # (B, 12) not (B, 64, 12)
                
                # ← REMOVE label expansion
                loss = criterion(logits, labels)  # Direct loss
            
            preds = torch.argmax(logits, dim=1)  # (B,) not (B, 64)
            metrics.update(preds, labels, loss.item())
```

---

### 5. LOSS FUNCTION: NO LABEL SMOOTHING (MAJOR)

**Current (73% accuracy):**
```python
# src/training/train.py line 120
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**Historical (89.39% accuracy):**
```python
# _corrupted_backup/train.py
criterion = nn.CrossEntropyLoss()  # No label smoothing
```

**Change:** Remove label smoothing parameter

```python
# src/training/train.py
criterion = nn.CrossEntropyLoss()
```

---

### 6. OPTIMIZER: ADAM + LOW WEIGHT DECAY (MAJOR)

**Current (73% accuracy):**
```python
# src/training/train.py lines 121-124
optimizer = torch.optim.AdamW(
    params, 
    lr=float(config["training"]["optimizer"]["lr"]),      # 1e-4
    weight_decay=float(config["training"]["optimizer"]["weight_decay"])  # 1e-4
)
```

**Historical (89.39% accuracy):**
```python
# _corrupted_backup/train.py
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=1e-5  # 10x lower!
)
```

**Change:** Switch to Adam with lower weight decay

```python
# src/training/train.py
optimizer = torch.optim.Adam(
    params,
    lr=float(config["training"]["optimizer"]["lr"]),       # 1e-4
    weight_decay=1e-5  # Changed from 1e-4
)
```

**Also update config.yaml:**
```yaml
training:
  optimizer:
    name: "adam"           # Changed from "adamw"
    lr: 1.0e-4
    weight_decay: 1.0e-5   # Changed from 1.0e-4
```

---

### 7. LEARNING RATE SCHEDULER: StepLR (MODERATE)

**Current (73% accuracy):**
```python
# src/training/train.py lines 127-133
scheduler_type = config["training"]["scheduler"]["type"]
if scheduler_type == "cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["num_epochs"]
    )
else:
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config["training"]["scheduler"]["step_size"], 
        gamma=config["training"]["scheduler"]["gamma"]
    )
```

**Historical (89.39% accuracy):**
```python
# _corrupted_backup/train.py
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=5, gamma=0.5
)
```

**Change:** Ensure StepLR is used

```yaml
# config.yaml
training:
  scheduler:
    type: "step"        # Changed from "cosine"
    step_size: 5        # Every 5 epochs
    gamma: 0.5          # Multiply by 0.5
```

---

### 8. TRAINING EPOCHS (MINOR)

**Current (73% accuracy):**
```yaml
# config.yaml
training:
  num_epochs: 50
```

**Historical (89.39% accuracy):**
```yaml
# config.yaml
training:
  num_epochs: 30
```

**Change:** Reduce epochs

```yaml
# config.yaml
training:
  num_epochs: 30
```

---

### 9. DISABLE EARLY STOPPING (MINOR)

**Current (73% accuracy):**
```python
# src/training/train.py lines 140-142
early_stopping = EarlyStopping(patience=15, metric="f1", mode="max", min_delta=0.001)

# In training loop:
if early_stopping(val_metrics['f1'], epoch):
    break
```

**Historical (89.39% accuracy):**
```python
# No early stopping
```

**Change:** Comment out early stopping

```python
# src/training/train.py (modify)
# early_stopping = EarlyStopping(patience=15, metric="f1", mode="max", min_delta=0.001)

# In training loop:
# if early_stopping(val_metrics['f1'], epoch):
#     break
```

---

### 10. MODEL SELECTION: ACCURACY (MINOR)

**Current (73% accuracy):**
```python
# src/training/train.py lines 210-230
if val_metrics['f1'] > best_val_f1:
    best_val_f1 = val_metrics['f1']
    best_val_acc = val_metrics['accuracy']
    # Save checkpoint...
```

**Historical (89.39% accuracy):**
```python
# _corrupted_backup/train.py
if val_acc > best_val_acc:
    best_val_acc = val_acc
    # Save checkpoint...
```

**Change:** Revert to accuracy-based selection

```python
# src/training/train.py
if val_metrics['accuracy'] > best_val_acc:
    best_val_acc = val_metrics['accuracy']
    # Save checkpoint...
```

---

## IMPLEMENTATION CHECKLIST

- [ ] **1.** Update `config.yaml`: num_frames: 64 → 16
- [ ] **2.** Update `config.yaml`: Remove augmentations (or set to minimal)
- [ ] **3.** Modify `src/data_prep/dataset.py`: Remove augmentation transforms
- [ ] **4.** Modify `src/models/temporal_brain.py`: Add temporal pooling in forward()
- [ ] **5.** Modify `src/training/train.py`: Remove label expansion in training
- [ ] **6.** Modify `src/training/train.py`: Remove label expansion in validation
- [ ] **7.** Modify `src/training/train.py`: Change CrossEntropyLoss (remove label_smoothing)
- [ ] **8.** Modify `src/training/train.py`: Switch optimizer to Adam with weight_decay=1e-5
- [ ] **9.** Update `config.yaml`: optimizer: "adam", weight_decay: 1e-5
- [ ] **10.** Update `config.yaml`: scheduler: type: "step", step_size: 5, gamma: 0.5
- [ ] **11.** Update `config.yaml`: num_epochs: 30
- [ ] **12.** Modify `src/training/train.py`: Remove or comment out EarlyStopping
- [ ] **13.** Modify `src/training/train.py`: Revert model selection to accuracy

---

## VALIDATION

After applying changes:

```bash
cd e:\sp\SportsAction

# Run with DenseNet121 + NONE (historical best)
python src/training/train.py
```

**Expected output:**
```
Epoch 1/30 | 42.3s | Train Loss: 0.6234 Acc: 0.8234 | 
Val Loss: 0.5123 Acc: 0.8543 F1: 0.8401

...

Epoch 15/30 | 41.8s | Train Loss: 0.1234 Acc: 0.9567 | 
Val Loss: 0.2456 Acc: 0.8934 F1: 0.8912

✅ Best model saved (val_f1: 0.8934, val_acc: 0.8934)

...

Epoch 30/30 | 41.5s | Train Loss: 0.0821 Acc: 0.9723 | 
Val Loss: 0.2634 Acc: 0.8939 F1: 0.8923

✅ Training complete. Best val F1: 0.8939, Best val Acc: 0.8939
```

**Success criteria:**
- ✓ Validation accuracy: 87-91% (target: 89.39%)
- ✓ Train-val gap: 2-5%
- ✓ Convergence by epoch 15-20
- ✓ Early stopping NOT triggered

---

## VERIFICATION CHECKLIST

After training completes:

1. **Check final validation accuracy**
   ```bash
   # Should show ~89% in final logs
   grep "Best val" experiments/densenet121_none/training_log.json
   ```

2. **Compare with historical checkpoint**
   - Historical: 89.39% (DenseNet121 + NONE)
   - Your result: Should be 87-91%
   - If within range: ✓ Reproduction successful

3. **Validate train-val gap**
   - Should be 2-5% (healthy)
   - Not 98% - 73% (current broken state)

4. **Check convergence speed**
   - Peak accuracy by epoch 15-20
   - Not plateau at 73% from epoch 1

---

## NOTES

- The **single most impactful change** is reverting from 64 frames to 16 frames
- The **second most impactful change** is reverting model output from per-frame to video-level
- Together, these two changes should recover ~14% of the lost accuracy
- Remaining changes fine-tune hyperparameters
- Early stopping is NOT needed (converges within 30 epochs anyway)

