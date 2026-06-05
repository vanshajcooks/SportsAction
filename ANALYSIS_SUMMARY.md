# Technical Analysis Summary
**Generated:** June 5, 2026

---

## KEY FINDINGS

### 1. Label Flow (Dataset → Loss)
- **Input:** Video folder (e.g., `data/frames/train/backhand/p1_backhand_s1/`)
- **Label Creation:** Derived from parent folder name → `class_to_idx["backhand"]` → scalar `0`
- **Batching:** 8 videos → labels tensor of shape `(8,)` (one label per video)
- **CRITICAL:** Labels expanded from `(8,)` → `(8, 64)` by repeating same label 64 times
- **Loss:** CrossEntropyLoss treats 512 flattened frames as independent examples with identical per-video labels

### 2. Classification vs. Segmentation
- **Project Type:** VIDEO-LEVEL CLASSIFICATION (not true segmentation)
- **Why?** Only video-level labels exist (no frame boundaries)
- **Evidence:** `data/annotations/` is empty; dataset only has folder names
- **Output:** Model generates per-frame logits `(B, T=64, num_classes=12)` but trains on replicated labels
- **Temporal Pooling:** VideoClassifier does `mean(dim=1)` pooling, destroying per-frame info

### 3. Label Replication Across Frames
- **Location:** [src/training/train.py](src/training/train.py#L104-106)
- **Code:** `labels.unsqueeze(1).repeat(1, logits.size(1))`
- **Effect:** `[5, 2, 11, ...]` (video classes) → `[[5,5,...,5], [2,2,...,2], ...]` (64 times each)
- **Impact:** Forces model to predict same action for entire video; cannot learn temporal boundaries

### 4. Output & Target Tensor Shapes

| Stage | Output Shape | Target Shape | Notes |
|-------|:---:|:---:|---|
| Model Input | `(8, 64, 3, 224, 224)` | — | 8 videos, 64 frames |
| Spatial Features | `(8, 64, 512)` | — | ResNet18 features |
| Temporal Output | `(8, 64, 12)` | — | Per-frame logits |
| Loss Flattened | `(512, 12)` | `(512,)` | 8×64 frames vs repeated labels |

### 5. Temporal Boundaries: NOT Being Learned
- **Constraint:** All frames must predict same class (loss penalizes deviation)
- **Result:** Model cannot learn transitions between action phases
- **Example:** All 64 frames of "backhand" video forced to predict "backhand" class
- **No Gradient Signal:** No mechanism to learn "preparation phase" vs "follow-through phase"

### 6. Train-Validation Leakage
- ✅ **NO LEAKAGE DETECTED**
- Stratified split by video ID with fixed seed (`random_state=42`)
- Clear separation of train/val/test sets
- **Note:** No frame-level continuity preserved (not necessary for classification)

### 7. Class Balance & Dataset Statistics

```
Total Videos: 1,980
├── Train: 1,380 (70%)
├── Val:   300 (15%)
└── Test:  300 (15%)

Per-class distribution: Perfectly balanced
├── Each class: 115 videos per split
├── Total frames: 126,720 (64 frames × 1,980 videos)
└── Per-class frames: 7,360 per split (8.33%)

Imbalance: NONE (unrealistic for real-world data)
```

### 8. Top 5 Critical Weaknesses

| # | Issue | Severity | Root Cause |
|:---:|---|:---:|---|
| 1 | Video-level labels, not frame-level | 🔴 CRITICAL | Dataset design |
| 2 | Labels replicated across all frames | 🔴 CRITICAL | Training loop (lines 104-106) |
| 3 | No frame-level annotations exist | 🔴 CRITICAL | Empty `data/annotations/` |
| 4 | Loss treats frames as independent | 🟠 MAJOR | Loss function design |
| 5 | BiLSTM bidirectionality violates causality | 🟠 MAJOR | Architecture choice |

---

## DETAILED FINDINGS

### Finding 1: Video-Level Classification, Not Segmentation
**Evidence:**
- Dataset returns scalar labels per video: `(video_tensor, label)` where label is 0-11
- No frame-level ground truth (data/annotations/ empty)
- Training expands labels: `(8,)` → `(8, 64)` by repetition
- Loss computed on 512 frames with identical-per-video targets

**Impact:** Cannot learn temporal action transitions; all frames forced to same class

---

### Finding 2: Label Replication Is Explicit & Problematic
**Code Location:** [src/training/train.py](src/training/train.py#L104-106)
```python
if labels.dim() == 1:
    labels = labels.unsqueeze(1).repeat(1, logits.size(1))  # (B,) → (B, T=64)
```

**Example:**
```
Input:  [class_id_video1, class_id_video2, ..., class_id_video8]  # (8,)
Output: [
    [class_id_video1, class_id_video1, ..., class_id_video1],    # 64 copies
    [class_id_video2, class_id_video2, ..., class_id_video2],    # 64 copies
    ...
]  # (8, 64)
```

**Loss Computation:**
```
CrossEntropyLoss(
    logits.reshape(512, 12),     # 512 frame predictions
    labels.reshape(512,)         # 512 frame labels (all identical per video)
)
```

---

### Finding 3: Temporal Boundaries Cannot Be Learned
**Why?**
```
Model Goal (with current setup):
  Minimize: sum of (cross_entropy(logits[frame], target_label) for all frames)
  Where: target_label = video_class (repeated for all 64 frames)

Consequence:
  ∂Loss/∂logits[frame_i] favors predicting video_class
  All frames weighted equally toward same class
  Zero gradient for learning phase transitions
```

**What Could Happen:**
```
Real scenario (desired):
  Frame 0-15:  "preparation"      ← Different action phases
  Frame 16-50: "strike"           ← Should predict different classes
  Frame 51-63: "follow-through"   ← Should predict different classes

What actually happens:
  Frame 0-15:  "backhand" (forced by loss)
  Frame 16-50: "backhand" (forced by loss)
  Frame 51-63: "backhand" (forced by loss)
  Loss penalizes any deviation from uniform prediction
```

---

### Finding 4: Output Shapes & Their Implications

**Full Shape Transformation:**
```
Input:     (8, 64, 3, 224, 224)    ← 8 videos, 64 frames, RGB, 224×224
   ↓ SpatialExtractor (ResNet18)
Features:  (8, 64, 512)            ← Spatial features per frame
   ↓ TemporalBrain (BiLSTM)
Logits:    (8, 64, 12)             ← Per-frame class logits
   ↓ Flatten for loss
Flat:      (512, 12) vs (512,)     ← Loss computation
```

**But VideoClassifier does:**
```
Logits:    (8, 64, 12)             ← Per-frame predictions
   ↓ Temporal pooling: mean(dim=1)
Pooled:    (8, 12)                 ← Video-level predictions (loses frame info!)
```

**Inconsistency:** TemporalBrain returns per-frame logits, but VideoClassifier pools them away

---

### Finding 5: Perfect Class Balance (Unrealistic)

**Dataset Distribution:**
```
backhand:          115 videos (8.33%)
backhand2hands:    115 videos (8.33%)
backhand_slice:    115 videos (8.33%)
... (all 12 classes identical)
```

**Real-World Impact:**
- Tennis matches have more forehands than smashes
- Dataset distribution is artificially uniform
- Model won't generalize to real match data

---

## FILE REFERENCES (Quick Links)

### Critical Code Locations

| Issue | File | Lines | Code |
|---|---|:---:|---|
| Label repetition | [src/training/train.py](src/training/train.py#L104-L106) | 104-106 | `labels.unsqueeze(1).repeat(1, logits.size(1))` |
| Video-level labels | [src/data_prep/dataset.py](src/data_prep/dataset.py#L75-L82) | 75-82 | `return video_tensor, torch.tensor(label)` |
| Loss computation | [src/training/train.py](src/training/train.py#L107-L112) | 107-112 | `loss = criterion(logits.reshape(-1, num_classes), ...)` |
| BiLSTM config | [src/models/temporal_brain.py](src/models/temporal_brain.py#L58-L65) | 58-65 | `bidirectional=is_bidir` |
| Temporal pooling | [src/models/video_classifier.py](src/models/video_classifier.py#L67-L70) | 67-70 | `clip_feat = out.mean(dim=1)` |
| Empty annotations | [data/annotations/](data/annotations/) | — | No frame-level labels |

---

## RECOMMENDATIONS FOR IMPROVEMENT

### Priority 1: Add Frame-Level Annotations
- Create CSV files with `video_id, frame_start, frame_end, action_class`
- Define temporal boundaries for action phases
- Enables true temporal segmentation training

### Priority 2: Remove Label Replication
- Load actual frame-level ground truth instead of repeating video labels
- Implement temporal loss (smoothness) constraint
- Add temporal boundary metrics (F1@IoU, precision/recall of transitions)

### Priority 3: Fix Architecture
- Use unidirectional (forward-only) LSTM instead of BiLSTM
- Remove temporal pooling; keep per-frame predictions for segmentation
- Add temporal attention mechanism

### Priority 4: Data Augmentation
- Add random crops, horizontal flips, color jitter
- Improves robustness with limited training data (1,380 videos)

### Priority 5: Metrics
- Implement F1@IoU (F1-score at different IoU thresholds)
- Add boundary precision/recall metrics
- Track temporal coherence (consistency of predictions)

---

## CONCLUSION

**Current State:** Video classification system with frame-level output architecture

**Claim:** "Action segmentation" system

**Reality Gap:** Vast. The system cannot learn temporal boundaries due to:
1. Video-level labels only
2. Label replication across all frames
3. Loss function that penalizes temporal variation
4. Evaluation metrics that don't measure boundary accuracy

**To Achieve True Segmentation:** Requires frame-level ground truth, proper loss functions, and temporal metrics

**Full Report:** See `TECHNICAL_ANALYSIS_REPORT.md` for complete analysis with code snippets and detailed explanations

