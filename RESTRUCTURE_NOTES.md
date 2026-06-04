# Project Restructuring Summary

## Overview

The SportsAction project has been restructured for better modularity, maintainability, and scalability. The new organization separates concerns into distinct modules while keeping all your experimental results intact.

## Key Changes

### ✅ New Directory Structure

```
tennis_segmentation/
├── data/
│   ├── raw/              # Original video files (new)
│   ├── frames/           # Extracted frames (moved from processed/)
│   └── annotations/      # Labels/metadata (new)
├── src/
│   ├── data_prep/        # Data processing module (NEW)
│   │   ├── extract_frames.py
│   │   └── dataset.py
│   ├── models/           # Model architectures module (NEW)
│   │   ├── spatial_extractor.py      (split from models.py)
│   │   ├── temporal_brain.py         (split from models.py)
│   │   ├── video_classifier.py       (main model)
│   │   └── __init__.py
│   ├── training/         # Training & evaluation (NEW)
│   │   ├── train.py      (from src/train.py)
│   │   ├── test.py       (from src/test_model.py)
│   │   ├── metrics.py    (NEW - evaluation metrics)
│   │   └── __init__.py
│   └── utils/            # Utilities (NEW)
│       ├── visualizer.py (NEW - visualization tools)
│       └── __init__.py
├── notebooks/            # Jupyter notebooks (NEW)
│   └── example_usage.ipynb
├── experiments/          # Trained models (UNCHANGED - preserved)
├── config.yaml           # Global configuration (NEW)
├── requirements.txt      # Dependencies (UNCHANGED)
└── README.md            # Updated documentation
```

### 📦 Module Organization

**Before:** All code in `src/` with mixed concerns
```
src/
├── train.py           # Training
├── test_model.py      # Testing
├── models.py          # All models mixed
├── dataset.py         # Dataset
└── prepare_data.py    # Data prep
```

**After:** Separated by functionality
```
src/
├── data_prep/         # Data processing
├── models/            # Model definitions
├── training/          # Training & evaluation
├── utils/             # Utilities
└── __init__.py
```

### 🎯 Key Improvements

1. **Modularity**: Each component has a single responsibility
2. **Scalability**: Easy to add new models, training methods, utilities
3. **Reusability**: Import specific modules without loading everything
4. **Maintainability**: Clear organization for future development
5. **Testing**: Each module can be tested independently

## Migration Guide

### For Existing Scripts

**Old imports:**
```python
from src.models import VideoClassifier
from src.dataset import SportsDataset
from src.prepare_data import extract_frames
```

**New imports:**
```python
from src.models.video_classifier import VideoClassifier
from src.data_prep.dataset import SportsDataset
from src.data_prep.extract_frames import extract_frames
```

### For Training

**Old:**
```bash
python src/train.py --backbone resnet18 --temporal bilstm
```

**New:**
```bash
python src/training/train.py --backbone resnet18 --temporal bilstm
```

### For Testing

**Old:**
```bash
python src/test_model.py --backbone resnet18 --temporal bilstm
```

**New:**
```bash
python src/training/test.py --backbone resnet18 --temporal bilstm
```

## New Features

### 1. Metrics Module (`src/training/metrics.py`)
Comprehensive evaluation metrics:
- Accuracy
- F1-Score (weighted, macro, micro)
- Precision & Recall
- Confusion Matrix
- IoU (Intersection over Union)
- `MetricsTracker` class for batch aggregation

### 2. Visualizer (`src/utils/visualizer.py`)
Video visualization tools:
- Overlay predictions on frames
- Create annotated videos
- Save visualizations
- Class-based color mapping

### 3. Configuration (`config.yaml`)
Centralized configuration:
- Data paths and preprocessing
- Model architecture settings
- Training hyperparameters
- Evaluation metrics
- Device & runtime options

### 4. Jupyter Notebook (`notebooks/example_usage.ipynb`)
Complete example showing:
- Dataset loading
- Model creation
- Forward passes
- Metric computation
- Architecture comparison

## Preserved Elements

✅ **Virtual Environment** (`venv/`) - Completely preserved
✅ **Experiments** (`experiments/`) - All trained models saved
✅ **Requirements** (`requirements.txt`) - Unchanged
✅ **Data** (`data/`) - Structure enhanced, not modified

## Getting Started with New Structure

### 1. Install (no changes needed)
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
```bash
python src/data_prep/extract_frames.py \
    --frame-size 224 \
    --output-dir data/frames
```

### 3. Train Model
```bash
python src/training/train.py \
    --backbone resnet18 \
    --temporal bilstm \
    --batch_size 8 \
    --epochs 30
```

### 4. Evaluate
```bash
python src/training/test.py \
    --backbone resnet18 \
    --temporal bilstm
```

### 5. Explore (Jupyter)
```bash
jupyter notebook notebooks/example_usage.ipynb
```

## File Mappings

| Old Location | New Location | Notes |
|---|---|---|
| `src/prepare_data.py` | `src/data_prep/extract_frames.py` | Renamed and moved |
| `src/dataset.py` | `src/data_prep/dataset.py` | Moved to data module |
| `src/models.py` | `src/models/*.py` | Split into spatial_extractor, temporal_brain, video_classifier |
| `src/train.py` | `src/training/train.py` | Moved to training module |
| `src/test_model.py` | `src/training/test.py` | Moved to training module |
| - | `src/training/metrics.py` | New metrics module |
| - | `src/utils/visualizer.py` | New visualization utilities |
| - | `config.yaml` | New configuration file |
| - | `notebooks/example_usage.ipynb` | New example notebook |

## Configuration

Edit `config.yaml` to customize default settings:
- Frame size, FPS, train/val/test splits
- Model architecture (backbone, temporal type)
- Training hyperparameters
- Batch sizes, learning rates, epochs
- Device settings

## Benefits of New Structure

1. **Cleaner Imports**: More specific, less namespace pollution
2. **Easier Debugging**: Smaller, focused modules
3. **Better Testing**: Unit tests per module
4. **Extensibility**: Add new models, losses, metrics easily
5. **Documentation**: Each module is self-contained
6. **Collaboration**: Clear boundaries for team development

## Questions?

- Check [README.md](../README.md) for usage guide
- See [config.yaml](../config.yaml) for configuration options
- Review [notebooks/example_usage.ipynb](../notebooks/example_usage.ipynb) for examples

---

**Restructure Date**: 2025-05-25
**venv Status**: ✅ Preserved (no changes)
**Experiments**: ✅ All saved models preserved
