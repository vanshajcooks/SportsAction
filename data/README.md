# Dataset Information

This directory contains the tennis action recognition dataset.

## Dataset Structure

```
data/
├── processed/
│   ├── train/          # Training data (70%)
│   ├── val/            # Validation data (15%) 
│   └── test/           # Test data (15%)
├── frames/             # Raw extracted frames
└── features/           # Pre-extracted features
```

## Statistics

- **Total Actions**: 12 tennis action classes
- **Total Samples**: ~39,000 frame sequences
- **Data Size**: ~833MB (frames not in git)
- **Split**: Train/Val/Test (70/15/15)

## Action Classes

1. **backhand** - Standard two-handed backhand
2. **backhand_slice** - Sliced backhand shot
3. **backhand_volley** - Backhand volley
4. **backhand2hands** - Two-handed backhand variant
5. **flat_service** - Flat serve
6. **forehand_flat** - Flat forehand
7. **forehand_openstands** - Open stance forehand
8. **forehand_slice** - Sliced forehand
9. **forehand_volley** - Forehand volley
10. **kick_service** - Kick serve
11. **slice_service** - Slice serve  
12. **smash** - Overhead smash

## Data Preparation

Raw videos are processed into frame sequences using:
```bash
python src/prepare_data.py
```

## Access

- Frame data stored separately (contact team for access)
- Sample frames available in `samples/` directory
- Data loading handled by `src/dataset.py`

## Usage

```python
from src.dataset import TennisActionDataset

dataset = TennisActionDataset('data/processed/train')
```

## Getting Model Files

1. Clone this repository
2. Request access to shared drive: [LINK]
3. Download model files to `experiments/` directories
4. Download dataset to `data/` directory
5. Run: `python src/test_model.py --experiment resnet18_bilstm`