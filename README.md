# tennis_segmentation - Video Action Recognition

A comprehensive deep learning framework for recognizing sports actions from video sequences using spatial-temporal modeling.

## Project Overview

This project implements a modular system for video action recognition using:
- **Spatial Branch**: CNN backbones (ResNet18, DenseNet121, Inception-v3) for frame-level features
- **Temporal Branch**: BiLSTM, LSTM, GRU, or TCN for temporal modeling
- **Multi-scale Architecture**: Supports flexible combinations of spatial and temporal modules

## Project Structure

```
tennis_segmentation/
├── data/
│   ├── raw/                 # Original .avi video files
│   ├── frames/              # Extracted frames (train/val/test splits)
│   └── annotations/         # Label files
├── src/
│   ├── data_prep/
│   │   ├── extract_frames.py    # Video frame extraction
│   │   └── dataset.py           # PyTorch Dataset loader
│   ├── models/
│   │   ├── spatial_extractor.py # CNN backbones
│   │   ├── temporal_brain.py    # Temporal modules (LSTM, GRU, TCN)
│   │   └── video_classifier.py  # Complete model
│   ├── training/
│   │   ├── train.py             # Main training loop
│   │   ├── test.py              # Evaluation script
│   │   └── metrics.py           # Metrics (F1, accuracy, IoU)
│   └── utils/
│       └── visualizer.py        # Visualization tools
├── notebooks/               # Jupyter notebooks
├── requirements.txt         # Dependencies
├── config.yaml              # Configuration
└── README.md
│   ├── dataset.py         # Data loading
│   └── prepare_data.py    # Data preprocessing
├── experiments/           # Trained models & results
├── data/                  # Dataset (not in git)
└── requirements.txt       # Dependencies
```

## 🎯 Action Classes

The model recognizes these 12 tennis actions:
- Backhand variations (standard, slice, volley, two-handed)
- Forehand variations (flat, open stance, slice, volley)  
- Serves (flat, kick, slice)
- Overhead smash

## 📈 Results

See individual experiment directories in `experiments/` for:
- Training curves and metrics
- Confusion matrices  
- Model performance summaries

## 👥 Team Access

**Large Files Not in Git:**
- Model weights (`*.pth` files)
- Training dataset frames
- Extracted features

**For Access:** Contact team lead for:
- Pre-trained model downloads
- Dataset access links
- Training results and logs

## 🛠️ Development

### Adding New Models
1. Define architecture in `src/models.py`
2. Update training script `src/train.py`  
3. Create experiment directory
4. Update this README

### Training Tips
- Use GPU for faster training
- Monitor with tensorboard (if configured)
- Save checkpoints regularly
- Log hyperparameters and results

## 📋 TODO

- [ ] Add tensorboard logging
- [ ] Implement data augmentation
- [ ] Add model ensemble methods
- [ ] Create inference API
- [ ] Add real-time video processing

## 🤝 Contributing

1. Create feature branch
2. Make changes to source code only
3. Update documentation
4. Test with small dataset
5. Submit pull request

---

**Note**: This repository contains code and documentation. Large model files and datasets are stored separately for team access.