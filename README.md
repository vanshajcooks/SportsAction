# SportsAction: Tennis Action Recognition

AI model for recognizing tennis actions from video sequences using deep learning.

## 🎾 Project Overview

This project implements a deep learning system to classify tennis actions from video frames. The system supports multiple CNN architectures (ResNet, DenseNet, InceptionV3) with optional temporal modeling using BiLSTM.

## 🏗️ Architecture

- **Input**: Sequence of video frames (tennis actions)
- **Models**: CNN feature extraction + optional BiLSTM temporal modeling  
- **Output**: 12 tennis action classes

## 📊 Current Experiments

| Experiment | Model | Status | Best Accuracy | Notes |
|-----------|--------|--------|---------------|-------|
| `densenet121_bilstm` | DenseNet121 + BiLSTM | ✅ Trained | - | Best combination model |
| `resnet18_bilstm` | ResNet18 + BiLSTM | ✅ Trained | - | Lightweight option |
| `inception_v3_none` | InceptionV3 | ✅ Trained | - | Good baseline |
| `resnet_bilstm` | ResNet + BiLSTM | ✅ Trained | - | Standard combination |
| `densenet121_none` | DenseNet121 | ✅ Trained | - | CNN only |
| `resnet18_none` | ResNet18 | ✅ Trained | - | Fastest inference |

## 🚀 Quick Start

### Setup Environment
```bash
pip install -r requirements.txt
```

### Train a Model
```bash
python src/train.py --model resnet18 --sequence_model bilstm --epochs 50
```

### Test Model
```bash
python src/test_model.py --experiment resnet18_bilstm
```

## 📁 Project Structure

```
SportsAction/
├── src/                    # Source code
│   ├── train.py           # Training script
│   ├── test_model.py      # Evaluation script  
│   ├── models.py          # Model architectures
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