# Experiment Results

This directory contains trained models and experiment results for tennis action recognition.

## Model Performance Summary

| Model | Architecture | Best Accuracy | Validation Loss | Training Time | Notes |
|-------|-------------|---------------|-----------------|---------------|-------|
| densenet121_bilstm | DenseNet121 + BiLSTM | - | - | - | Combination model |
| densenet121_none | DenseNet121 | - | - | - | CNN only |
| inception_v3_none | InceptionV3 | - | - | - | CNN only |
| resnet_bilstm | ResNet + BiLSTM | - | - | - | Combination model |
| resnet18_bilstm | ResNet18 + BiLSTM | - | - | - | Smaller combination |
| resnet18_none | ResNet18 | - | - | - | Lightweight CNN |

## Directory Structure

Each experiment directory contains:
- `best_model.pth` - Best model weights (not in git)
- `training_log.json` - Training metrics and hyperparameters
- `confusion_matrix.png` - Model performance visualization
- `training_curves.png` - Loss/accuracy curves

## Action Classes

The model recognizes the following tennis actions:
- backhand
- backhand_slice  
- backhand_volley
- backhand2hands
- flat_service
- forehand_flat
- forehand_openstands
- forehand_slice
- forehand_volley
- kick_service
- slice_service
- smash

## Usage

To reproduce experiments:
```bash
python src/train.py --model densenet121 --sequence_model bilstm
python src/test_model.py --experiment densenet121_bilstm
```

## Notes

- Model files are stored separately (Git LFS or cloud storage)
- Contact team lead for access to trained weights
- See `requirements.txt` for environment setup