# Multimodal Emotion Recognition Across Domains: MOSEI to CAER

This repository contains the code used in the Bachelor thesis project on evaluating how pretrained multimodal emotion recognition models transfer across domains. The project explores zero-shot transfer, fine-tuning, and training from scratch using lightweight GRU-based architectures.

## Project Overview

The study investigates how well models trained on the CMU-MOSEI dataset generalize to other datasets like CAER, CREMA-D, and Movies-28. It evaluates:
- Domain shift challenges
- Performance gains from limited fine-tuning
- Robustness for underrepresented emotions

## Model Architecture

- Visual encoder: ResNet-18 + GRU
- Audio encoder: BiGRU on log-Mel spectrograms
- Text encoder: GloVe or W2V + GRU
- Late fusion using MLP → 7-class softmax output

## Code Overview

- `extract_features.py`: Extracts visual, audio, and text features
- `train_model.py`: Handles training and evaluation of models
- `utils.py`: Utility functions for padding, evaluation metrics, etc.
- `run_experiments.ipynb`: Notebook that runs the 3 training regimes:
  - Zero-shot (MOSEI pretrained)
  - Fine-tuning (10% target domain)
  - Scratch (from random init)

## Metrics

- Macro-F1 (main metric)
- Balanced Accuracy
- Per-class Precision & Recall

## Requirements

- Python 3.10+
- PyTorch
- librosa
- OpenCV
- HuggingFace Transformers (for Whisper)
- tqdm
- scikit-learn

```bash
pip install -r requirements.txt

## Datasets

Public datasets used:
- [CMU-MOSEI](https://github.com/ecfm/CMU-MultimodalSDK.git)
- [CAER](https://caer-dataset.github.io/)
- [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)

Movies-28 is a custom-curated dataset used under fair use (not included).


For academic use only. Cite the thesis or related work when using.


