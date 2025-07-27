# DISTILBERT-FINETUNED-SQUADV2

A production-ready, demo-friendly Question Answering (QA) system using DistilBERT fine-tuned on SQuAD v2. The project provides training, inference, and evaluation pipelines.

## Features

- Fine-tuning and inference with HuggingFace Transformers (DistilBERT)
- Preprocessing and evaluation utilities for SQuAD v2
- Modular, extensible codebase

## Streamlit Demo
Try the live demo here:
👉 https://distilbert-finetuned-squadv2.streamlit.app/

## Project Structure

```text
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker image for app
├── docker-compose.yml    # Compose for resource limits, volume, port
├── src/                  # Source code
│   ├── config.py         # Configuration constants
│   ├── data.py           # Data loading & preprocessing
│   ├── evaluate.py       # Evaluation metrics
│   ├── infer.py          # Inference utilities
│   ├── model.py          # Model & tokenizer loading
│   ├── train.py          # Training pipeline
│   └── ...
└── outputs/              # Model checkpoints, outputs
```

## Quickstart

### 1. Build & Run with Docker

```bash
docker-compose up --build
```

### 2. Local Run (Python 3.10+)

```bash
pip install -r requirements.txt
```

## Training

Edit `src/train.py` and run:

```bash
python src/train.py
```

## Inference

Run the inference script:

```bash
python src/infer.py
```

## Evaluation

Run the evaluation script:

```bash
python src/eval.py
```

## Model

- Default: `ngthcong/distilbert-finetuned-squadv2` (can be changed in `src/config.py`)
- Model artifacts stored in `outputs/final_model/`

## Customization

- Update data loading in `src/data.py`
- Adjust training parameters in `src/train.py`

## Requirements

- Python 3.10+
- Docker (optional)